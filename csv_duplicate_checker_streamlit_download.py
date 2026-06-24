# csv_tool_full_enhanced_rate_limited.py
import streamlit as st
import pandas as pd
import io
import re
import csv
import requests
import numpy as np
import time
from io import StringIO
from typing import Tuple

st.set_page_config(page_title="Data Duplicate & Geofilter Checker", layout="wide")

# Initialize DNC list in temporary session memory (wipes on browser refresh)
if "dnc_list" not in st.session_state:
    st.session_state["dnc_list"] = set()

# ------------------ Helpers ------------------
def normalize_phone(x: str) -> str:
    """Normalize phone by keeping digits only and last 10 digits if longer."""
    if pd.isna(x):
        return ""
    s = str(x)
    digits = re.sub(r"\D", "", s)
    if len(digits) > 10:
        digits = digits[-10:]
    return digits

def robust_read_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    try:
        sample = raw.decode("utf-8")
        df = pd.read_csv(StringIO(sample), dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        pass
    try:
        sample = raw.decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","
    try:
        df = pd.read_csv(StringIO(sample), sep=delimiter, dtype=str, keep_default_na=False, engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep=";", dtype=str, keep_default_na=False, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, engine="python", dtype=str, keep_default_na=False)

def read_data_file(uploaded_file, sheet_name=None) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    else:
        return robust_read_csv(uploaded_file)

def select_phone_column_auto(columns: list) -> str:
    lowered = [str(c).lower() for c in columns]
    priority = ["phone", "mobile", "contact", "telephone", "tel", "mob"]
    for p in priority:
        for i, c in enumerate(lowered):
            if p in c:
                return columns[i]
    for i, c in enumerate(lowered):
        if any(k in c for k in ["number", "no.", "num"]):
            return columns[i]
    return None

def build_phone_series(df: pd.DataFrame, col: str) -> pd.Series:
    series = df[col].astype(str).fillna("")
    return series.apply(normalize_phone)

def create_download_button(df: pd.DataFrame, filename: str, label: str = None):
    if label is None:
        label = f"Download {filename}"
    dl_filename = filename if filename.endswith(".csv") else f"{filename.rsplit('.', 1)[0]}.csv"
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return st.download_button(label=label, data=towrite, file_name=dl_filename, mime="text/csv")

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0 
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(pd.to_numeric(lat2, errors="coerce").astype(float))
    lon2_rad = np.radians(pd.to_numeric(lon2, errors="coerce").astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ------------------ Geocoding backends (cached) ------------------
@st.cache_data(show_spinner=False)
def geocode_nominatim(query: str) -> Tuple[float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "csv-tool/1.0"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

@st.cache_data(show_spinner=False)
def geocode_locationiq(query: str, api_key: str) -> Tuple[float, float]:
    url = "https://us1.locationiq.com/v1/search.php"
    params = {"key": api_key, "q": query, "format": "json", "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

@st.cache_data(show_spinner=False)
def geocode_opencage(query: str, api_key: str) -> Tuple[float, float]:
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": query, "key": api_key, "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and data.get("results"):
                res = data["results"][0]["geometry"]
                return float(res["lat"]), float(res["lng"])
    except Exception:
        pass
    return None, None

def geocode_dispatch(query: str, provider: str, api_key: str = None) -> Tuple[float, float]:
    if provider == "nominatim":
        return geocode_nominatim(query)
    elif provider == "locationiq":
        if not api_key:
            return None, None
        return geocode_locationiq(query, api_key)
    elif provider == "opencage":
        if not api_key:
            return None, None
        return geocode_opencage(query, api_key)
    else:
        return None, None

# ------------------ Session state containers ------------------
if "removed_rows" not in st.session_state:
    st.session_state["removed_rows"] = {}
if "removed_single" not in st.session_state:
    st.session_state["removed_single"] = None
if "radius_results" not in st.session_state:
    st.session_state["radius_results"] = {}

# ------------------ App UI ------------------
st.title("Data Duplicate & Geofilter Checker")
st.markdown("""
**Cloud-Ready Version:** Upload your DNC list to apply it to your current session.
Features:
- Supports **.csv and .xlsx files (with multiple sheets)**.
- Compare New vs Old files and Internal Duplicate Removal.
- Upload a **Do Not Call (DNC)** list for your session to auto-remove numbers.
""")

tabs = st.tabs(["Compare New vs Old", "Internal Duplicate Remover", "Filter by Radius", "Settings & DNC"])

# ------------------ Settings & DNC tab ------------------
with tabs[3]:
    st.header("Settings & DNC (Do Not Call) List")
    
    st.subheader("1. Session DNC / Blacklist")
    st.markdown("Load a DNC file from your computer for this session, or manually type numbers. *Note: Data resets when you refresh the page.*")
    
    col1, col2 = st.columns(2)
    with col1:
        manual_dnc = st.text_area("Enter comma-separated phone numbers", help="E.g., 9876543210, 8765432109")
    with col2:
        dnc_upload = st.file_uploader("Upload Master DNC File (.csv or .xlsx)", type=["csv", "xlsx"])
        if dnc_upload:
            dnc_sheet = None
            if dnc_upload.name.endswith('.xlsx'):
                xls_dnc = pd.ExcelFile(dnc_upload)
                dnc_sheet = st.selectbox("Select sheet for DNC", xls_dnc.sheet_names, key="dnc_sheet_sel")
            
            dnc_df = read_data_file(dnc_upload, sheet_name=dnc_sheet)
            auto_col = select_phone_column_auto(dnc_df.columns.tolist())
            dnc_col = st.selectbox("Select phone column in DNC file", dnc_df.columns.tolist(), index=dnc_df.columns.tolist().index(auto_col) if auto_col else 0)

    if st.button("Load / Update DNC List into Memory"):
        new_dnc_phones = set()
        
        # Parse manual text
        if manual_dnc.strip():
            raw_nums = manual_dnc.replace('\n', ',').split(',')
            for num in raw_nums:
                normalized = normalize_phone(num.strip())
                if normalized:
                    new_dnc_phones.add(normalized)
                    
        # Parse uploaded file
        if dnc_upload and dnc_col:
            dnc_series = build_phone_series(dnc_df, dnc_col)
            valid_dnc_series = dnc_series[dnc_series.str.len() > 0]
            new_dnc_phones.update(valid_dnc_series.tolist())
            
        if new_dnc_phones:
            st.session_state["dnc_list"].update(new_dnc_phones)
            st.success(f"Added {len(new_dnc_phones)} numbers to this session. Total DNC size: {len(st.session_state['dnc_list'])}")
        else:
            st.warning("No valid numbers found to add.")
            
    if st.session_state["dnc_list"]:
        st.info(f"Current DNC List contains **{len(st.session_state['dnc_list'])}** numbers in memory.")
        
        # Feature to download the updated DNC list to keep locally
        dnc_export_df = pd.DataFrame(list(st.session_state["dnc_list"]), columns=["Phone"])
        create_download_button(dnc_export_df, "my_master_dnc_list.csv", label="💾 Download Updated DNC List to your PC")
        
        if st.button("Clear Memory"):
            st.session_state["dnc_list"] = set()
            st.rerun()

    st.markdown("---")
    st.subheader("2. Geocoding Provider & API keys")
    geo_provider = st.radio("Choose geocoding provider (default: Nominatim)", ["nominatim", "locationiq", "opencage"], index=0, key="provider_choice")
    loc_api_key = st.text_input("LocationIQ API key", value="", key="loc_key", type="password")
    oc_api_key = st.text_input("OpenCage API key", value="", key="oc_key", type="password")

provider = st.session_state.get("provider_choice", "nominatim")
loc_key = st.session_state.get("loc_key", "")
oc_key = st.session_state.get("oc_key", "")
RATE_LIMITS = {"nominatim": 1.0, "locationiq": 0.5, "opencage": 0.2}

# ------------------ Tab: Compare New vs Old ------------------
with tabs[0]:
    st.header("Compare New vs Old (by phone)")
    st.info("Upload Old and New files. The session's DNC numbers will automatically be filtered out of the New files.")
    col_a, col_b = st.columns([1,1])
    with col_a:
        old_files = st.file_uploader("Upload Old File(s)", type=["csv", "xlsx"], accept_multiple_files=True, key="old_tab_files")
    with col_b:
        new_files = st.file_uploader("Upload New File(s)", type=["csv", "xlsx"], accept_multiple_files=True, key="new_tab_files")

    old_file_configs = {}
    if old_files:
        for f in old_files:
            try:
                sheet = None
                if f.name.endswith('.xlsx'):
                    xls = pd.ExcelFile(f)
                    sheet = st.selectbox(f"Sheet for `{f.name}`", xls.sheet_names, key=f"old_sheet_{f.name}")
                df_ = read_data_file(f, sheet_name=sheet)
                auto = select_phone_column_auto(df_.columns.tolist())
                default_index = df_.columns.tolist().index(auto) if auto in df_.columns.tolist() else 0
                sel_col = st.selectbox(f"Phone column for `{f.name}`", df_.columns.tolist(), index=default_index if default_index < len(df_.columns) else 0, key=f"oldcol_sel_{f.name}")
                old_file_configs[f.name] = {"sheet": sheet, "col": sel_col}
            except Exception as e:
                st.error(f"Can't read {f.name}: {e}")

    new_file_configs = {}
    if new_files:
        for f in new_files:
            try:
                sheet = None
                if f.name.endswith('.xlsx'):
                    xls = pd.ExcelFile(f)
                    sheet = st.selectbox(f"Sheet for `{f.name}`", xls.sheet_names, key=f"new_sheet_{f.name}")
                df_ = read_data_file(f, sheet_name=sheet)
                auto = select_phone_column_auto(df_.columns.tolist())
                default_index = df_.columns.tolist().index(auto) if auto in df_.columns.tolist() else 0
                sel_col = st.selectbox(f"Phone column for `{f.name}`", df_.columns.tolist(), index=default_index if default_index < len(df_.columns) else 0, key=f"newcol_sel_{f.name}")
                new_file_configs[f.name] = {"sheet": sheet, "col": sel_col}
            except Exception as e:
                st.error(f"Can't read {f.name}: {e}")

    if st.button("Process Compare (New vs Old)"):
        if not old_files or not new_files:
            st.error("Please upload at least one Old file and one New file.")
        else:
            old_phones = set()
            for f in old_files:
                config = old_file_configs.get(f.name)
                if not config: continue
                try:
                    df_old = read_data_file(f, sheet_name=config["sheet"])
                    phones = build_phone_series(df_old, config["col"])
                    old_phones.update(phones[phones.str.len() > 0].tolist())
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")
            
            dnc_set = st.session_state.get("dnc_list", set())
            combined_blacklist = old_phones.union(dnc_set)
            
            st.success(f"Filtering against {len(old_phones)} Old phones + {len(dnc_set)} DNC phones.")

            for f in new_files:
                config = new_file_configs.get(f.name)
                if not config: continue
                try:
                    df_new = read_data_file(f, sheet_name=config["sheet"])
                    df_new["_normalized_phone_for_check"] = build_phone_series(df_new, config["col"])
                    mask = df_new["_normalized_phone_for_check"].isin(combined_blacklist) & (df_new["_normalized_phone_for_check"].str.len() > 0)
                    removed_df = df_new.loc[mask].drop(columns=["_normalized_phone_for_check"])
                    cleaned_df = df_new.loc[~mask].drop(columns=["_normalized_phone_for_check"])
                    removed_count = int(mask.sum())
                    
                    st.info(f"For `{f.name}`, removed {removed_count} row(s) matching Old/DNC list.")
                    st.session_state["removed_rows"][f.name] = removed_df.reset_index(drop=True)
                    create_download_button(cleaned_df, f"update_{f.name}", label=f"Download cleaned `{f.name}`")
                except Exception as e:
                    st.error(f"Failed to process `{f.name}`: {e}")

# ------------------ Tab: Internal Duplicate Remover ------------------
with tabs[1]:
    st.header("Internal Duplicate Remover (single file)")
    st.info("Removes duplicates within the file itself AND removes any numbers loaded in your Session DNC list.")
    
    single = st.file_uploader("Upload a single file (.csv or .xlsx)", type=["csv", "xlsx"], key="single_tab_file")
    if single:
        try:
            sheet = None
            if single.name.endswith('.xlsx'):
                xls_single = pd.ExcelFile(single)
                sheet = st.selectbox(f"Sheet for `{single.name}`", xls_single.sheet_names, key=f"single_sheet_sel")
                
            df_single = read_data_file(single, sheet_name=sheet)
            auto = select_phone_column_auto(df_single.columns.tolist())
            default_index = df_single.columns.tolist().index(auto) if auto in df_single.columns.tolist() else 0
            phone_col = st.selectbox("Select phone column", df_single.columns.tolist(), index=default_index if default_index < len(df_single.columns) else 0, key="single_phone_col")
            
            if st.button("Process Internal Deduplication"):
                df_single["_normalized_phone_for_check"] = build_phone_series(df_single, phone_col)
                dup_mask = df_single["_normalized_phone_for_check"].duplicated(keep="first")
                dnc_set = st.session_state.get("dnc_list", set())
                dnc_mask = df_single["_normalized_phone_for_check"].isin(dnc_set)
                final_mask = (dup_mask | dnc_mask) & (df_single["_normalized_phone_for_check"].str.len() > 0)
                
                removed_df = df_single.loc[final_mask].drop(columns=["_normalized_phone_for_check"])
                cleaned_df = df_single.loc[~final_mask].drop(columns=["_normalized_phone_for_check"])
                removed_count = int(final_mask.sum())
                
                st.info(f"Removed {removed_count} row(s) (internal duplicates + DNC hits).")
                st.session_state["removed_single"] = removed_df.reset_index(drop=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    create_download_button(cleaned_df, f"update_{single.name}", label="Download Cleaned File")
                with col2:
                    if removed_count > 0:
                        create_download_button(removed_df, f"removed_{single.name}", label="Download Removed Rows")
        except Exception as e:
            st.error(f"Failed to read/process file: {e}")

# ------------------ Tab: Filter by Radius ------------------
with tabs[2]:
    st.header("Filter rows within a radius of point A")
    
    radius_file = st.file_uploader("Upload file to filter (.csv or .xlsx)", type=["csv", "xlsx"], key="radius_tab_file")
    if radius_file:
        try:
            sheet = None
            if radius_file.name.endswith('.xlsx'):
                xls_radius = pd.ExcelFile(radius_file)
                sheet = st.selectbox(f"Sheet for `{radius_file.name}`", xls_radius.sheet_names, key=f"radius_sheet_sel")
                
            df_radius = read_data_file(radius_file, sheet_name=sheet)
            method = st.radio("Method", ["Latitude & Longitude columns", "Address / Pincode column"], key="radius_method_choice")

            if method == "Latitude & Longitude columns":
                lat_col = st.selectbox("Latitude column", df_radius.columns.tolist(), key="radius_lat_col")
                lon_col = st.selectbox("Longitude column", df_radius.columns.tolist(), key="radius_lon_col")
                ref_lat = st.number_input("Reference latitude", value=30.7333, key="radius_ref_lat")
                ref_lon = st.number_input("Reference longitude", value=76.7794, key="radius_ref_lon")
                radius_km = st.number_input("Radius (km)", value=20.0, step=1.0, key="radius_km_input")

                if st.button("Filter by Radius (Lat/Lon)", key="filter_latlon_button"):
                    distances = haversine_np(ref_lat, ref_lon, df_radius[lat_col], df_radius[lon_col])
                    df_radius['__distance_km__'] = distances
                    inside = df_radius[df_radius['__distance_km__'] <= radius_km].reset_index(drop=True)
                    outside = df_radius[df_radius['__distance_km__'] > radius_km].reset_index(drop=True)
                    st.success(f"Found {len(inside)} inside and {len(outside)} outside.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if len(inside) > 0:
                            create_download_button(inside.drop(columns=['__distance_km__'], errors='ignore'), f"inside_{radius_file.name}")
                    with col2:
                        if len(outside) > 0:
                            create_download_button(outside.drop(columns=['__distance_km__'], errors='ignore'), f"outside_{radius_file.name}")

            else:
                addr_col = st.selectbox("Address column", df_radius.columns.tolist(), key="radius_addr_col")
                pincode_options = [None] + df_radius.columns.tolist()
                pincode_col = st.selectbox("Pincode column (optional)", pincode_options, index=0, key="radius_pin_col")
                ref_lat = st.number_input("Reference latitude", value=30.7333, key="addr_ref_lat_input")
                ref_lon = st.number_input("Reference longitude", value=76.7794, key="addr_ref_lon_input")
                radius_km = st.number_input("Radius (km)", value=20.0, step=1.0, key="addr_radius_km_input")

                if st.button("Filter by Radius (Address)", key="filter_addr_button"):
                    queries = []
                    for idx, row in df_radius.iterrows():
                        addr = "" if pd.isna(row.get(addr_col, "")) else str(row.get(addr_col, "")).strip()
                        pin = None
                        if pincode_col:
                            val = row.get(pincode_col)
                            if pd.notna(val) and str(val).strip() != "":
                                pin = str(val).strip()
                        q = f"{addr} {pin}" if pin else addr
                        queries.append(q)

                    unique_queries = list(dict.fromkeys(queries))
                    st.write(f"Geocoding using `{provider}` provider...")

                    progress = st.progress(0)
                    geocode_map = {}
                    total = len(unique_queries)
                    per_request_delay = RATE_LIMITS.get(provider, 1.0)

                    for i, q in enumerate(unique_queries):
                        latlon = geocode_dispatch(q, provider, api_key=loc_key if provider == "locationiq" else (oc_key if provider == "opencage" else None))
                        geocode_map[q] = latlon
                        time.sleep(per_request_delay)
                        progress.progress(int((i + 1) / total * 100))

                    progress.empty()

                    df_radius['__geocoded_lat__'] = [geocode_map.get(q, (None, None))[0] for q in queries]
                    df_radius['__geocoded_lon__'] = [geocode_map.get(q, (None, None))[1] for q in queries]

                    valid_mask = df_radius['__geocoded_lat__'].notna() & df_radius['__geocoded_lon__'].notna()
                    df_radius.loc[valid_mask, '__distance_km__'] = haversine_np(ref_lat, ref_lon, df_radius.loc[valid_mask, '__geocoded_lat__'], df_radius.loc[valid_mask, '__geocoded_lon__'])
                    df_radius.loc[~valid_mask, '__distance_km__'] = np.nan

                    inside = df_radius[df_radius['__distance_km__'] <= radius_km].reset_index(drop=True)
                    outside = df_radius[(df_radius['__distance_km__'] > radius_km) | (df_radius['__distance_km__'].isna())].reset_index(drop=True)

                    st.success(f"Geocoding complete. {len(inside)} inside, {len(outside)} outside.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if len(inside) > 0:
                            create_download_button(inside.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"inside_{radius_file.name}")
                    with col2:
                        if len(outside) > 0:
                            create_download_button(outside.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"outside_{radius_file.name}")

        except Exception as e:
            st.error(f"Failed to read/process uploaded file: {e}")