# csv_tool_full_enhanced.py
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

st.set_page_config(page_title="CSV Duplicate & Geofilter Checker (Enhanced)", layout="wide")

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
    """Try various ways to robustly read csv/tsv files and strip headers."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    # try utf-8 simple read
    try:
        sample = raw.decode("utf-8")
        df = pd.read_csv(StringIO(sample), dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        pass

    # sniff delimiter
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
        # fallback trying semicolon or python engine
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep=";", dtype=str, keep_default_na=False, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, engine="python", dtype=str, keep_default_na=False)

def select_phone_column_auto(columns: list) -> str:
    """Auto-select a phone-like column name if present in columns list."""
    lowered = [c.lower() for c in columns]
    priority = ["phone", "mobile", "contact", "telephone", "tel", "mob"]
    for p in priority:
        for i, c in enumerate(lowered):
            if p in c:
                return columns[i]
    # fallback: return first column that looks numeric-ish in name
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
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return st.download_button(label=label, data=towrite, file_name=filename, mime="text/csv")

def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized Haversine. lat1/lon1 scalars, lat2/lon2 arrays or Series."""
    R = 6371.0  # km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(pd.to_numeric(lat2, errors="coerce").astype(float))
    lon2_rad = np.radians(pd.to_numeric(lon2, errors="coerce").astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ------------------ Geocoding backends ------------------

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
    # LocationIQ endpoint (forward geocoding)
    # Docs: https://locationiq.com/docs
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
    """Dispatch geocoding to chosen provider; caches used via specific cached functions above."""
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

st.title("CSV Duplicate & Geofilter Checker — Enhanced")
st.markdown("""
Features:
- Compare New vs Old CSVs by phone (auto-detect phone columns, or choose manually).  
- Internal duplicate remover (single file).  
- Filter by radius using lat/lon or address+pincode with geocoding.  
- Choose geocoding provider (Nominatim, LocationIQ, OpenCage) and supply API key if needed.  
- Caches geocoding responses and shows progress for batch geocoding.
""")

tabs = st.tabs(["Compare New vs Old", "Internal Duplicate Remover", "Filter by Radius", "Settings"])

# ------------------ Settings tab ------------------
with tabs[3]:
    st.header("Settings")
    st.markdown("**Geocoding provider & API keys**")
    geo_provider = st.radio("Choose geocoding provider (default: Nominatim)", ["nominatim", "locationiq", "opencage"], index=0, key="provider_choice")
    st.markdown("If you choose LocationIQ or OpenCage, paste your API key below (recommended for large datasets).")
    loc_api_key = st.text_input("LocationIQ API key", value="", key="loc_key")
    oc_api_key = st.text_input("OpenCage API key", value="", key="oc_key")
    st.markdown("""
    Recommendations:
    - For small quick tests, **Nominatim** (free) is fine. Respect 1 request/sec.
    - For larger batches, use **LocationIQ** (10k/month free tier) or **OpenCage** (free tier), enter API key above.
    """)

# Ensure provider selection is available in session for other tabs
provider = st.session_state.get("provider_choice", "nominatim")
loc_key = st.session_state.get("loc_key", "")
oc_key = st.session_state.get("oc_key", "")

# ------------------ Tab: Compare New vs Old ------------------
with tabs[0]:
    st.header("Compare New vs Old (by phone)")
    st.info("Upload Old and New CSVs. The app tries to auto-select a phone column if it exists (you can still change it). Select columns BEFORE pressing Process.")
    col_a, col_b = st.columns([1,1])
    with col_a:
        old_files = st.file_uploader("Upload Old CSV(s)", type=["csv"], accept_multiple_files=True, key="old_tab_files")
    with col_b:
        new_files = st.file_uploader("Upload New CSV(s)", type=["csv"], accept_multiple_files=True, key="new_tab_files")

    old_cols = {}
    if old_files:
        st.markdown("**Old files — choose phone column (auto-selected if possible)**")
        for f in old_files:
            try:
                df_ = robust_read_csv(f)
                auto = select_phone_column_auto(df_.columns.tolist())
                default_index = df_.columns.tolist().index(auto) if auto in df_.columns.tolist() else 0
                # use selectbox with index set to auto if found
                old_cols[f.name] = st.selectbox(f"{f.name} phone column", df_.columns.tolist(), index=default_index if default_index < len(df_.columns) else 0, key=f"oldcol_sel_{f.name}")
            except Exception as e:
                st.error(f"Can't read {f.name}: {e}")

    new_cols = {}
    if new_files:
        st.markdown("**New files — choose phone column (auto-selected if possible)**")
        for f in new_files:
            try:
                df_ = robust_read_csv(f)
                auto = select_phone_column_auto(df_.columns.tolist())
                default_index = df_.columns.tolist().index(auto) if auto in df_.columns.tolist() else 0
                new_cols[f.name] = st.selectbox(f"{f.name} phone column", df_.columns.tolist(), index=default_index if default_index < len(df_.columns) else 0, key=f"newcol_sel_{f.name}")
            except Exception as e:
                st.error(f"Can't read {f.name}: {e}")

    if "removed_rows" not in st.session_state:
        st.session_state["removed_rows"] = {}

    if st.button("Process Compare (New vs Old)"):
        if not old_files or not new_files:
            st.error("Please upload at least one Old file and one New file.")
        else:
            # build old phones set
            old_phones = set()
            st.write("Collecting phones from Old files...")
            for f in old_files:
                try:
                    df_old = robust_read_csv(f)
                    col = old_cols.get(f.name)
                    if not col:
                        st.warning(f"No phone column selected for {f.name}; skipping.")
                        continue
                    phones = build_phone_series(df_old, col)
                    old_phones.update(phones[phones.str.len() > 0].tolist())
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")
            st.success(f"Collected {len(old_phones)} unique normalized phone(s).")
            st.write("---")

            for f in new_files:
                try:
                    df_new = robust_read_csv(f)
                    col = new_cols.get(f.name)
                    if not col:
                        st.warning(f"No phone column selected for {f.name}; skipping.")
                        continue
                    df_new["_normalized_phone_for_check"] = build_phone_series(df_new, col)
                    mask = df_new["_normalized_phone_for_check"].isin(old_phones) & (df_new["_normalized_phone_for_check"].str.len() > 0)
                    removed_df = df_new.loc[mask].drop(columns=["_normalized_phone_for_check"])
                    cleaned_df = df_new.loc[~mask].drop(columns=["_normalized_phone_for_check"])
                    removed_count = int(mask.sum())
                    st.info(f"For `{f.name}` removed {removed_count} row(s) matching Old files.")
                    # store removed rows in session for preview
                    st.session_state["removed_rows"][f.name] = removed_df.reset_index(drop=True)
                    # preview small sample
                    if removed_count > 0:
                        st.dataframe(removed_df.head(200))
                    # downloads
                    create_download_button(cleaned_df, f"update_{f.name}", label=f"Download cleaned `{f.name}`")
                except Exception as e:
                    st.error(f"Failed to process `{f.name}`: {e}")

    # Previews from previous process run
    if st.session_state["removed_rows"]:
        st.write("---")
        st.markdown("**Removed rows from last run (preview & download)**")
        for fname, df_removed in st.session_state["removed_rows"].items():
            ck = st.checkbox(f"Show removed rows for `{fname}`", key=f"preview_removed_{fname}")
            if ck:
                if df_removed is None or df_removed.empty:
                    st.info("No rows removed for this file.")
                else:
                    st.dataframe(df_removed.head(500))
                    create_download_button(df_removed, f"removed_rows_{fname}", label=f"Download removed rows `{fname}`")

# ------------------ Tab: Internal Duplicate Remover ------------------
with tabs[1]:
    st.header("Internal Duplicate Remover (single file)")
    single = st.file_uploader("Upload a single CSV", type=["csv"], key="single_tab_file")
    if single:
        try:
            df_single = robust_read_csv(single)
            st.write(f"Columns: {', '.join(df_single.columns.tolist()[:20])}{'...' if len(df_single.columns)>20 else ''}")
            auto = select_phone_column_auto(df_single.columns.tolist())
            default_index = df_single.columns.tolist().index(auto) if auto in df_single.columns.tolist() else 0
            phone_col = st.selectbox("Select phone column", df_single.columns.tolist(), index=default_index if default_index < len(df_single.columns) else 0, key="single_phone_col")
            df_single["_normalized_phone_for_check"] = build_phone_series(df_single, phone_col)
            dup_mask = df_single["_normalized_phone_for_check"].duplicated(keep="first") & (df_single["_normalized_phone_for_check"].str.len() > 0)
            removed_df = df_single.loc[dup_mask].drop(columns=["_normalized_phone_for_check"])
            cleaned_df = df_single.loc[~dup_mask].drop(columns=["_normalized_phone_for_check"])
            removed_count = int(dup_mask.sum())
            st.info(f"Removed {removed_count} duplicate row(s) in this file.")
            st.session_state["removed_single"] = removed_df.reset_index(drop=True)
            if removed_count > 0:
                st.dataframe(removed_df.head(200))
                create_download_button(removed_df, f"removed_duplicates_{single.name}", label="Download removed duplicates")
            create_download_button(cleaned_df, f"update_{single.name}", label="Download cleaned file (duplicates removed)")
        except Exception as e:
            st.error(f"Failed to read/process file: {e}")

# ------------------ Tab: Filter by Radius ------------------
with tabs[2]:
    st.header("Filter rows within a radius of point A")
    st.info("You can use lat/lon columns, or address + optional pincode. Choose geocoding provider in Settings tab.")

    radius_file = st.file_uploader("Upload CSV to filter", type=["csv"], key="radius_tab_file")
    if radius_file:
        try:
            df_radius = robust_read_csv(radius_file)
            st.write(f"Columns: {', '.join(df_radius.columns.tolist()[:20])}{'...' if len(df_radius.columns)>20 else ''}")
            method = st.radio("Method", ["Latitude & Longitude columns", "Address / Pincode column"], key="radius_method_choice")

            if method == "Latitude & Longitude columns":
                lat_col = st.selectbox("Latitude column", df_radius.columns.tolist(), key="radius_lat_col")
                lon_col = st.selectbox("Longitude column", df_radius.columns.tolist(), key="radius_lon_col")
                ref_lat = st.number_input("Reference latitude", value=30.7333, key="radius_ref_lat")
                ref_lon = st.number_input("Reference longitude", value=76.7794, key="radius_ref_lon")
                radius_km = st.number_input("Radius (km)", value=20.0, step=1.0, key="radius_km_input")

                if st.button("Filter by Radius (Lat/Lon)", key="filter_latlon_button"):
                    try:
                        distances = haversine_np(ref_lat, ref_lon, df_radius[lat_col], df_radius[lon_col])
                        df_radius['__distance_km__'] = distances
                        inside = df_radius[df_radius['__distance_km__'] <= radius_km].reset_index(drop=True)
                        outside = df_radius[df_radius['__distance_km__'] > radius_km].reset_index(drop=True)
                        st.success(f"Found {len(inside)} row(s) inside and {len(outside)} row(s) outside radius.")
                        st.session_state["radius_results"][radius_file.name] = (inside, outside)
                        if len(inside) > 0:
                            create_download_button(inside.drop(columns=['__distance_km__'], errors='ignore'), f"inside_radius_{radius_file.name}", label=f"Download inside_radius_{radius_file.name}")
                        if len(outside) > 0:
                            create_download_button(outside.drop(columns=['__distance_km__'], errors='ignore'), f"outside_radius_{radius_file.name}", label=f"Download outside_radius_{radius_file.name}")
                        if len(inside) > 0:
                            st.subheader("Preview: inside radius")
                            st.dataframe(inside.head(500))
                        if len(outside) > 0:
                            st.subheader("Preview: outside radius")
                            st.dataframe(outside.head(500))
                    except Exception as e:
                        st.error(f"Failed to compute distances: {e}")

            else:
                # Address / Pincode branch
                addr_col = st.selectbox("Address column", df_radius.columns.tolist(), key="radius_addr_col")
                pincode_options = [None] + df_radius.columns.tolist()
                pincode_col = st.selectbox("Pincode column (optional)", pincode_options, index=0, key="radius_pin_col")
                ref_lat = st.number_input("Reference latitude", value=30.7333, key="addr_ref_lat_input")
                ref_lon = st.number_input("Reference longitude", value=76.7794, key="addr_ref_lon_input")
                radius_km = st.number_input("Radius (km)", value=20.0, step=1.0, key="addr_radius_km_input")

                if st.button("Filter by Radius (Address)", key="filter_addr_button"):
                    # prepare address queries and unique list
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

                    unique_queries = list(dict.fromkeys(queries))  # preserve order, unique
                    st.write(f"Geocoding {len(unique_queries)} unique address strings using `{provider}` provider...")

                    # progress bar + timing
                    progress = st.progress(0)
                    geocode_map = {}
                    total = len(unique_queries)
                    start_time = time.time()

                    for i, q in enumerate(unique_queries):
                        if provider == "nominatim":
                            # respect Nominatim usage policy ~1 req/sec
                            latlon = geocode_dispatch(q, "nominatim")
                            geocode_map[q] = latlon
                            # small sleep to be polite / respect rate-limit
                            time.sleep(1.0)
                        elif provider == "locationiq":
                            latlon = geocode_dispatch(q, "locationiq", api_key=loc_key)
                            geocode_map[q] = latlon
                            # slight pause to avoid hitting free-tier burst limits
                            time.sleep(0.1)
                        elif provider == "opencage":
                            latlon = geocode_dispatch(q, "opencage", api_key=oc_key)
                            geocode_map[q] = latlon
                            time.sleep(0.1)
                        else:
                            geocode_map[q] = (None, None)

                        # update progress
                        elapsed = time.time() - start_time
                        completed = i + 1
                        progress.progress(int(completed / total * 100))

                    progress.empty()

                    # map lat/lon back to df_radius rows
                    lat_list = []
                    lon_list = []
                    for q in queries:
                        lat, lon = geocode_map.get(q, (None, None))
                        lat_list.append(lat)
                        lon_list.append(lon)

                    df_radius['__geocoded_lat__'] = lat_list
                    df_radius['__geocoded_lon__'] = lon_list

                    valid_mask = df_radius['__geocoded_lat__'].notna() & df_radius['__geocoded_lon__'].notna()
                    if valid_mask.sum() == 0:
                        st.warning("No rows could be geocoded successfully. Check addresses / API limits.")
                    # compute distances for valid rows
                    df_radius.loc[valid_mask, '__distance_km__'] = haversine_np(ref_lat, ref_lon, df_radius.loc[valid_mask, '__geocoded_lat__'], df_radius.loc[valid_mask, '__geocoded_lon__'])
                    df_radius.loc[~valid_mask, '__distance_km__'] = np.nan

                    inside = df_radius[df_radius['__distance_km__'] <= radius_km].reset_index(drop=True)
                    outside = df_radius[(df_radius['__distance_km__'] > radius_km) | (df_radius['__distance_km__'].isna())].reset_index(drop=True)

                    st.success(f"Geocoding complete. Found {len(inside)} inside, {len(outside)} outside (includes not-geocoded).")
                    st.session_state["radius_results"][radius_file.name] = (inside, outside)

                    # downloads
                    if len(inside) > 0:
                        create_download_button(inside.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"inside_radius_{radius_file.name}", label=f"Download inside_radius_{radius_file.name}")
                    if len(outside) > 0:
                        create_download_button(outside.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"outside_radius_{radius_file.name}", label=f"Download outside_radius_{radius_file.name}")

                    # previews
                    if len(inside) > 0:
                        st.subheader("Preview: inside radius")
                        st.dataframe(inside.head(500))
                    if len(outside) > 0:
                        st.subheader("Preview: outside radius (includes not-geocoded)")
                        st.dataframe(outside.head(500))

        except Exception as e:
            st.error(f"Failed to read/process uploaded CSV: {e}")

    # Show previous radius results and let user preview / download again
    if st.session_state["radius_results"]:
        st.write("---")
        st.markdown("**Saved radius results (this session)**")
        for fname, (inside_df, outside_df) in st.session_state["radius_results"].items():
            st.markdown(f"**{fname}** — inside: {len(inside_df)} rows, outside: {len(outside_df)} rows")
            if st.checkbox(f"Show inside rows for `{fname}`", key=f"show_inside_{fname}"):
                if not inside_df.empty:
                    st.dataframe(inside_df.head(500))
                    create_download_button(inside_df.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"inside_radius_{fname}", label=f"Download inside_radius_{fname}")
                else:
                    st.info("No rows inside radius for this file.")
            if st.checkbox(f"Show outside rows for `{fname}`", key=f"show_outside_{fname}"):
                if not outside_df.empty:
                    st.dataframe(outside_df.head(500))
                    create_download_button(outside_df.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"outside_radius_{fname}", label=f"Download outside_radius_{fname}")
                else:
                    st.info("No rows outside radius for this file.")

# ------------------ Footer Notes ------------------
st.markdown("---")
st.markdown("**Notes & recommendations**")
st.markdown("""
- Nominatim is free but rate-limited (please respect ~1 request/sec). This app enforces ~1s sleep between Nominatim calls when geocoding batches.
- For higher volume geocoding, use LocationIQ or OpenCage (enter API keys in Settings). The app uses their APIs if keys are provided.
- Geocoding quality varies by address formatting. You may want to pre-clean addresses (normalize city names, include pincode) for better accuracy.
- Phone normalization keeps last 10 digits; change `normalize_phone` if you need different behaviour.
""")
