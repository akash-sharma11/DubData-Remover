# csv_tool_full.py
import streamlit as st
import pandas as pd
import io
import re
import csv
import requests
import numpy as np
from io import StringIO

st.set_page_config(page_title="CSV Duplicate & Geofilter Checker", layout="wide")

# ------------------ Helpers ------------------

def normalize_phone(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    digits = re.sub(r"\D", "", s)
    # keep last 10 digits if longer
    if len(digits) > 10:
        digits = digits[-10:]
    return digits

def robust_read_csv(uploaded_file) -> pd.DataFrame:
    """
    Read uploaded file robustly: try to sniff delimiter and handle common encodings.
    Returns a DataFrame (all columns as strings).
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    # try simple read first
    try:
        df = pd.read_csv(StringIO(raw.decode("utf-8")), dtype=str, keep_default_na=False)
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
        delimiter = ','

    # try reading with detected delimiter
    try:
        df = pd.read_csv(StringIO(sample), sep=delimiter, dtype=str, keep_default_na=False, engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        # fallback tries
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep=";", dtype=str, keep_default_na=False, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, engine="python", dtype=str, keep_default_na=False)

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
    """Vectorized Haversine. lat1, lon1 are scalars; lat2, lon2 arrays/Series."""
    R = 6371.0  # km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Cache geocoding results. Use st.cache_data to persist across reruns.
@st.cache_data(show_spinner=False)
def geocode_nominatim(query: str):
    """
    Query Nominatim and return (lat, lon) or (None, None).
    We cache results to avoid repeated calls for the same address.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "csv-tool/1.0 (+https://example.com)"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                return lat, lon
    except Exception:
        pass
    return None, None

# ------------------ Initialize session_state containers ------------------

if "removed_rows" not in st.session_state:
    st.session_state["removed_rows"] = {}      # stores removed rows per file (New vs Old)
if "removed_single" not in st.session_state:
    st.session_state["removed_single"] = None  # stores removed rows for internal duplicate file
if "radius_results" not in st.session_state:
    st.session_state["radius_results"] = {}    # store radius results per file: dict of {filename: (inside_df, outside_df)}

# ------------------ App UI ------------------

st.title("CSV Duplicate & Geofilter Checker")
st.markdown("""
**Features**
- Compare New vs Old CSVs by phone number → remove matches from New files, preview removed rows, download cleaned files.  
- Internal duplicate remover (single file).  
- Filter CSV rows within a radius of a reference point (by lat/lon or by address/pincode using Nominatim).  
""")

tabs = st.tabs(["Compare New vs Old", "Internal Duplicate Remover", "Filter by Radius"])

# ------------------ Tab: Compare New vs Old ------------------
with tabs[0]:
    st.header("Compare New vs Old CSVs (by phone)")
    st.info("Upload Old files and New files, select phone columns for each BEFORE processing. After processing you'll get cleaned New files and previews of removed rows.")
    col1, col2 = st.columns([1,1])

    with col1:
        old_files = st.file_uploader("Upload Old CSV file(s)", type=["csv"], accept_multiple_files=True, key="old_files_tab")
    with col2:
        new_files = st.file_uploader("Upload New CSV file(s)", type=["csv"], accept_multiple_files=True, key="new_files_tab")

    old_cols = {}
    if old_files:
        st.markdown("**Select phone column for each Old file**")
        for f in old_files:
            try:
                df_old_sample = robust_read_csv(f)
                old_cols[f.name] = st.selectbox(f"{f.name} phone column", df_old_sample.columns.tolist(), key=f"oldcol_{f.name}")
            except Exception as e:
                st.error(f"Can't read {f.name}: {e}")

    new_cols = {}
    if new_files:
        st.markdown("**Select phone column for each New file**")
        for f in new_files:
            try:
                df_new_sample = robust_read_csv(f)
                new_cols[f.name] = st.selectbox(f"{f.name} phone column", df_new_sample.columns.tolist(), key=f"newcol_{f.name}")
            except Exception as e:
                st.error(f"Can't read {f.name}: {e}")

    process_compare = st.button("Process Compare (New vs Old)")

    if process_compare:
        # basic validation
        if not old_files or not new_files:
            st.error("Please upload at least one Old file and one New file.")
        else:
            # build set of normalized phones from Old files (only current batch)
            old_phones = set()
            st.write("Collecting phone numbers from Old files...")
            for f in old_files:
                try:
                    df_old = robust_read_csv(f)
                    col = old_cols.get(f.name)
                    if col is None:
                        st.warning(f"No phone column selected for {f.name}; skipping.")
                        continue
                    phones = build_phone_series(df_old, col)
                    old_phones.update(phones[phones.str.len() > 0].tolist())
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")

            st.success(f"Collected {len(old_phones)} unique normalized phone(s) from Old files.")
            st.write("---")

            # process each New file
            for f in new_files:
                try:
                    df_new = robust_read_csv(f)
                    col = new_cols.get(f.name)
                    if col is None:
                        st.warning(f"No phone column selected for {f.name}; skipping.")
                        continue
                    df_new["_normalized_phone_for_check"] = build_phone_series(df_new, col)
                    mask = df_new["_normalized_phone_for_check"].isin(old_phones) & (df_new["_normalized_phone_for_check"].str.len() > 0)
                    removed_df = df_new.loc[mask].drop(columns=["_normalized_phone_for_check"])
                    cleaned_df = df_new.loc[~mask].drop(columns=["_normalized_phone_for_check"])

                    removed_count = int(mask.sum())
                    st.info(f"For `{f.name}` — removed {removed_count} row(s) matching Old files.")
                    if removed_count > 0:
                        # store removed rows in session_state for persistent preview
                        st.session_state["removed_rows"][f.name] = removed_df.reset_index(drop=True)
                        # show a small preview
                        st.dataframe(removed_df.head(200))
                    else:
                        # store empty df too, so preview checkbox can show "No rows removed"
                        st.session_state["removed_rows"][f.name] = removed_df.reset_index(drop=True)

                    # provide cleaned download
                    if cleaned_df.empty:
                        st.warning(f"After removing matches, cleaned file for `{f.name}` is empty.")
                    create_download_button(cleaned_df, f"update_{f.name}", label=f"Download cleaned `{f.name}`")
                except Exception as e:
                    st.error(f"Failed to process `{f.name}`: {e}")

    # Preview checkboxes (outside processing block so they persist and respond without recompute)
    if st.session_state["removed_rows"]:
        st.write("---")
        st.markdown("**Preview removed rows (from last Process Compare run):**")
        for fname, df_removed in st.session_state["removed_rows"].items():
            # show checkbox for each file
            checkbox_key = f"preview_removed_{fname}"
            if st.checkbox(f"Show removed rows for `{fname}`", key=checkbox_key):
                if df_removed is None or df_removed.empty:
                    st.info("No rows were removed for this file.")
                else:
                    st.dataframe(df_removed.head(500))
                    # also allow download of removed rows
                    create_download_button(df_removed, f"removed_rows_{fname}", label=f"Download removed rows `{fname}`")

# ------------------ Tab: Internal Duplicate Remover ------------------
with tabs[1]:
    st.header("Internal Duplicate Remover (single file)")
    single = st.file_uploader("Upload a single CSV", type=["csv"], key="single_file_tab")
    if single:
        try:
            df_single = robust_read_csv(single)
            st.write(f"Columns: {', '.join(df_single.columns.tolist()[:20])}{'...' if len(df_single.columns)>20 else ''}")
            phone_col = st.selectbox("Select phone column", df_single.columns.tolist(), key="single_phone_col")
            df_single["_normalized_phone_for_check"] = build_phone_series(df_single, phone_col)
            dup_mask = df_single["_normalized_phone_for_check"].duplicated(keep="first") & (df_single["_normalized_phone_for_check"].str.len() > 0)
            removed_df = df_single.loc[dup_mask].drop(columns=["_normalized_phone_for_check"])
            cleaned_df = df_single.loc[~dup_mask].drop(columns=["_normalized_phone_for_check"])
            removed_count = int(dup_mask.sum())
            st.info(f"Removed {removed_count} duplicate row(s) in this file.")
            # store and preview
            st.session_state["removed_single"] = removed_df.reset_index(drop=True)
            if removed_count > 0:
                st.dataframe(removed_df.head(200))
                create_download_button(removed_df, f"removed_duplicates_{single.name}", label="Download removed duplicates")
            # cleaned download
            create_download_button(cleaned_df, f"update_{single.name}", label="Download cleaned file (duplicates removed)")
        except Exception as e:
            st.error(f"Failed to read/process file: {e}")

# ------------------ Tab: Filter by Radius ------------------
with tabs[2]:
    st.header("Filter CSV rows within a radius of point A")
    st.info("Filter rows that are within a radius (km) of a reference point. Use lat/lon columns or address (+ optional pincode) with Nominatim geocoding.")
    radius_file = st.file_uploader("Upload CSV to filter", type=["csv"], key="radius_file_tab")

    if radius_file:
        try:
            df_radius = robust_read_csv(radius_file)
            st.write(f"Columns: {', '.join(df_radius.columns.tolist()[:20])}{'...' if len(df_radius.columns)>20 else ''}")
            method = st.radio("Method", ["Latitude & Longitude columns", "Address / Pincode column"], key="radius_method")

            if method == "Latitude & Longitude columns":
                lat_col = st.selectbox("Latitude column", df_radius.columns.tolist(), key="latcol")
                lon_col = st.selectbox("Longitude column", df_radius.columns.tolist(), key="loncol")
                ref_lat = st.number_input("Reference latitude", value=30.7333, format="%.6f", key="ref_lat")
                ref_lon = st.number_input("Reference longitude", value=76.7794, format="%.6f", key="ref_lon")
                radius_km = st.number_input("Radius (km)", value=20.0, step=1.0, key="radius_km")

                if st.button("Filter by Radius (Lat/Lon)", key="filter_latlon_btn"):
                    # calculate distances (handle missing / non-numeric gracefully)
                    try:
                        distances = haversine_np(ref_lat, ref_lon, pd.to_numeric(df_radius[lat_col], errors="coerce"), pd.to_numeric(df_radius[lon_col], errors="coerce"))
                        df_radius['__distance_km__'] = distances
                        inside = df_radius[df_radius['__distance_km__'] <= radius_km].reset_index(drop=True)
                        outside = df_radius[df_radius['__distance_km__'] > radius_km].reset_index(drop=True)
                        st.success(f"Found {len(inside)} row(s) inside radius and {len(outside)} row(s) outside radius.")
                        # store in session_state
                        st.session_state["radius_results"][radius_file.name] = (inside, outside)
                        # downloads
                        if len(inside) > 0:
                            create_download_button(inside.drop(columns=['__distance_km__'], errors='ignore'), f"inside_radius_{radius_file.name}", label=f"Download inside_radius_{radius_file.name}")
                        if len(outside) > 0:
                            create_download_button(outside.drop(columns=['__distance_km__'], errors='ignore'), f"outside_radius_{radius_file.name}", label=f"Download outside_radius_{radius_file.name}")
                        # preview
                        if len(inside) > 0:
                            st.subheader("Preview: rows inside radius")
                            st.dataframe(inside.head(500))
                        if len(outside) > 0:
                            st.subheader("Preview: rows outside radius")
                            st.dataframe(outside.head(500))
                    except Exception as e:
                        st.error(f"Failed to compute distances: {e}")

            else:
                # Address / Pincode branch
                addr_col = st.selectbox("Address column", df_radius.columns.tolist(), key="addrcol")
                # include None + columns for optional pincode selection
                pincode_options = [None] + df_radius.columns.tolist()
                pincode_col = st.selectbox("Pincode column (optional)", pincode_options, index=0, key="pincodecol")
                ref_lat = st.number_input("Reference latitude", value=30.7333, format="%.6f", key="addr_ref_lat")
                ref_lon = st.number_input("Reference longitude", value=76.7794, format="%.6f", key="addr_ref_lon")
                radius_km = st.number_input("Radius (km)", value=20.0, step=1.0, key="addr_radius_km")

                if st.button("Filter by Radius (Address)", key="filter_addr_btn"):
                    # build address strings and geocode unique addresses
                    st.write("Geocoding addresses (using Nominatim). Results are cached to reduce repeated calls.")
                    addresses = []
                    queries = []
                    for idx, row in df_radius.iterrows():
                        addr = "" if pd.isna(row.get(addr_col, "")) else str(row.get(addr_col, "")).strip()
                        pin = None
                        if pincode_col:
                            val = row.get(pincode_col)
                            if pd.notna(val) and str(val).strip() != "":
                                pin = str(val).strip()
                        if pin:
                            q = f"{addr} {pin}"
                        else:
                            q = addr
                        addresses.append((idx, q))
                        queries.append(q)

                    # geocode unique queries only
                    unique_queries = list(dict.fromkeys(queries))  # preserves order
                    st.write(f"Geocoding {len(unique_queries)} unique address strings...")
                    geocode_map = {}
                    progress_bar = st.progress(0)
                    for i, q in enumerate(unique_queries):
                        lat_lon = geocode_nominatim(q)
                        geocode_map[q] = lat_lon
                        # progress update
                        progress_bar.progress(int((i + 1) / len(unique_queries) * 100))
                    progress_bar.empty()

                    # assemble lat/lon arrays in same order as df_radius
                    lat_list = []
                    lon_list = []
                    for idx, q in addresses:
                        lat, lon = geocode_map.get(q, (None, None))
                        lat_list.append(lat)
                        lon_list.append(lon)

                    df_radius['__geocoded_lat__'] = lat_list
                    df_radius['__geocoded_lon__'] = lon_list

                    # filter rows with missing lat/lon: considered 'outside' (or you can treat separately)
                    # compute distance for rows with valid coords
                    valid_mask = df_radius['__geocoded_lat__'].notna() & df_radius['__geocoded_lon__'].notna()
                    if valid_mask.sum() == 0:
                        st.warning("No valid geocoded coordinates found for any row. Check address/pincode data or try smaller batch.")
                    # compute distances (non-valid will become NaN)
                    df_radius.loc[valid_mask, '__distance_km__'] = haversine_np(ref_lat, ref_lon, df_radius.loc[valid_mask, '__geocoded_lat__'], df_radius.loc[valid_mask, '__geocoded_lon__'])
                    df_radius.loc[~valid_mask, '__distance_km__'] = np.nan

                    inside = df_radius[df_radius['__distance_km__'] <= radius_km].reset_index(drop=True)
                    outside = df_radius[(df_radius['__distance_km__'] > radius_km) | (df_radius['__distance_km__'].isna())].reset_index(drop=True)

                    st.success(f"Found {len(inside)} row(s) inside radius and {len(outside)} row(s) outside or not-geocoded.")
                    # store results
                    st.session_state["radius_results"][radius_file.name] = (inside, outside)

                    # downloads
                    if len(inside) > 0:
                        create_download_button(inside.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"inside_radius_{radius_file.name}", label=f"Download inside_radius_{radius_file.name}")
                    if len(outside) > 0:
                        create_download_button(outside.drop(columns=['__geocoded_lat__','__geocoded_lon__','__distance_km__'], errors='ignore'), f"outside_radius_{radius_file.name}", label=f"Download outside_radius_{radius_file.name}")

                    # previews
                    if len(inside) > 0:
                        st.subheader("Preview: rows inside radius")
                        st.dataframe(inside.head(500))
                    if len(outside) > 0:
                        st.subheader("Preview: rows outside radius (includes not-geocoded rows)")
                        st.dataframe(outside.head(500))

        except Exception as e:
            st.error(f"Failed to read/process uploaded CSV: {e}")

    # Show previous radius results and let user preview / download again
    if st.session_state["radius_results"]:
        st.write("---")
        st.markdown("**Previous radius results (from this session):**")
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

# ------------------ Footer / Notes ------------------

st.markdown("---")
st.markdown("**Notes & recommendations**")
st.markdown("""
- Geocoding uses the free Nominatim service (OpenStreetMap). Be kind: cache results (this app caches by query), avoid heavy rapid batches, and consider obtaining your own geocoding key for high-volume work.  
- Address geocoding can be imperfect — review `outside` results that are 'not geocoded' (missing coords).  
- Phone normalization keeps the **last 10 digits** (useful for matching Indian mobile numbers with country code). Change `normalize_phone` if you need a different rule.  
- If you have very large files (10k+ rows), consider splitting into smaller batches to avoid geocoding rate limits or switch to a paid geocoding provider (LocationIQ, OpenCage, Google) for bulk operations.
""")
