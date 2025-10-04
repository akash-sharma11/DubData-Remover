import streamlit as st
import pandas as pd
import io
import re
import requests
import numpy as np

st.set_page_config(page_title="CSV Duplicate & Geofilter Checker", layout="wide")

# ------------------ Helper Functions ------------------

def normalize_phone(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    digits = re.sub(r"\D", "", s)
    if len(digits) > 10:
        digits = digits[-10:]
    return digits

def read_csv_file(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    except:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep=";", dtype=str, keep_default_na=False)
        except:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, engine="python", dtype=str, keep_default_na=False)

def build_phone_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).fillna("").apply(normalize_phone)

def create_download_link(df: pd.DataFrame, filename: str):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return st.download_button(label=f"Download cleaned: {filename}", data=towrite, file_name=filename, mime="text/csv")

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

@st.cache_data(show_spinner=False)
def geocode_address(address, pincode=None):
    query = str(address)
    if pincode:
        query += f" {pincode}"
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    try:
        response = requests.get(url, params=params, headers={"User-Agent": "streamlit-app"})
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data['lat']), float(data['lon'])
    except:
        pass
    return None, None

# ------------------ App Layout ------------------

st.title("CSV Duplicate & Geofilter Checker")
st.markdown("""
- Compare New vs Old CSVs by phone number and remove duplicates.  
- Internal duplicate remover (single file).  
- Filter CSV rows within a given radius of a reference point (lat/lon or address).  
""")

tabs = st.tabs(["Compare New vs Old", "Internal Duplicate Remover", "Filter by Radius"])

# ------------------ Tab 1: Compare New vs Old ------------------
with tabs[0]:
    st.header("Compare New vs Old CSVs")
    col1, col2 = st.columns([1,1])
    with col1:
        old_files = st.file_uploader("Upload Old CSV file(s)", type=["csv"], accept_multiple_files=True, key="old_files_tab")
    with col2:
        new_files = st.file_uploader("Upload New CSV file(s)", type=["csv"], accept_multiple_files=True, key="new_files_tab")

    # ------------------ Select Phone Columns ------------------
    old_cols_dict = {}
    if old_files:
        st.markdown("**Select phone column for each Old file**")
        for f in old_files:
            df_old = read_csv_file(f)
            old_cols_dict[f.name] = st.selectbox(f"{f.name} phone column", df_old.columns.tolist(), key=f"oldcol_{f.name}")

    new_cols_dict = {}
    if new_files:
        st.markdown("**Select phone column for each New file**")
        for f in new_files:
            df_new = read_csv_file(f)
            new_cols_dict[f.name] = st.selectbox(f"{f.name} phone column", df_new.columns.tolist(), key=f"newcol_{f.name}")

    process_btn = st.button("Process Compare (New vs Old)")

    if process_btn:
        if not old_files or not new_files:
            st.error("Upload at least one Old file and one New file.")
        else:
            old_phones_set = set()
            st.write("Reading Old files and collecting phone numbers...")
            for f in old_files:
                df_old = read_csv_file(f)
                phones = build_phone_series(df_old, old_cols_dict[f.name])
                old_phones_set.update(phones[phones.str.len()>0].tolist())
            st.success(f"Total unique normalized phone numbers in Old files: {len(old_phones_set)}")

            for f in new_files:
                df_new = read_csv_file(f)
                df_new["_normalized_phone_for_check"] = build_phone_series(df_new, new_cols_dict[f.name])
                mask_in_old = df_new["_normalized_phone_for_check"].isin(old_phones_set) & (df_new["_normalized_phone_for_check"].str.len()>0)
                removed_count = mask_in_old.sum()
                st.write(f"**{removed_count} rows removed from `{f.name}` that match Old files.**")
                cleaned_df = df_new.loc[~mask_in_old].drop(columns=["_normalized_phone_for_check"])
                create_download_link(cleaned_df, f"update_{f.name}")
                if removed_count > 0:
                    st.dataframe(df_new.loc[mask_in_old].drop(columns=["_normalized_phone_for_check"]).head(200))

# ------------------ Tab 2: Internal Duplicate Remover ------------------
with tabs[1]:
    st.header("Internal Duplicate Remover")
    single_file = st.file_uploader("Upload a single CSV", type=["csv"], key="single_file_tab")
    if single_file:
        df_single = read_csv_file(single_file)
        phone_col = st.selectbox("Select phone column", df_single.columns.tolist(), key="single_phone_col_tab")
        df_single["_normalized_phone_for_check"] = build_phone_series(df_single, phone_col)
        duplicated_mask = df_single["_normalized_phone_for_check"].duplicated(keep="first") & (df_single["_normalized_phone_for_check"].str.len()>0)
        removed_count = duplicated_mask.sum()
        st.write(f"**{removed_count} duplicate rows removed**")
        cleaned_df = df_single.loc[~duplicated_mask].drop(columns=["_normalized_phone_for_check"])
        create_download_link(cleaned_df, f"update_{single_file.name}")
        if removed_count > 0:
            st.dataframe(df_single.loc[duplicated_mask].drop(columns=["_normalized_phone_for_check"]).head(200))

# ------------------ Tab 3: Filter by Radius ------------------
with tabs[2]:
    st.header("Filter CSV rows within a radius of point A")
    radius_file = st.file_uploader("Upload CSV to filter", type=["csv"], key="radius_file_tab")
    if radius_file:
        df_radius = read_csv_file(radius_file)
        st.write(f"Columns: {', '.join(df_radius.columns.tolist()[:20])}{'...' if len(df_radius.columns)>20 else ''}")
        
        method = st.radio("Filter method:", ["Latitude & Longitude columns", "Address / Pincode column"], key="radius_method")
        
        if method == "Latitude & Longitude columns":
            lat_col = st.selectbox("Latitude column", df_radius.columns.tolist(), key="lat_col")
            lon_col = st.selectbox("Longitude column", df_radius.columns.tolist(), key="lon_col")
            ref_lat = st.number_input("Reference point latitude", value=0.0)
            ref_lon = st.number_input("Reference point longitude", value=0.0)
            radius_km = st.number_input("Radius (km)", value=20.0)
            
            if st.button("Filter by Radius (Lat/Lon)"):
                df_radius['distance_km'] = haversine_np(ref_lat, ref_lon, df_radius[lat_col].astype(float), df_radius[lon_col].astype(float))
                filtered_df = df_radius[df_radius['distance_km'] <= radius_km]
                st.write(f"**{len(filtered_df)} rows within {radius_km} km**")
                create_download_link(filtered_df.drop(columns=['distance_km']), f"radius_filtered_{radius_file.name}")
                if len(filtered_df) > 0:
                    st.dataframe(filtered_df.head(200))
        
        else:  # Address / Pincode
            addr_col = st.selectbox("Address column", df_radius.columns.tolist(), key="addr_col")
            pincode_col = st.selectbox("Pincode column (optional)", [None]+df_radius.columns.tolist(), key="pin_col")
            ref_lat = st.number_input("Reference point latitude", value=0.0, key="addr_ref_lat")
            ref_lon = st.number_input("Reference point longitude", value=0.0, key="addr_ref_lon")
            radius_km = st.number_input("Radius (km)", value=20.0, key="addr_radius_km")
            
            if st.button("Filter by Radius (Address)"):
                lat_list, lon_list = [], []
                for idx, row in df_radius.iterrows():
                    lat, lon = geocode_address(row[addr_col], row[pincode_col] if pincode_col else None)
                    lat_list.append(lat)
                    lon_list.append(lon)
                df_radius['lat'] = lat_list
                df_radius['lon'] = lon_list
                df_radius['distance_km'] = haversine_np(ref_lat, ref_lon, df_radius['lat'].astype(float), df_radius['lon'].astype(float))
                filtered_df = df_radius[df_radius['distance_km'] <= radius_km]
                st.write(f"**{len(filtered_df)} rows within {radius_km} km**")
                create_download_link(filtered_df.drop(columns=['lat','lon','distance_km']), f"radius_filtered_{radius_file.name}")
                if len(filtered_df) > 0:
                    st.dataframe(filtered_df.head(200))
# ------------------ End of App ------------------