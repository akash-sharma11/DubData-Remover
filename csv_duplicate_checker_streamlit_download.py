import streamlit as st
import pandas as pd
import io
import re
from io import StringIO
import csv

st.set_page_config(page_title="CSV Duplicate Checker by Phone", layout="wide")

# ---------------------------- Helpers ----------------------------
def normalize_phone(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    digits = re.sub(r"\D", "", s)
    if len(digits) > 10:
        digits = digits[-10:]
    return digits

def read_csv_file(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    try:
        sample = raw.decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample[:1024], delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ','

    try:
        df = pd.read_csv(StringIO(sample), sep=delimiter, dtype=str, keep_default_na=False, engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        raise Exception(f"Failed to read CSV with delimiter `{delimiter}`: {e}")

def build_phone_series(df: pd.DataFrame, col: str) -> pd.Series:
    series = df[col].astype(str).fillna("")
    return series.apply(normalize_phone)

def create_download_link(df: pd.DataFrame, filename: str):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return st.download_button(label=f"Download cleaned: {filename}", data=towrite, file_name=filename, mime="text/csv")

# ---------------------------- UI ----------------------------
st.title("CSV Duplicate Checker — by Phone Number")
st.markdown("""
Upload **New CSV file(s)** and **Old CSV file(s)**. The app will:
- Remove from each New file any rows whose phone numbers appear in the uploaded Old files.  
- You can preview the removed rows for each New file.  
- Cleaned New files can be downloaded directly.
""")

tabs = st.tabs(["Compare New vs Old (multiple files)", "Internal duplicates (single file)"])

# ---------------------------- Compare New vs Old ----------------------------
with tabs[0]:
    st.header("Compare New vs Old")
    st.info("Upload at least one Old file and at least one New file. Select the phone column for each before processing.")

    col1, col2 = st.columns([1,1])
    
    with col1:
        old_files = st.file_uploader("Upload Old CSV file(s) (min 1)", type=["csv"], accept_multiple_files=True, key="old_files")
        old_cols = {}
        if old_files:
            for f in old_files:
                try:
                    df_old = read_csv_file(f)
                    old_cols[f.name] = st.selectbox(f"Select phone column for old file `{f.name}`", options=df_old.columns.tolist(), key=f"oldcol_{f.name}")
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

    with col2:
        new_files = st.file_uploader("Upload New CSV file(s) (min 1)", type=["csv"], accept_multiple_files=True, key="new_files")
        new_cols = {}
        if new_files:
            for f in new_files:
                try:
                    df_new = read_csv_file(f)
                    new_cols[f.name] = st.selectbox(f"Select phone column for new file `{f.name}`", options=df_new.columns.tolist(), key=f"newcol_{f.name}")
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

    # initialize session_state dict to store removed rows
    if "removed_rows" not in st.session_state:
        st.session_state["removed_rows"] = {}

    process_btn = st.button("Process Compare (New vs Old)")

    if process_btn:
        if not old_files or not new_files:
            st.error("Please upload at least one Old file and at least one New file.")
        else:
            # combine old phones
            old_phones_set = set()
            for f in old_files:
                try:
                    df_old = read_csv_file(f)
                    col_choice = old_cols[f.name]
                    phones = build_phone_series(df_old, col_choice)
                    old_phones_set.update(phones[phones.str.len()>0].tolist())
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
            st.success(f"Total unique normalized phone numbers in Old files: {len(old_phones_set)}")
            st.write("---")

            # process each new file
            for f in new_files:
                try:
                    df_new = read_csv_file(f)
                    col_choice = new_cols[f.name]
                    phones = build_phone_series(df_new, col_choice)
                    df_new["_normalized_phone_for_check"] = phones
                    mask_in_old = df_new["_normalized_phone_for_check"].isin(old_phones_set) & (df_new["_normalized_phone_for_check"].str.len()>0)
                    
                    removed_rows = df_new.loc[mask_in_old].drop(columns=["_normalized_phone_for_check"])
                    cleaned_df = df_new.loc[~mask_in_old].drop(columns=["_normalized_phone_for_check"])
                    cleaned_name = f"update_{f.name}"

                    if cleaned_df.empty:
                        st.warning(f"After removing duplicates, `{cleaned_name}` is empty.")
                    else:
                        create_download_link(cleaned_df, cleaned_name)

                    # STORE REMOVED ROWS IN SESSION_STATE
                    st.session_state["removed_rows"][f.name] = removed_rows

                except Exception as e:
                    st.error(f"Failed to process `{f.name}`: {e}")

    # ------------------------- SHOW PREVIEWS -------------------------
    # Display preview checkboxes and render data from session_state
    if "removed_rows" in st.session_state:
        for fname, removed_df in st.session_state["removed_rows"].items():
            if st.checkbox(f"Show removed rows preview for `{fname}`", key=f"preview_{fname}"):
                if not removed_df.empty:
                    st.dataframe(removed_df.head(200))
                else:
                    st.info("No rows removed for this file.")

# ---------------------------- Internal duplicates ----------------------------
with tabs[1]:
    st.header("Internal Duplicate Remover (single file)")
    st.info("Upload one CSV file and remove duplicate phone numbers inside the file.")

    single_file = st.file_uploader("Upload a single CSV file", type=["csv"], accept_multiple_files=False, key="single_file")
    if single_file:
        try:
            df_single = read_csv_file(single_file)
            st.write(f"Columns: {', '.join(df_single.columns.tolist()[:20])}{'...' if len(df_single.columns)>20 else ''}")
            phone_col = st.selectbox("Select phone column", options=df_single.columns.tolist(), key="single_phone_col")
            phones = build_phone_series(df_single, phone_col)
            df_single["_normalized_phone_for_check"] = phones
            duplicated_mask = df_single["_normalized_phone_for_check"].duplicated(keep="first") & (df_single["_normalized_phone_for_check"].str.len()>0)
            removed_rows = df_single.loc[duplicated_mask].drop(columns=["_normalized_phone_for_check"])
            cleaned_df = df_single.loc[~duplicated_mask].drop(columns=["_normalized_phone_for_check"])
            cleaned_name = f"update_{single_file.name}"

            if cleaned_df.empty:
                st.warning("After removing duplicates, resulting file is empty.")
            else:
                create_download_link(cleaned_df, cleaned_name)

            # store for preview
            st.session_state["removed_single"] = removed_rows
            if st.checkbox("Show removed duplicate rows preview", key="preview_internal_removed"):
                if "removed_single" in st.session_state and not st.session_state["removed_single"].empty:
                    st.dataframe(st.session_state["removed_single"].head(200))
                else:
                    st.info("No duplicate rows removed.")

        except Exception as e:
            st.error(f"Failed to read/process file: {e}")

# ---------------------------- Notes ----------------------------
st.markdown("---")
st.markdown("**Notes & heuristics**")
st.markdown("""
- Phone normalization removes non-digit characters and keeps the last 10 digits.  
- For Compare New vs Old, only the **currently uploaded Old files** are used to remove matches from New files.  
- Cleaned CSV files are named `update_<original_filename>.csv`.
""")
