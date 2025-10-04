import streamlit as st
import pandas as pd
import io
import re
import csv
from io import StringIO
from typing import List

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
Upload **New** CSV file(s) and **Old** CSV file(s). The app will:
- Compare phone numbers from New files against Old files and remove records from New that exist in Old (based on normalized phone number).  
- OR use the *Internal duplicates* tool to upload a single file and remove duplicate phone numbers inside that file.  
- Cleaned file(s) will be available for download with filenames starting `update_<originalname>.csv`.
""")

tabs = st.tabs(["Compare New vs Old (multiple files)", "Internal duplicates (single file)"])

# ---------------------------- Compare New vs Old ----------------------------
with tabs[0]:
    st.header("Compare New vs Old")
    st.info("Upload at least one Old file and at least one New file. Phone numbers are detected from a chosen column in each file; normalization removes non-digits and keeps last 10 digits if longer.")
    
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

    process_btn = st.button("Process Compare (New vs Old)")

    if process_btn:
        if not old_files or not new_files:
            st.error("Please upload at least one Old file and at least one New file.")
        else:
            # Combine all old phones
            old_phones_set = set()
            st.write("Reading Old files and collecting phone numbers...")
            for f in old_files:
                try:
                    df_old = read_csv_file(f)
                    col_choice = old_cols[f.name]
                    phones = build_phone_series(df_old, col_choice)
                    count_nonempty = phones.str.len().astype(int).gt(0).sum()
                    st.write(f"Collected {count_nonempty} phone entries from {f.name}.")
                    old_phones_set.update(phones[phones.str.len()>0].tolist())
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")

            st.success(f"Total unique normalized phone numbers in Old files: {len(old_phones_set)}")
            st.write("---")

            # Process each new file
            for f in new_files:
                try:
                    df_new = read_csv_file(f)
                    new_col_choice = new_cols[f.name]
                    new_phones = build_phone_series(df_new, new_col_choice)
                    df_new["_normalized_phone_for_check"] = new_phones
                    mask_in_old = df_new["_normalized_phone_for_check"].isin(old_phones_set) & (df_new["_normalized_phone_for_check"].str.len()>0)
                    found_count = mask_in_old.sum()
                    st.write(f"Found {found_count} rows in `{f.name}` that match phones from Old files.")

                    cleaned_df = df_new.loc[~mask_in_old].drop(columns=["_normalized_phone_for_check"])
                    cleaned_name = f"update_{f.name}"
                    if cleaned_df.empty:
                        st.warning(f"After removing duplicates, `{cleaned_name}` is empty.")
                    else:
                        create_download_link(cleaned_df, cleaned_name)

                    # Store removed rows in session state for checkbox preview
                    if found_count > 0:
                        key_removed = f"removed_rows_{f.name}"
                        st.session_state[key_removed] = df_new.loc[mask_in_old].drop(columns=["_normalized_phone_for_check"])
                        show_preview = st.checkbox(f"Show removed rows preview for `{f.name}`", key=f"show_removed_{f.name}")
                        if show_preview and key_removed in st.session_state:
                            st.dataframe(st.session_state[key_removed].head(200))

                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")

# ---------------------------- Internal duplicates ----------------------------
with tabs[1]:
    st.header("Internal Duplicate Remover (single file)")
    st.info("Upload one CSV file and the app will detect repeated phone numbers inside the file and produce a cleaned CSV named `update_<originalname>.csv`.")
    
    single_file = st.file_uploader("Upload a single CSV file", type=["csv"], accept_multiple_files=False, key="single_file")
    if single_file:
        try:
            df_single = read_csv_file(single_file)
            st.write(f"Columns: {', '.join(df_single.columns.tolist()[:20])}{'...' if len(df_single.columns)>20 else ''}")
            phone_col = st.selectbox("Select phone column to check for duplicates", options=df_single.columns.tolist(), key="single_phone_col")
            phones = build_phone_series(df_single, phone_col)
            df_single["_normalized_phone_for_check"] = phones
            duplicated_mask = df_single["_normalized_phone_for_check"].duplicated(keep="first") & (df_single["_normalized_phone_for_check"].str.len()>0)
            total_duplicates = duplicated_mask.sum()
            st.write(f"Found {total_duplicates} duplicate rows (same normalized phone) inside the file.")

            cleaned_df = df_single.loc[~duplicated_mask].drop(columns=["_normalized_phone_for_check"])
            cleaned_name = f"update_{single_file.name}"
            if cleaned_df.empty:
                st.warning("After removing duplicates, resulting file is empty.")
            else:
                create_download_link(cleaned_df, cleaned_name)

            # Store removed rows for preview
            if total_duplicates > 0:
                key_removed_single = f"removed_rows_single"
                st.session_state[key_removed_single] = df_single.loc[duplicated_mask].drop(columns=["_normalized_phone_for_check"])
                show_preview_single = st.checkbox("Show removed duplicate rows preview", key="preview_internal_removed")
                if show_preview_single and key_removed_single in st.session_state:
                    st.dataframe(st.session_state[key_removed_single].head(200))

        except Exception as e:
            st.error(f"Failed to read or process file: {e}")

# ---------------------------- Notes ----------------------------
st.markdown("---")
st.markdown("**Notes & heuristics**")
st.markdown("""
- Phone normalization removes non-digit characters. If a phone number has more than 10 digits, the app keeps the **last 10 digits**.  
- The app relies on selecting the correct phone column.  
- Cleaned CSV files are named `update_<original_filename>.csv`.
""")
