from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_CSV = Path("/media/ee303/4TB/Gemma/laion_gender_age_race_long_captioned_prompt2.csv")
REQUIRED_COLUMNS = ("image_path", "caption_rf", "long_caption")


st.set_page_config(page_title="CSV Caption Viewer", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def resolve_image_path(image_path: str, csv_path: Path) -> Path:
    path = Path(str(image_path)).expanduser()
    if path.is_absolute():
        return path
    return (csv_path.parent / path).resolve()


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def filter_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df

    query = query.strip().lower()
    if not query:
        return df

    searchable = (
        df["image_path"].fillna("").astype(str)
        + " "
        + df["caption_rf"].fillna("").astype(str)
        + " "
        + df["long_caption"].fillna("").astype(str)
    ).str.lower()
    return df[searchable.str.contains(query, regex=False, na=False)]


st.title("CSV Caption Viewer")

with st.sidebar:
    st.header("Data")
    csv_input = st.text_input("CSV path", value=str(DEFAULT_CSV))
    query = st.text_input("Search")
    rows_per_page = st.number_input("Rows per page", min_value=1, max_value=100, value=12)
    show_missing_images = st.checkbox("Show missing images", value=True)

csv_path = Path(csv_input).expanduser()
if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}")
    st.stop()

try:
    df = load_csv(str(csv_path))
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

missing_columns = validate_columns(df)
if missing_columns:
    st.error(
        "CSV is missing required columns: "
        + ", ".join(missing_columns)
        + f". Available columns: {', '.join(df.columns)}"
    )
    st.stop()

view_df = df.loc[:, REQUIRED_COLUMNS].copy()
view_df = filter_rows(view_df, query)

if not show_missing_images:
    image_exists = view_df["image_path"].map(lambda value: resolve_image_path(value, csv_path).exists())
    view_df = view_df[image_exists]

total_rows = len(view_df)
total_pages = max((total_rows + rows_per_page - 1) // rows_per_page, 1)

with st.sidebar:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    st.caption(f"{total_rows:,} matched rows / {len(df):,} total rows")

start = (page - 1) * rows_per_page
end = start + rows_per_page
page_df = view_df.iloc[start:end]

st.caption(f"Showing rows {start + 1:,}-{min(end, total_rows):,} of {total_rows:,}")

if page_df.empty:
    st.info("No rows matched.")
    st.stop()

for row_index, row in page_df.iterrows():
    image_path = resolve_image_path(row["image_path"], csv_path)

    with st.container(border=True):
        image_col, text_col = st.columns([1, 2], vertical_alignment="top")

        with image_col:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Image not found")
            st.code(str(row["image_path"]), language=None)

        with text_col:
            st.subheader(f"Row {row_index}")
            st.markdown("**caption_rf**")
            st.write(row["caption_rf"] if pd.notna(row["caption_rf"]) else "")
            st.markdown("**long_caption**")
            st.write(row["long_caption"] if pd.notna(row["long_caption"]) else "")
