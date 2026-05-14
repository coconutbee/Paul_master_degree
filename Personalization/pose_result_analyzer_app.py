import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image


DEFAULT_JSON = "/media/ee303/4TB/Personalization/summary/Flux2_PP_noref_metadata.json"
DEFAULT_IMAGE_ROOT = "/media/ee303/4TB/flux2/prompt_test_512/ori"
DEFAULT_SUMMARY_DIR = "/media/ee303/4TB/Personalization/summary"
DEFAULT_METHOD_ROOTS = {
    "Flux2_PP": "/media/ee303/4TB/flux2/prompt_test_512/ori",
    "Sana1.5_PP": "/media/ee303/4TB/Sana/prompt_test_512/ori",
    "Infinity_PP": "/media/ee303/4TB/Infinity/generated/modify_short_prompt",
    "hart_PP": "/media/ee303/4TB/hart/generated",
    "Emu3.5_PP": "/media/ee303/4TB/Emu3.5/prompt_test_512/ori_4_8B",
    "Janus-Pro_7B_PP": "/media/ee303/disk1/Janus/prompt_test_512/janus_generated_samples",
    "Lumina_PP": "/media/ee303/4TB/SoftREPA/generated/lumina/PP",
    "Lumina_GP": "/media/ee303/4TB/SoftREPA/generated/lumina/GP",
    "SD3_PP": "/media/ee303/4TB/SoftREPA/generated/PP_vanilla",
    "SD3_GP": "/media/ee303/4TB/SoftREPA/generated/GP_vanilla",
    "SoftREPA_PP": "/media/ee303/4TB/SoftREPA/generated/PP_SoftREPA",
    "SoftREPA_GP": "/media/ee303/4TB/SoftREPA/generated/GP_SoftREPA",
    "SoftREPA_FT_PP": "/media/ee303/4TB/SoftREPA/generated/PP_FT_v2",
    "SoftREPA_FT_GP": "/media/ee303/4TB/SoftREPA/generated/GP_FT_v2",
    "flux2_GP_20": "/media/ee303/4TB/flux2/prompt_test_512/wo_id_20prompt",
    "infinity_GP_20": "/media/ee303/4TB/Infinity/generated/20prompt",
}


st.set_page_config(page_title="Pose Result Analyzer", layout="wide")


@st.cache_data(show_spinner=False)
def load_metadata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty:
        return df

    for col in [
        "t2i_head_body_yaw",
        "t2i_head_pitch",
        "t2i_pose_match",
        "pose_match",
        "t2i_yaw_match",
        "t2i_pitch_match",
        "t2i_scenario_score",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "id",
        "image",
        "prompt",
        "gt_yaw",
        "gt_pitch",
        "gt_pose",
        "t2i_yaw",
        "t2i_pitch",
        "t2i_pose",
        "t2i_pose_status",
        "t2i_scenario_reasoning",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    df["row_id"] = range(len(df))
    df["pose_score"] = df["t2i_pose_match"].fillna(df["pose_match"]).fillna(0.0)
    df["yaw_correct"] = df["t2i_yaw_match"].fillna(0).astype(int)
    df["pitch_correct"] = df["t2i_pitch_match"].fillna(0).astype(int)

    def error_type(row):
        if row["yaw_correct"] and row["pitch_correct"]:
            return "Both correct"
        if row["yaw_correct"]:
            return "Pitch wrong only"
        if row["pitch_correct"]:
            return "Yaw wrong only"
        return "Both wrong"

    df["error_type"] = df.apply(error_type, axis=1)
    df["yaw_transition"] = df["gt_yaw"] + " -> " + df["t2i_yaw"]
    df["pitch_transition"] = df["gt_pitch"] + " -> " + df["t2i_pitch"]
    df["prompt_text"] = df["prompt"].str.replace("_", " ", regex=False)
    return df



def method_name_from_path(json_path):
    name = Path(json_path).stem
    for suffix in ("_noref_metadata", "_metadata"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def list_summary_jsons(summary_dir):
    if not summary_dir:
        return []
    
    if os.path.isfile(summary_dir) and summary_dir.endswith(".jsonl"):
        paths = []
        try:
            with open(summary_dir, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "json" in data and os.path.exists(data["json"]):
                            paths.append(data["json"])
        except Exception:
            pass
        return paths

    if not os.path.isdir(summary_dir):
        return []
    return sorted(str(p) for p in Path(summary_dir).glob("*.json"))


def default_method_roots_text():
    return "\n".join(f"{method}={root}" for method, root in DEFAULT_METHOD_ROOTS.items())


def parse_method_roots(raw_text):
    roots = dict(DEFAULT_METHOD_ROOTS)
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        method, root = line.split("=", 1)
        method = method.strip()
        root = root.strip()
        if method:
            roots[method] = root
    return roots


def build_compare_frame(json_paths, key_col):
    frames = []
    import re
    # Function to unify prompt text so it aligns across all models
    def normalize_prompt(text):
        if not isinstance(text, str): return str(text)
        # Some models use 'A_asian_man', some use 'asian_man' in their prompt fields.
        # Remove common leading articles or underscores to make them identical
        text = text.replace(" ", "_")
        text = re.sub(r'^(A_|an_|a_)', '', text, flags=re.IGNORECASE)
        # Sometimes people capitalize the first letter differently
        return text.lower()

    for json_path in json_paths:
        df = load_metadata(json_path).copy()
        if df.empty or key_col not in df.columns:
            continue
        method = method_name_from_path(json_path)
        df["method"] = method
        df["json_path"] = json_path
        
        if key_col == "prompt":
            df["compare_key"] = df[key_col].apply(normalize_prompt)
        else:
            # If using 'image' as align key, try to strip numeric prefixes
            df["compare_key"] = df[key_col].apply(lambda x: re.sub(r'^([a-zA-Z0-9]+_)?(A_|an_)', r'\2', str(x)).rsplit('.', 1)[0].lower())
            
        keep_cols = [
            "method",
            "json_path",
            "compare_key",
            "row_id",
            "id",
            "image",
            "prompt",
            "prompt_text",
            "gt_yaw",
            "gt_pitch",
            "gt_pose",
            "t2i_yaw",
            "t2i_pitch",
            "t2i_pose",
            "t2i_head_body_yaw",
            "t2i_head_pitch",
            "t2i_yaw_match",
            "t2i_pitch_match",
            "t2i_pose_match",
            "pose_score",
            "error_type",
            "t2i_pose_status",
            "t2i_scenario_score",
            "t2i_scenario_reasoning",
        ]
        frames.append(df[[c for c in keep_cols if c in df.columns]])
    if not frames:
        return pd.DataFrame()
        
    df_concat = pd.concat(frames, ignore_index=True)
    # Remove duplicate prompts per method so a model only shows up once per test case
    df_concat = df_concat.drop_duplicates(subset=["method", "compare_key"], keep="first")
    return df_concat


def render_method_card(row, method_roots, image_indexes, index_images):
    method = row.get("method", "Unknown")
    root = method_roots.get(method, "")
    image_index = image_indexes.get(method, {}) if index_images else {}
    path = resolve_image_path(row, row.get("json_path", ""), root, image_index)

    st.markdown(f"**{method}**")
    if path:
        st.image(path, use_container_width=True)
    else:
        st.info("Image not found")

    score = row.get("pose_score", row.get("t2i_pose_match", 0.0))
    try:
        score_text = f"{float(score):.1f}"
    except (TypeError, ValueError):
        score_text = "N/A"
    yaw = row.get("t2i_head_body_yaw")
    pitch = row.get("t2i_head_pitch")
    yaw_text = f"{float(yaw):.2f}" if pd.notna(yaw) else "N/A"
    pitch_text = f"{float(pitch):.2f}" if pd.notna(pitch) else "N/A"

    st.caption(f"score {score_text} | yaw_angle {yaw_text} | pitch_angle {pitch_text}")
    
    comp_list = [
        {"Component": "Yaw", "GT": row.get("gt_yaw"), "T2I": row.get("t2i_yaw"), "Match": row.get("t2i_yaw_match")},
        {"Component": "Pitch", "GT": row.get("gt_pitch"), "T2I": row.get("t2i_pitch"), "Match": row.get("t2i_pitch_match")},
        {"Component": "Pose", "GT": row.get("gt_pose"), "T2I": row.get("t2i_pose"), "Match": row.get("t2i_pose_match")},
    ]
    if "t2i_scenario_score" in row and pd.notna(row["t2i_scenario_score"]):
        comp_list.append({"Component": "Scenario", "GT": "N/A", "T2I": "N/A", "Match": row.get("t2i_scenario_score")})
        
    compare = pd.DataFrame(comp_list)
    st.dataframe(compare, use_container_width=True, hide_index=True)
    if "t2i_scenario_reasoning" in row and pd.notna(row["t2i_scenario_reasoning"]):
        with st.expander("Scenario Reasoning"):
            st.write(row["t2i_scenario_reasoning"])


def render_method_comparison(summary_dir, method_roots, index_images):
    summary_files = list_summary_jsons(summary_dir)
    if not summary_files:
        st.warning(f"No JSON files found in {summary_dir}")
        return

    method_options = {method_name_from_path(path): path for path in summary_files}
    default_methods = [m for m in method_options if m in {"Flux2_PP", "Sana1.5_PP"}]
    if not default_methods:
        default_methods = list(method_options)[: min(4, len(method_options))]

    selected_methods = st.multiselect(
        "Methods",
        options=list(method_options),
        default=default_methods,
    )
    key_col = st.radio("Align by", ["image", "prompt"], horizontal=True)

    selected_paths = [method_options[m] for m in selected_methods]
    compare_df = build_compare_frame(selected_paths, key_col)
    if compare_df.empty:
        st.info("No comparable rows loaded.")
        return

    key_counts = compare_df.groupby("compare_key")["method"].nunique().sort_values(ascending=False)
    min_methods = st.slider("Minimum methods available for prompt", 1, max(1, len(selected_methods)), min(2, max(1, len(selected_methods))))
    valid_keys = key_counts[key_counts >= min_methods].index.tolist()
    
    if not compare_df.empty:
        method_counts = compare_df.groupby("method").size().sort_values()
        min_method_name = method_counts.index[0]
        min_method_count = method_counts.iloc[0]
        
        use_min_method_base = st.checkbox(
            f"Restrict to prompts available in the smallest dataset ({min_method_name}: {min_method_count} items)", 
            value=True
        )
        if use_min_method_base:
            min_method_keys = set(compare_df[compare_df["method"] == min_method_name]["compare_key"])
            valid_keys = [k for k in valid_keys if k in min_method_keys]

    if not valid_keys:
        st.warning("No prompt/image key is shared by the selected methods under the current threshold.")
        return

    query = st.text_input("Search prompt/image key", "")
    if query:
        q = query.lower()
        valid_keys = [k for k in valid_keys if q in k.lower()]
    if not valid_keys:
        st.warning("No key matches the search.")
        return

    selected_key = st.selectbox(
        "Prompt / image key",
        valid_keys,
        format_func=lambda k: f"{k} ({int(key_counts[k])} methods)",
    )
    rows = compare_df[compare_df["compare_key"] == selected_key].sort_values("method")

    prompt = rows["prompt_text"].iloc[0] if "prompt_text" in rows else selected_key
    st.subheader("Prompt")
    st.code(prompt, language="text")

    summary_cols = [
        "method",
        "gt_yaw",
        "gt_pitch",
        "t2i_yaw",
        "t2i_pitch",
        "t2i_head_body_yaw",
        "t2i_head_pitch",
        "t2i_yaw_match",
        "t2i_pitch_match",
        "t2i_pose_match",
        "t2i_pose_status",
        "t2i_scenario_score",
    ]
    st.dataframe(rows[[c for c in summary_cols if c in rows.columns]], use_container_width=True, hide_index=True)

    image_indexes = {}
    if index_images:
        with st.spinner("Indexing method image roots..."):
            image_indexes = {method: build_image_index(method_roots.get(method, "")) for method in selected_methods}

    cols_per_row = st.slider("Cards per row", 2, 5, 3)
    cols = st.columns(cols_per_row)
    for i, (_, row) in enumerate(rows.iterrows()):
        with cols[i % cols_per_row]:
            render_method_card(row.to_dict(), method_roots, image_indexes, index_images)

@st.cache_data(show_spinner=False)
def build_image_index(image_root):
    if not image_root or not os.path.isdir(image_root):
        return {}
    index = {}
    for root, _, files in os.walk(image_root):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif")):
                index.setdefault(filename, os.path.join(root, filename))
    return index


def resolve_image_path(item, json_path, image_root, image_index):
    image_path = item.get("image_path")
    if isinstance(image_path, str) and os.path.exists(image_path):
        return image_path

    filename = item.get("image")
    if not isinstance(filename, str) or not filename:
        return None

    candidates = [
        os.path.join(os.path.dirname(json_path), filename),
        os.path.join(image_root, filename) if image_root else "",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    exact_match = image_index.get(filename)
    if exact_match:
        return exact_match

    basename_no_ext = os.path.splitext(filename)[0]
    for k, v in image_index.items():
        if k.startswith(basename_no_ext + "_") or ("_" + basename_no_ext + "_" in k):
            return v

    return None


def metric_pct(value):
    return f"{value * 100:.1f}%"


def count_chart(df, col, title, height=220):
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{col}:N", sort="-x", title=None),
            tooltip=[col, "count"],
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def confusion_table(df, gt_col, pred_col):
    table = pd.crosstab(df[gt_col], df[pred_col], dropna=False)
    return table


def filter_dataframe(df):
    with st.sidebar:
        st.header("Filters")
        score_options = ["All"] + [str(x) for x in sorted(df["pose_score"].dropna().unique())]
        score_choice = st.selectbox("Pose score", score_options)

        error_types = ["All"] + sorted(df["error_type"].unique())
        error_choice = st.selectbox("Error type", error_types)

        gt_yaws = ["All"] + sorted(df["gt_yaw"].dropna().unique())
        gt_yaw = st.selectbox("GT yaw", gt_yaws)

        gt_pitches = ["All"] + sorted(df["gt_pitch"].dropna().unique())
        gt_pitch = st.selectbox("GT pitch", gt_pitches)

        pred_yaws = ["All"] + sorted(df["t2i_yaw"].dropna().unique())
        pred_yaw = st.selectbox("T2I yaw", pred_yaws)

        pred_pitches = ["All"] + sorted(df["t2i_pitch"].dropna().unique())
        pred_pitch = st.selectbox("T2I pitch", pred_pitches)

        if "t2i_scenario_score" in df.columns:
            scenario_options = ["All"] + [str(x) for x in sorted(df["t2i_scenario_score"].dropna().unique())]
            scenario_choice = st.selectbox("Scenario score", scenario_options)

        query = st.text_input("Prompt/image contains")

    out = df.copy()
    if score_choice != "All":
        out = out[out["pose_score"] == float(score_choice)]
    if "t2i_scenario_score" in df.columns and scenario_choice != "All":
        out = out[out["t2i_scenario_score"] == float(scenario_choice)]
    if error_choice != "All":
        out = out[out["error_type"] == error_choice]
    if gt_yaw != "All":
        out = out[out["gt_yaw"] == gt_yaw]
    if gt_pitch != "All":
        out = out[out["gt_pitch"] == gt_pitch]
    if pred_yaw != "All":
        out = out[out["t2i_yaw"] == pred_yaw]
    if pred_pitch != "All":
        out = out[out["t2i_pitch"] == pred_pitch]
    if query:
        q = query.lower()
        out = out[
            out["prompt"].str.lower().str.contains(q, na=False)
            | out["image"].str.lower().str.contains(q, na=False)
        ]
    return out


def render_item_detail(item, json_path, image_root, image_index):
    left, right = st.columns([1.05, 1.4])
    image_path = resolve_image_path(item, json_path, image_root, image_index)

    with left:
        if image_path:
            st.image(Image.open(image_path), use_container_width=True)
            st.caption(image_path)
        else:
            st.warning("Image not found. Set the image root in the sidebar.")

    with right:
        st.subheader("Prompt")
        st.code(item.get("prompt_text", item.get("prompt", "")), language="text")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pose score", f"{float(item.get('pose_score', 0.0)):.1f}")
        m2.metric("Yaw match", int(item.get("yaw_correct", 0)))
        m3.metric("Pitch match", int(item.get("pitch_correct", 0)))
        m4.metric("Status", item.get("t2i_pose_status", "Unknown"))

        a1, a2 = st.columns(2)
        a1.metric("head_body_yaw", f"{float(item['t2i_head_body_yaw']):.2f}" if pd.notna(item.get("t2i_head_body_yaw")) else "N/A")
        a2.metric("head_pitch", f"{float(item['t2i_head_pitch']):.2f}" if pd.notna(item.get("t2i_head_pitch")) else "N/A")

        comp_list = [
            {"Component": "Yaw", "GT": item.get("gt_yaw"), "T2I": item.get("t2i_yaw"), "Match": item.get("yaw_correct")},
            {"Component": "Pitch", "GT": item.get("gt_pitch"), "T2I": item.get("t2i_pitch"), "Match": item.get("pitch_correct")},
            {"Component": "Pose", "GT": item.get("gt_pose"), "T2I": item.get("t2i_pose"), "Match": item.get("pose_score")},
        ]
        if "t2i_scenario_score" in item and pd.notna(item["t2i_scenario_score"]):
            comp_list.append({"Component": "Scenario", "GT": "N/A", "T2I": "N/A", "Match": item.get("t2i_scenario_score")})
            
        compare = pd.DataFrame(comp_list)
        st.dataframe(compare, use_container_width=True, hide_index=True)
        
        if "t2i_scenario_reasoning" in item and pd.notna(item["t2i_scenario_reasoning"]):
            with st.expander("Scenario Reasoning"):
                st.write(item["t2i_scenario_reasoning"])


def main():
    st.title("Pose Result Analyzer")

    with st.sidebar:
        st.header("Data")
        json_path = st.text_input("Metadata JSON", DEFAULT_JSON)
        image_root = st.text_input("Image root", DEFAULT_IMAGE_ROOT)
        summary_dir = st.text_input("Summary dir for method compare", DEFAULT_SUMMARY_DIR)
        index_images = st.checkbox("Recursively index image roots", value=False)
        with st.expander("Method image roots"):
            method_roots_text = st.text_area(
                "One method=root per line",
                default_method_roots_text(),
                height=220,
            )
        method_roots = parse_method_roots(method_roots_text)
        st.caption("If images are not beside the JSON, set the folder containing generated images.")

    if not os.path.exists(json_path):
        st.error(f"JSON not found: {json_path}")
        return

    df = load_metadata(json_path)
    if df.empty:
        st.warning("No rows found in JSON.")
        return

    image_index = build_image_index(image_root) if index_images else {}
    filtered = filter_dataframe(df)

    st.caption(f"Loaded {len(df)} rows from `{json_path}`. Showing {len(filtered)} rows after filters.")

    cols = st.columns(6 if "t2i_scenario_score" in filtered.columns else 5)
    cols[0].metric("Rows", len(filtered))
    cols[1].metric("Avg pose score", metric_pct(filtered["pose_score"].mean() if len(filtered) else 0))
    cols[2].metric("Yaw accuracy", metric_pct(filtered["yaw_correct"].mean() if len(filtered) else 0))
    cols[3].metric("Pitch accuracy", metric_pct(filtered["pitch_correct"].mean() if len(filtered) else 0))
    cols[4].metric("Full match", metric_pct((filtered["pose_score"] == 1.0).mean() if len(filtered) else 0))
    if "t2i_scenario_score" in filtered.columns:
        cols[5].metric("Avg scenario", f"{filtered['t2i_scenario_score'].mean():.2f}" if len(filtered) else "0.00")

    tabs = st.tabs(["Overview", "Error Distribution", "Browse Images", "Compare Methods", "Data Table"])

    with tabs[0]:
        top_left, top_center, top_right = st.columns(3)
        with top_left:
            count_chart(filtered, "pose_score", "Pose score distribution")
        with top_center:
            if "t2i_scenario_score" in filtered.columns:
                count_chart(filtered, "t2i_scenario_score", "Scenario score distribution")
        with top_right:
            count_chart(filtered, "error_type", "Error type distribution")

        scatter_cols = [
            "image",
            "gt_yaw",
            "gt_pitch",
            "t2i_yaw",
            "t2i_pitch",
            "pose_score",
            "t2i_head_body_yaw",
            "t2i_head_pitch",
        ]
        scatter_df = filtered[scatter_cols].dropna(subset=["t2i_head_body_yaw", "t2i_head_pitch"]).copy()
        scatter_df["pose_score"] = scatter_df["pose_score"].astype(str)
        if not scatter_df.empty:
            chart = (
                alt.Chart(scatter_df)
                .mark_circle(size=60, opacity=0.65)
                .encode(
                    x=alt.X("t2i_head_body_yaw:Q", title="head_body_yaw"),
                    y=alt.Y("t2i_head_pitch:Q", title="head_pitch"),
                    color=alt.Color("pose_score:N", title="Pose score"),
                    tooltip=["image", "gt_yaw", "gt_pitch", "t2i_yaw", "t2i_pitch", "pose_score"],
                )
                .properties(height=420, title="Angle scatter by pose score")
            )
            st.altair_chart(chart, use_container_width=True)

    with tabs[1]:
        e1, e2 = st.columns(2)
        with e1:
            st.subheader("Yaw confusion")
            st.dataframe(confusion_table(filtered, "gt_yaw", "t2i_yaw"), use_container_width=True)
            yaw_errors = filtered[filtered["yaw_correct"] == 0]
            if not yaw_errors.empty:
                count_chart(yaw_errors, "yaw_transition", "Top yaw errors", height=360)
        with e2:
            st.subheader("Pitch confusion")
            st.dataframe(confusion_table(filtered, "gt_pitch", "t2i_pitch"), use_container_width=True)
            pitch_errors = filtered[filtered["pitch_correct"] == 0]
            if not pitch_errors.empty:
                count_chart(pitch_errors, "pitch_transition", "Top pitch errors", height=360)

    with tabs[2]:
        if filtered.empty:
            st.info("No rows match the current filters.")
        else:
            row_options = filtered["row_id"].tolist()
            selected_row = st.selectbox(
                "Select row",
                row_options,
                format_func=lambda rid: f"{rid} | {df.loc[df['row_id'] == rid, 'image'].iloc[0]}",
            )
            item = df[df["row_id"] == selected_row].iloc[0].to_dict()
            render_item_detail(item, json_path, image_root, image_index)

            st.divider()
            st.subheader("Gallery")
            gallery_count = st.slider("Number of filtered rows to show", 3, 60, 12, step=3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(filtered.head(gallery_count).iterrows()):
                with cols[i % 3]:
                    row_dict = row.to_dict()
                    path = resolve_image_path(row_dict, json_path, image_root, image_index)
                    if path:
                        st.image(path, use_container_width=True)
                    else:
                        st.info("Image not found")
                    st.caption(f"Score {row.pose_score:.1f} | yaw {row.t2i_yaw_match} | pitch {row.t2i_pitch_match}")
                    st.code(row.prompt_text, language="text")

    with tabs[3]:
        render_method_comparison(summary_dir, method_roots, index_images)

    with tabs[4]:
        cols = [
            "row_id",
            "id",
            "image",
            "prompt",
            "gt_yaw",
            "gt_pitch",
            "t2i_yaw",
            "t2i_pitch",
            "t2i_head_body_yaw",
            "t2i_head_pitch",
            "t2i_yaw_match",
            "t2i_pitch_match",
            "t2i_pose_match",
            "error_type",
        ]
        st.dataframe(filtered[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered CSV",
            filtered[cols].to_csv(index=False).encode("utf-8"),
            file_name="pose_result_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
