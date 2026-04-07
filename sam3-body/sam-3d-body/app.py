import streamlit as st
import pandas as pd
from PIL import Image
import os

# 設定頁面寬度
st.set_page_config(layout="wide", page_title="SAM-3D Pose Visualizer")

st.title("📸 SAM-3D 姿勢預測品質檢查工具")

# 1. 讀取資料
csv_path = "deepfahsion.csv" # <--- 修改為你的 CSV 路徑
if not os.path.exists(csv_path):
    st.error(f"找不到檔案: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

# 2. 側邊欄篩選器
st.sidebar.header("篩選條件")
yaw_range = st.sidebar.slider(
    "Yaw 範圍", 
    float(df['sam3d_head_body_yaw'].min()), 
    float(df['sam3d_head_body_yaw'].max()), 
    (float(df['sam3d_head_body_yaw'].min()), float(df['sam3d_head_body_yaw'].max()))
)

pitch_range = st.sidebar.slider(
    "Pitch 範圍", 
    float(df['sam3d_head_pitch'].min()), 
    float(df['sam3d_head_pitch'].max()), 
    (float(df['sam3d_head_pitch'].min()), float(df['sam3d_head_pitch'].max()))
)

status_filter = st.sidebar.multiselect("狀態篩選", options=df['sam3d_status'].unique(), default=df['sam3d_status'].unique())

# 應用篩選
filtered_df = df[
    (df['sam3d_head_body_yaw'].between(yaw_range[0], yaw_range[1])) &
    (df['sam3d_head_pitch'].between(pitch_range[0], pitch_range[1])) &
    (df['sam3d_status'].isin(status_filter))
]

st.write(f"當前顯示筆數: {len(filtered_df)} / {len(df)}")

# 3. 網格顯示圖片
cols_per_row = 4
rows = (len(filtered_df) // cols_per_row) + 1

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i * cols_per_row + j
        if idx < len(filtered_df):
            row = filtered_df.iloc[idx]
            img_path = row['image_path']
            
            with cols[j]:
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                    st.caption(f"**Yaw:** {row['sam3d_head_body_yaw']:.2f}")
                    st.caption(f"**Pitch:** {row['sam3d_head_pitch']:.2f}")
                    st.caption(f"📂 {os.path.basename(img_path)}")
                else:
                    st.warning(f"找不到圖片: {os.path.basename(img_path)}")