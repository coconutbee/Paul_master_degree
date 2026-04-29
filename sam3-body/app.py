import streamlit as st
import pandas as pd
from PIL import Image
import os

st.set_page_config(layout="wide")
st.title("Image Caption & Refined Caption Viewer")

# 允許使用者在 UI 上修改 CSV 路徑，預設為剛剛輸出的 labeled csv
default_csv = "/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age_race_labeled.csv"
csv_path = st.text_input("CSV Path", default_csv)

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

if csv_path:
    df = load_data(csv_path)
    if df is not None:
        st.write(f"**Total Records:** {len(df)}")
        
        if 'index' not in st.session_state:
            st.session_state.index = 0
            
        # 控制區 (上一張、滑桿、下一張)
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.index = max(0, st.session_state.index - 1)
        with col2:
            st.session_state.index = st.slider("Select Image Index", 0, len(df)-1, st.session_state.index)
        with col3:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.index = min(len(df)-1, st.session_state.index + 1)
                
        # 取得當前圖片資訊
        row = df.iloc[st.session_state.index]
        img_path = row.get("image_path", "")
        caption = row.get("caption", "No original caption available")
        caption_rf = row.get("caption_rf", "No refined caption available")
        
        # 顯示區
        col_img, col_text = st.columns([1, 1])
        
        with col_img:
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path)
                    st.image(image, caption="Current Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Cannot load image: {e}")
            else:
                st.error(f"Image not found at path:\n{img_path}")
                
        with col_text:
            st.subheader("Original Caption")
            st.info(caption)
            
            st.subheader("Refined Caption (caption_rf)")
            st.success(caption_rf)
            
            st.markdown("---")
            st.subheader("Attributes")
            # 把顯示的資訊整理一下
            show_keys = ['race', 'age', 'mivolo_age', 'gender', 'mivolo_gender', 'sam3d_head_body_yaw', 'sam3d_head_pitch']
            attributes = {k: row[k] for k in show_keys if k in row}
            st.json(attributes)
            
    else:
        st.error("CSV file not found. Please check the path.")
