import streamlit as st
import json
import pandas as pd
import os

st.set_page_config(page_title="GT & T2I Debugger", layout="wide")

st.title("GT & T2I Prediction Debugger 🧪")
st.markdown("用來視覺化與除錯 JSON 檔案中的 Ground Truth (GT) 與 Text-to-Image (T2I) 預測結果。")

# Sidebar
st.sidebar.header("設定 (Settings)")
json_path = st.sidebar.text_input("輸入 JSON 檔案路徑", value="20prompt_TEstt_TE.json")
num_samples = st.sidebar.number_input("顯示數量 (0 表示全部)", min_value=0, max_value=10000, value=50)

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

if st.sidebar.button("載入資料 (Load Data)"):
    data = load_json(json_path)
    if data is not None:
        st.session_state['json_data'] = data
        st.session_state['json_path'] = json_path
        st.sidebar.success(f"成功載入: {json_path}")
    else:
        st.sidebar.error(f"找不到檔案: {json_path}")

# Display Data
if 'json_data' in st.session_state:
    data = st.session_state['json_data']
    st.write(f"**目前載入檔案:** `{st.session_state['json_path']}` | **總筆數:** `{len(data)}`")
    
    limit = num_samples if num_samples > 0 else len(data)
    display_data = data[:limit]
    
    # DataFrame View
    st.subheader("📊 表格總覽 (Data Table)")
    df = pd.DataFrame(display_data)
    # 將 prompt 移到前面方便看
    cols = df.columns.tolist()
    if 'prompt' in cols:
        cols.insert(1, cols.pop(cols.index('prompt')))
        df = df[cols]
    st.dataframe(df, use_container_width=True)
    
    # Detail View
    st.subheader("🔍 詳細檢視 (Detailed View)")
    for i, item in enumerate(display_data):
        prompt_text = item.get('prompt', '')
        short_prompt = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
        
        with st.expander(f"[{i+1}] ID: {item.get('id', 'N/A')} | {short_prompt}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📝 Prompt:**")
                st.info(prompt_text)
                
                if 'image' in item:
                    st.markdown("**🖼️ Image Filename:**")
                    st.code(item.get('image', ''))
                    
            with col2:
                st.markdown("**🎯 Ground Truth / Predictions:**")
                st.write(f"- **Gender:** `{item.get('gt_gender', '')}`")
                st.write(f"- **Expression:** `{item.get('gt_expression', '')}`")
                st.write(f"- **Yaw:** `{item.get('gt_yaw', '')}`")
                st.write(f"- **Pitch:** `{item.get('gt_pitch', '')}`")
                st.write(f"- **Pose:** `{item.get('gt_pose', '')}`")
                
                # 如果有其他未列出的欄位，也可以印出來
                other_keys = [k for k in item.keys() if k not in ['id', 'image', 'prompt', 'gt_gender', 'gt_expression', 'gt_yaw', 'gt_pitch', 'gt_pose']]
                if other_keys:
                    st.markdown("**⚙️ 其他屬性 (Other Attributes):**")
                    for k in other_keys:
                        st.write(f"- **{k}:** `{item[k]}`")

