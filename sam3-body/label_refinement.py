import os
import pandas as pd
from tqdm import tqdm

def _map_gender_and_age_to_role(age, gender_label):
    if pd.isna(age) or pd.isna(gender_label):
        return "person", "their"

    # 判斷是否為成人 (閾值為 20 歲)
    is_adult = float(age) >= 20.0
    g = str(gender_label).lower()
    if g in {"male", "man", "boy", "m"}:
        return ("man", "his") if is_adult else ("boy", "his")
    if g in {"female", "woman", "girl", "w", "f"}:
        return ("woman", "her") if is_adult else ("girl", "her")
    return "person", "their"

def generate_caption(row):
    # 提取基本屬性，如果沒有則給預設值
    race = str(row.get('race', 'unknown')).lower()
    age = row.get('mivolo_age', None)
    gender = row.get('mivolo_gender', None)
    
    # 根據年齡與性別決定 subject 與代名詞
    subject, pronoun = _map_gender_and_age_to_role(age, gender)
    
    # 格式化年齡字串
    if pd.notna(age):
        age_str = str(int(age))
    else:
        age_str = "unknown"

    # 1. 解析 Yaw (左右偏轉)
    yaw = row.get('sam3d_head_body_yaw', 0)
    if yaw > 40:
        yaw_desc = f"turned {pronoun} head to {pronoun} left over the shoulder"
    elif yaw < -40:
        yaw_desc = f"turned {pronoun} head to {pronoun} right over the shoulder"
    elif 20 < yaw:
        yaw_desc = f"turned {pronoun} head to {pronoun} left"
    elif yaw < -20:
        yaw_desc = f"turned {pronoun} head to {pronoun} right"
    else:
        yaw_desc = "facing forward"
        
    # 2. 解析 Pitch (上下抬頭)
    pitch = row.get('sam3d_head_pitch', 0)
    if pitch > 25:
        pitch_desc = "chin up"
    elif pitch < -25:
        pitch_desc = "chin down"
    else:
        pitch_desc = "looking straight"
        
    # 組合最終 caption_rf
    # Format: A {race} {subject} in {age} years old, {yaw_desc} and {pitch_desc}
    caption_rf = f"A {race} {subject} in {age_str} years old, {yaw_desc} and {pitch_desc}"
    return caption_rf

# --- 核心修改部分 ---
if __name__ == "__main__":
    # 1. 讀取 CSV 檔案
    csv_input_path = '/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age_race.csv' 
    df = pd.read_csv(csv_input_path)

    # 2. 開啟 tqdm 的 pandas 擴充功能
    tqdm.pandas(desc="處理圖片中")

    # 3. 使用 progress_apply 產生 caption_rf
    df['caption_rf'] = df.progress_apply(generate_caption, axis=1)

    # 檢查結果 (印出前 5 筆)
    if 'image_path' in df.columns:
        print(df[['image_path', 'caption_rf']].head())
    else:
        print(df[['caption_rf']].head())

    # 存檔成新的 CSV
    output_path = '/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age_race_labeled.csv'
    df.to_csv(output_path, index=False)
    print(f"處理完成，結果已儲存至 {output_path}")
