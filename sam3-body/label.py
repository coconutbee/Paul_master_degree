import pandas as pd

def generate_caption(row):
    # 1. 解析性別 (根據路徑中是否有 'MEN' 判斷)
    image_path = row['image_path']
    if 'MEN' in str(image_path).upper(): # 轉大寫比對較保險
        gender = "man"
        pronoun = "his"
    else:
        gender = "woman"
        pronoun = "her"
    
    # 2. 解析 Yaw (左右偏轉)
    yaw = row['sam3d_head_body_yaw']
    if yaw > 40:
        yaw_desc = f"turning {pronoun} head to {pronoun} left over the shoulder"
    elif yaw < -40:
        yaw_desc = f"turning {pronoun} head to {pronoun} right over the shoulder"
    elif 20 < yaw:
        yaw_desc = f"turning {pronoun} head to {pronoun} left"
    elif yaw < -20:
        yaw_desc = f"turning {pronoun} head to {pronoun} right"
    else:
        yaw_desc = "facing forward"
        
    # 3. 解析 Pitch (上下抬頭)
    pitch = row['sam3d_head_pitch']
    if pitch > 25:
        pitch_desc = "tilted up"
    elif pitch < -25:
        pitch_desc = "tilted down"
    else:
        pitch_desc = "looking straight"
        
    return f"A {gender} {yaw_desc} and {pitch_desc}"

# --- 核心修改部分 ---

# 1. 讀取 CSV 檔案
# 請確保 input.csv 的第一行（Header）有名稱為 'image_path', 'sam3d_head_body_yaw', 'sam3d_head_pitch' 的欄位
csv_input_path = 'sam3_results.csv' 
df = pd.read_csv(csv_input_path)

# 2. 產生 Caption
df['caption'] = df.apply(generate_caption, axis=1)

# 3. 檢查結果 (印出前 5 筆)
print(df[['image_path', 'caption']].head())

# 4. 存檔成新的 CSV
df.to_csv('sam3_labeled.csv', index=False)
print("處理完成，結果已儲存至 sam3_labeled.csv")