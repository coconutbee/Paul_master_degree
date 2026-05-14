import argparse
import json
import os
import re

# ==========================================
# 1. 定義 Mapping Rules (Hard Coded)
# ==========================================

# --- Expression Mapping ---
EXPRESSION_MAPPING_RULES = {
    'happy': ['happy', 'grin', 'smile', 'smiles', 'grinning', 'smiling', 'laugh', 'giggles', 'joy', 'smirk', 'joyful', 'enjoy'],
    'surprise': ['surprised', 'surprise', 'amazed', 'astonished', 'skeptical', 'shock'],
    'confuse': ['confuse', 'frown', 'puzzled', 'questioning', 'thoughtful', 'confused'],
    'neutral': ['neutral', 'lips pursed', 'satisfaction', 'calmness', 'dreamy', 'serene', 'calm'],
    'sad': ['sad', 'sadness', 'crying', 'gloomy', 'depressed'], 
    'others': ['others']
}

# --- Gender Mapping (Regex) ---
MALE_KEYWORDS = [r'\bman\b', r'\bboy\b', r'\bmale\b', r'\bmen\b', r'\bguy\b']
FEMALE_KEYWORDS = [r'\bwoman\b', r'\bgirl\b', r'\bfemale\b', r'\blady\b', r'\bwomen\b']

# --- Pose Mapping ---
YAW_DESCS = {
    "forward": "facing forward",
    "left": "turned his/her head to his/her left",
    "right": "turned his/her head to his/her right",
    "left_over_shoulder": "turned his/her head to his/her left over the shoulder",
    "right_over_shoulder": "turned his/her head to his/her right over the shoulder",
}
PITCH_DESCS = {
    "straight": "looking straight",
    "up": "chin up",
    "down": "chin down",
}
# ==========================================
# 2. 定義標註邏輯函數
# ==========================================

def normalize_prompt_text(text):
    """
    Normalize prompt text only for matching rules.
    Some generated metadata stores spaces as underscores, e.g.
    A_white_young_boy_turned_his_head_to_his_left_and_chin_down
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()

def get_expression(text):
    text_lower = normalize_prompt_text(text).lower()
    for label, keywords in EXPRESSION_MAPPING_RULES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return label
    return "others"

def get_gender(text):
    text_lower = normalize_prompt_text(text).lower()
    is_male = False
    is_female = False
    
    for pattern in MALE_KEYWORDS:
        if re.search(pattern, text_lower):
            is_male = True
            break
            
    for pattern in FEMALE_KEYWORDS:
        if re.search(pattern, text_lower):
            is_female = True
            break
    
    if is_male and is_female: return "Both"
    elif is_male: return "Male"
    elif is_female: return "Female"
    else: return "Unknown"

def _canonical_pose_label(yaw, pitch):
    if yaw == "forward":
        if pitch == "up":
            return "head chin up"
        if pitch == "down":
            return "head chin down"
        return "head facing forward straight"

    if yaw in {"left", "right", "left_over_shoulder", "right_over_shoulder"}:
        direction = "left" if "left" in yaw else "right"
        label = f"head turned to his/her {direction}"
        if yaw.endswith("_over_shoulder"):
            label += " over the shoulder"
        if pitch == "up":
            label += " and chin up"
        elif pitch == "down":
            label += " and chin down"
        return label

    return "Unknown"

def get_yaw(text):
    text_lower = normalize_prompt_text(text).lower()
    pronoun = r"(?:his|her|their|his/her|the)"
    verb = r"(?:turned|turning|turns)"

    yaw_patterns = [
        ("left_over_shoulder", rf"(?:{verb} {pronoun} head to {pronoun} left|head {verb} to {pronoun} left) over the shoulders?"),
        ("right_over_shoulder", rf"(?:{verb} {pronoun} head to {pronoun} right|head {verb} to {pronoun} right) over the shoulders?"),
        ("left", rf"(?:{verb} {pronoun} head to {pronoun} left|head {verb} to {pronoun} left)"),
        ("right", rf"(?:{verb} {pronoun} head to {pronoun} right|head {verb} to {pronoun} right)"),
    ]
    for yaw, pattern in yaw_patterns:
        if re.search(pattern, text_lower):
            return YAW_DESCS[yaw]

    if re.search(r"\bfacing forward\b|\bfaces? forward\b|\bhead facing forward straight\b", text_lower):
        return YAW_DESCS["forward"]

    return "Unknown"

def get_pitch(text):
    text_lower = normalize_prompt_text(text).lower()
    if re.search(r"\bchin up\b|\bhead chin up\b", text_lower):
        return PITCH_DESCS["up"]
    if re.search(r"\bchin down\b|\bhead chin down\b", text_lower):
        return PITCH_DESCS["down"]
    if re.search(r"\blooking straight\b|\blooks straight\b|\bhead facing forward straight\b", text_lower):
        return PITCH_DESCS["straight"]
    return PITCH_DESCS["straight"]

def get_pose(text):
    yaw_desc = get_yaw(text)
    pitch_desc = get_pitch(text)
    yaw_key = next((key for key, desc in YAW_DESCS.items() if desc == yaw_desc), None)
    pitch_key = next((key for key, desc in PITCH_DESCS.items() if desc == pitch_desc), None)
    if yaw_key is None or pitch_key is None:
        return "Unknown"
    return _canonical_pose_label(yaw_key, pitch_key)

# ==========================================
# 3. 主程式
# ==========================================

def process_json_data(input_data):
    """
    輸入: List of Dict (JSON 格式)
    輸出: 標註後的 List of Dict
    """
    labeled_data = []
    
    print(f"Processing {len(input_data)} items...")
    
    for item in input_data:
        prompt = item.get("prompt", "")
        
        # 進行標註
        item["gt_expression"] = get_expression(prompt)
        item["gt_gender"] = get_gender(prompt)
        item["gt_yaw"] = get_yaw(prompt)
        item["gt_pitch"] = get_pitch(prompt)
        item["gt_pose"] = get_pose(prompt)
        
        labeled_data.append(item)
        
    return labeled_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files for labeling.")
    parser.add_argument("--input", type=str, default="metadata.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="metadata.json", help="Output JSON file")
    args = parser.parse_args()

    # 2. 讀取 JSON 檔案
    if os.path.exists(args.input):
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                # 這裡就是你要的：從檔案讀取並存入變數
                raw_json_input = json.load(f) 
                
            # 3. 執行處理
            result_data = process_json_data(raw_json_input)

            # 4. 儲存結果
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
                
            print(f"Success! Processed {len(result_data)} items.")
            print(f"Results saved to: {args.output}")
        except json.JSONDecodeError:
            print(f"Error: {args.input} 的格式錯誤 (不是合法的 JSON)。")
    else:
        print(f"Error: 找不到檔案 {args.input}，請確認檔案是否在同一個資料夾內。")
