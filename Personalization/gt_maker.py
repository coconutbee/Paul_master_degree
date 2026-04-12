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
    'confuse': ['confuse', 'forward', 'puzzled', 'questioning', 'thoughtful', 'confused'],
    'neutral': ['neutral', 'lips pursed', 'satisfaction', 'calmness', 'dreamy', 'serene', 'calm'],
    'sad': ['sad', 'sadness', 'crying', 'gloomy', 'depressed'], 
    'others': ['others']
}

# --- Gender Mapping (Regex) ---
MALE_KEYWORDS = [r'\bman\b', r'\bboy\b', r'\bmale\b', r'\bmen\b', r'\bguy\b']
FEMALE_KEYWORDS = [r'\bwoman\b', r'\bgirl\b', r'\bfemale\b', r'\blady\b', r'\bwomen\b']

# --- Pose Mapping ---
# 評分端 eval_pose_v2 的最終標籤空間目前固定為下列四類。
EVAL_POSE_LABELS = {
    "Frontal",
    "Head_Turn_Left",
    "Head_Turn_Right",
    "Back_Over_Shoulder",
}

# 先做精確 phrase 規則，再用 regex 做變形兜底。
POSE_MAPPING_RULES = {
    # --- Back View 類 ---
    "turns her head back over her shoulder": "Back_Over_Shoulder",
    "turns her head over her right shoulder": "Back_Over_Shoulder",
    "looking back over the left shoulder": "Back_Over_Shoulder",
    "looking back over the right shoulder": "Back_Over_Shoulder",
    "back view, looking to the left": "Back_Over_Shoulder",
    "back view, looking to the right": "Back_Over_Shoulder",

    # --- Head Turn 類 ---
    "turns her head left": "Head_Turn_Left",
    "looks sideways toward the left": "Head_Turn_Left",
    "frontal view, looking to the left": "Head_Turn_Left",
    "looks down to her left": "Head_Turn_Left",
    "looks up and to his left": "Head_Turn_Left",
    "turns his face upward to the left": "Head_Turn_Left",

    "turns his head right": "Head_Turn_Right",
    "looks to his right": "Head_Turn_Right",
    "turns his head slightly to the right": "Head_Turn_Right",
    "frontal view, looking to the right": "Head_Turn_Right",

    # --- Frontal 類 ---
    "looks straight": "Frontal",
    "tilts her head downward": "Frontal",
    "faces downward": "Frontal",
    "faces slightly downward": "Frontal",
    "looks upward, head tilted back": "Frontal",
    "looks upward": "Frontal",
    "tilts her head backward": "Frontal",
    "head facing forward straight": "Frontal",
    "head tilted down": "Frontal",
    "head tilted up": "Frontal",
}

POSE_REGEX_RULES = [
    (r"over the shoulder", "Back_Over_Shoulder"),
    (r"head turned to (?:his|her|his/her) left", "Head_Turn_Left"),
    (r"head turned to (?:his|her|his/her) right", "Head_Turn_Right"),
    (r"looking to (?:his|her|the) left", "Head_Turn_Left"),
    (r"looking to (?:his|her|the) right", "Head_Turn_Right"),
    (r"head facing forward straight", "Frontal"),
    (r"head tilted (?:up|down)", "Frontal"),
]

'''
"正面": ["Frontal", "Head_Slight_Right", "Head_Slight_Left"],

"背對": ["Back_View_Straight", "Back_Over_Shoulder", "Back_View_Side_Looking_Away"],

"側向": ["Head_Turn_Right", "Head_Turn_Left", "Body_Turn_Right_Face_Front", "Body_Turn_Left_Face_Front", "Side_View_Right", "Side_View_Left"],

"傾斜/歪頭": ["Body_Lean_Right", "Body_Lean_Left", "Head_Tilt_Right", "Head_Tilt_Left"]
'''
# ==========================================
# 2. 定義標註邏輯函數
# ==========================================

def get_expression(text):
    text_lower = text.lower()
    for label, keywords in EXPRESSION_MAPPING_RULES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return label
    return "others"

def get_gender(text):
    text_lower = text.lower()
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

def get_pose(text):
    text_lower = text.lower()
    
    # 關鍵步驟：依照字串長度排序 (由長到短)
    # 這樣可以避免 "looks up" 先匹配到 "looks up and to his left" 的情況
    sorted_keys = sorted(POSE_MAPPING_RULES.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        # 使用簡單的 substring check，只要 prompt 包含這個規則就當作匹配
        if key.lower() in text_lower:
            return POSE_MAPPING_RULES[key]

    for pattern, label in POSE_REGEX_RULES:
        if re.search(pattern, text_lower):
            return label
            
    return "Unknown"

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