import argparse
import json
import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==========================================
# 1. 規則邏輯 (保持不變)
# ==========================================
EXPRESSION_MAPPING_RULES = {
    'happy': ['happy', 'grin', 'smile', 'smiles', 'grinning', 'smiling', 'laugh', 'giggles', 'joy', 'smirk', 'joyful', 'enjoy'],
    'surprise': ['surprised', 'surprise', 'amazed', 'astonished', 'skeptical', 'shock'],
    'confuse': ['confuse', 'frown', 'puzzled', 'questioning', 'thoughtful', 'confused'],
    'neutral': ['neutral', 'lips pursed', 'satisfaction', 'calmness', 'dreamy', 'serene', 'calm'],
    'sad': ['sad', 'sadness', 'crying', 'gloomy', 'depressed'], 
    'others': ['others']
}

MALE_KEYWORDS = [r'\bman\b', r'\bboy\b', r'\bmale\b', r'\bmen\b', r'\bguy\b']
FEMALE_KEYWORDS = [r'\bwoman\b', r'\bgirl\b', r'\bfemale\b', r'\blady\b', r'\bwomen\b']

def normalize_prompt_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()

def get_expression(text):
    text_lower = normalize_prompt_text(text).lower()
    for label, keywords in EXPRESSION_MAPPING_RULES.items():
        for keyword in keywords:
            if keyword in text_lower: return label
    return "others"

def get_gender(text):
    text_lower = normalize_prompt_text(text).lower()
    is_male = any(re.search(p, text_lower) for p in MALE_KEYWORDS)
    is_female = any(re.search(p, text_lower) for p in FEMALE_KEYWORDS)
    if is_male and is_female: return "Both"
    return "Male" if is_male else "Female" if is_female else "Unknown"

# ==========================================
# 2. T5/UL2 Pose Classifier (批次優化版)
# ==========================================
T5_TO_GT_YAW = {
    "turning head to the left": "turned his/her head to his/her left",
    "turning head to the right": "turned his/her head to his/her right",
    "turning head to the left over the shoulder": "turned his/her head to his/her left over the shoulder",
    "turning head to the right over the shoulder": "turned his/her head to his/her right over the shoulder",
    "facing forward": "facing forward"
}

class T5EncoderBatchClassifier:
    def __init__(self, model_id, cache_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Loading Local T5/UL2 Encoder from {cache_dir}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        # 設定 padding token，確保批次推論不報錯
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        self.yaw_labels = list(T5_TO_GT_YAW.keys())
        
        print("⚡ Precomputing label embeddings...")
        self.yaw_embeddings = self._get_embeddings(self.yaw_labels) # Shape: (Num_Labels, Hidden_Dim)

    def _get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # 優化: 使用 inference_mode 取代 no_grad
        with torch.inference_mode():
            # 兼容模型結構 (有些 AutoModel 會有 encoder 屬性，如果沒指定 encoder 就取 output)
            outputs = self.model.encoder(**inputs) if hasattr(self.model, 'encoder') else self.model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            embeddings = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(embeddings, p=2, dim=1)

    def batch_classify(self, prompts):
        """一次處理整個 Batch 的文字"""
        prompts_lower = [p.lower() for p in prompts]
        
        # 1. 一次性獲取整個 Batch 的 Embeddings (Shape: Batch_Size, Hidden_Dim)
        batch_embs = self._get_embeddings(prompts_lower)
        
        # 2. 矩陣乘法計算相似度 (Batch_Size, Num_Labels)
        # 因為已經做過 L2 Normalize，內積 (Dot Product) 就等於 Cosine Similarity
        batch_sims = torch.matmul(batch_embs, self.yaw_embeddings.T)
        
        results = []
        # 3. 對 Batch 內的每一筆資料應用硬性規則 (CPU 處理字串很快)
        for i, prompt_lower in enumerate(prompts_lower):
            sims = batch_sims[i].clone()
            
            has_left = "left" in prompt_lower
            has_right = "right" in prompt_lower
            has_over_shoulder = any(k in prompt_lower for k in ["over the shoulder", "over shoulder"])
            has_forward = "forward" in prompt_lower # 移除 "straight"，因為 straight 是 pitch 屬性 (looking straight) 而非 yaw 屬性
            
            for j, label in enumerate(self.yaw_labels):
                if has_left and not has_right and "right" in label: sims[j] -= 2.0
                if has_right and not has_left and "left" in label: sims[j] -= 2.0
                if has_over_shoulder and "over the shoulder" not in label: sims[j] -= 2.0
                elif not has_over_shoulder and "over the shoulder" in label: sims[j] -= 2.0
                if has_forward and "forward" not in label: sims[j] -= 2.0

            best_yaw_idx = torch.argmax(sims).item()
            
            pitch_match = re.search(r'(chin up|chin down)', prompt_lower)
            best_pitch = pitch_match.group(0) if pitch_match else "looking straight"

            results.append({"yaw": self.yaw_labels[best_yaw_idx], "pitch": best_pitch})

        return results

# ==========================================
# 3. 執行處理 (Batch 流程)
# ==========================================
def process_metadata(input_p, output_p, model_id, cache_dir, batch_size=64):
    classifier = T5EncoderBatchClassifier(model_id, cache_dir)
    
    if not os.path.exists(input_p):
        print(f"❌ Error: Input file not found: {input_p}")
        return

    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📦 Processing {len(data)} items with Batch Size {batch_size}...")
    
    # 將資料切分成 Batches
    for i in tqdm(range(0, len(data), batch_size)):
        batch_items = data[i:i+batch_size]
        
        # 預先清理這個 Batch 的文字
        clean_texts = [normalize_prompt_text(item.get("prompt", "")) for item in batch_items]
        
        # 一次推論整個 Batch
        batch_results = classifier.batch_classify(clean_texts)
        
        # 將結果寫回對應的 item
        for item, clean_text, res in zip(batch_items, clean_texts, batch_results):
            item["gt_expression"] = get_expression(clean_text)
            item["gt_gender"] = get_gender(clean_text)
            
            item["gt_yaw"] = T5_TO_GT_YAW[res["yaw"]]
            item["gt_pitch"] = res["pitch"]
            
            if res["yaw"] == "facing forward":
                item["gt_pose"] = "head facing forward straight" if res["pitch"] == "looking straight" else f"head {res['pitch']}"
            else:
                dir_str = "left" if "left" in res["yaw"] else "right"
                pose_str = f"head turned to his/her {dir_str}"
                if "over the shoulder" in res["yaw"]: pose_str += " over the shoulder"
                if res["pitch"] != "looking straight": pose_str += f" and {res['pitch']}"
                item["gt_pose"] = pose_str
            
    with open(output_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✨ Success! Saved to {output_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="20PP_test.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="small_batch_PP_v2_t5gemma.json", help="Output JSON file")
    # 增加 batch_size 參數，方便你隨時調整
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()
    
    MODEL_ID = "google/t5gemma-2b-2b-ul2"
    CACHE_DIR = "/media/ee303/4TB/Gemma/checkpoints" # 可依需求修改
    
    process_metadata(args.input, args.output, MODEL_ID, CACHE_DIR, args.batch_size)