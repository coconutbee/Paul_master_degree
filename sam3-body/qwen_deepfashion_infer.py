import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline


SAM3_RESULTS_CSV = "/media/ee303/4TB/sam3-body/sam3_results.csv"
SAM3_LABELED_CSV = "/media/ee303/4TB/sam3-body/sam3_labeled.csv"
MODEL_NAME = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3.5-4B")
OUTPUT_COLUMN = "qwen_detailed_caption"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0"))


def _resolve_image_path(path_str):
    s = str(path_str).strip()
    
    p1 = Path(s)
    if p1.exists():
        return str(p1)
        
    s_norm = s.replace("/4TB/paul/", "/4TB/")
    p2 = Path(s_norm)
    if p2.exists():
        return str(p2)
        
    return s_norm


def _set_generation_defaults(pipe_obj, max_new_tokens=700):
    cfg_candidates = [
        getattr(pipe_obj.model, "generation_config", None),
        getattr(pipe_obj.model, "config", None),
        getattr(getattr(pipe_obj.model, "config", None), "text_config", None),
    ]
    for cfg in cfg_candidates:
        if cfg is None:
            continue
        if hasattr(cfg, "max_length"):
            cfg.max_length = None
        if hasattr(cfg, "max_new_tokens"):
            cfg.max_new_tokens = max_new_tokens

    eos_id = getattr(pipe_obj.tokenizer, "eos_token_id", None)
    if eos_id is not None:
        if hasattr(pipe_obj.model.generation_config, "pad_token_id"):
            pipe_obj.model.generation_config.pad_token_id = eos_id
        if hasattr(pipe_obj.model.config, "pad_token_id"):
            pipe_obj.model.config.pad_token_id = eos_id


def _extract_assistant_text(result):
    try:
        if not result or not isinstance(result, list):
            return str(result)
            
        generated = result[0].get("generated_text", result[0])
        
        if isinstance(generated, list) and len(generated) > 0:
            last_turn = generated[-1]
            if isinstance(last_turn, dict) and "content" in last_turn:
                return str(last_turn.get("content", "")).strip()
                
        if isinstance(generated, str):
            if "<|im_start|>assistant" in generated:
                return generated.split("<|im_start|>assistant")[-1].strip()
            return generated.strip()
            
    except Exception as e:
        print(f"Extraction parsing error: {e}")
    return ""


def _extract_final_prompt(generated_text: str) -> str:
    # 移除思考標籤
    text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.IGNORECASE | re.DOTALL)
    
    # 尋找 <final_prompt> 標籤，並只提取裡面的內容
    matches = re.findall(r"<final_prompt>(.*?)</final_prompt>", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        candidate = matches[-1].strip() # 取最後一個標籤內的文字
        if len(candidate) > 10:
            return candidate
            
    return text.strip()


def _clean_and_ensure_pose(text: str, pose_prefix: str) -> str:
    s = str(text)

    # 1. 清理常見的「非截斷式」廢話 (使用替換為空格，絕對不截斷！)
    bad_phrases = [
        r"(?i)The user wants a detailed image description based on the provided image and specific pose information\.",
        r"(?i)I need to follow a strict structure:",
        r"(?i)Constraint: Output ONLY the final paragraph\. No planning or filler\.",
        r"(?i)Here is a detailed description.*?:",
        r"(?i)Based on the provided image.*?:",
    ]
    for pattern in bad_phrases:
        s = re.sub(pattern, " ", s)

    # 2. 清除 Markdown 符號與殘留的 XML 標籤
    s = re.sub(r"</?[a-zA-Z0-9_]+>", "", s)
    s = re.sub(r"^\s*[*-]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*\*", "", s)
    
    # 3. 壓平空格
    s = " ".join(s.split()).strip()

    # 4. 句尾修復：如果模型真的沒寫完，直接補句號 (不刪除任何文字！)
    if s and not s.endswith(('.', '!', '?', '"', "'")):
        s += "."

    # 5. 確保 Pose Prefix 完美位於句首且不重複
    if s:
        # 安全地把句中可能重複的 pose_prefix 移除
        s = s.replace(pose_prefix, "").strip()
        # 移除開頭殘留的標點符號
        s = re.sub(r"^[.,;:\-\s]+", "", s)
        s = f"{pose_prefix} {s}"
    else:
        s = pose_prefix

    return s.strip()


def _load_and_merge_csvs():
    df_labeled = pd.read_csv(SAM3_LABELED_CSV)
    df_results = pd.read_csv(SAM3_RESULTS_CSV)

    df_labeled["_safe_path"] = df_labeled["image_path"].map(lambda x: str(x).split("/")[-1])
    df_results["_safe_path"] = df_results["image_path"].map(lambda x: str(x).split("/")[-1])

    merge_cols = ["_safe_path", "sam3d_head_body_yaw", "sam3d_head_pitch"]
    df_results_small = df_results[merge_cols].drop_duplicates(subset=["_safe_path"])

    merged = df_labeled.merge(
        df_results_small,
        how="left",
        on="_safe_path",
        suffixes=("", "_res"),
    )

    merged["sam3d_head_body_yaw"] = merged["sam3d_head_body_yaw"].fillna(merged["sam3d_head_body_yaw_res"])
    merged["sam3d_head_pitch"] = merged["sam3d_head_pitch"].fillna(merged["sam3d_head_pitch_res"])
    merged = merged.drop(columns=["_safe_path", "sam3d_head_body_yaw_res", "sam3d_head_pitch_res"])

    return merged


def _get_gender_info(caption_str):
    c = str(caption_str).lower()
    if "woman" in c or "female" in c or "girl" in c:
        return "woman", "Her"
    elif "man" in c or "male" in c or "boy" in c:
        return "man", "His"
    else:
        return "person", "Their"


def _build_prompts(yaw, pitch, original_caption):
    yaw_dir = "left" if yaw > 0 else "right"
    pitch_dir = "up" if pitch > 0 else "down"
    yaw_abs = abs(yaw)
    pitch_abs = abs(pitch)

    noun, possessive_pronoun = _get_gender_info(original_caption)
    
    clean_caption = str(original_caption).strip().rstrip(".")
    if clean_caption.lower() == "nan" or not clean_caption:
        pose_prefix = (
            f"A {noun} is facing the camera. {possessive_pronoun} head is turned {yaw_abs:.2f} degrees to "
            f"{possessive_pronoun.lower()} {yaw_dir}, and tilted {pitch_abs:.2f} degrees {pitch_dir}."
        )
    else:
        clean_caption = clean_caption[:1].upper() + clean_caption[1:]
        pose_prefix = (
            f"{clean_caption}. {possessive_pronoun} head is turned {yaw_abs:.2f} degrees to "
            f"{possessive_pronoun.lower()} {yaw_dir}, and tilted {pitch_abs:.2f} degrees {pitch_dir}."
        )

    # 重新設計的 Prompt：強制使用標籤，並明確禁止在標籤內寫計畫
    user_prompt = (
        f"You are an expert image annotator. Write a single, fluid, highly detailed paragraph (approx. 300 words) describing the image.\n\n"
        f"CRITICAL RULES:\n"
        f"1. You MUST enclose your ENTIRE description within <final_prompt> and </final_prompt> tags.\n"
        f"2. DO NOT include any internal thoughts, plans, or conversational text inside the tags. ONLY the raw description.\n"
        f"3. Inside the tags, the VERY FIRST SENTENCE must be exactly this:\n"
        f"\"{pose_prefix}\"\n"
        f"4. After the pose sentence, seamlessly describe the physical appearance, facial expression, skin tone, and hair.\n"
        f"5. Detail the clothing extensively (colors, fabric, folds, exact text/graphics).\n"
        f"6. Describe posture, limb positioning, lighting, and background."
    )

    return user_prompt, pose_prefix


def main():
    df = _load_and_merge_csvs()
    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS).copy()
        print(f"Debug mode enabled. Processing first {len(df)} rows.")

    print(f"Loading model: {MODEL_NAME}")
    pipe = pipeline(
        "image-text-to-text",
        model=MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
    )
    _set_generation_defaults(pipe, max_new_tokens=700)

    qwen_captions = []

    print("Starting batch inference...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = _resolve_image_path(row["image_path"])

        if not Path(image_path).exists():
            print(f"\n[{index}] Error: Image file does not exist at {image_path}")
            qwen_captions.append("")
            continue

        try:
            yaw = float(row["sam3d_head_body_yaw"])
            pitch = float(row["sam3d_head_pitch"])
            original_caption = str(row.get("caption", "")) 
        except Exception:
            print(f"\n[{index}] Invalid yaw/pitch/caption for image: {image_path}")
            qwen_captions.append("")
            continue

        user_prompt, pose_prefix = _build_prompts(yaw, pitch, original_caption)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}, 
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        try:
            # 直接傳遞 max_new_tokens 即可
            result = pipe(messages, max_new_tokens=700)
            
            raw_response = _extract_assistant_text(result)
            extracted_prompt = _extract_final_prompt(raw_response)
            assistant_response = _clean_and_ensure_pose(extracted_prompt, pose_prefix)
            
            if not assistant_response or assistant_response == pose_prefix:
                print(f"\n[{index}] WARNING: Model generated empty text or only the pose. Raw Output:\n{raw_response}")

            preview = assistant_response[:120].replace("\n", " ")
            print(f"[{index}] {preview}...")
            
        except Exception as e:
            print(f"\nError processing index {index} ({image_path}): {e}")
            assistant_response = ""

        qwen_captions.append(assistant_response)

    df[OUTPUT_COLUMN] = qwen_captions
    df.to_csv(SAM3_LABELED_CSV, index=False)
    print(f"\nDone. Captions appended to: {SAM3_LABELED_CSV}")


if __name__ == "__main__":
    main()