import os
import sys

import cv2
import pandas as pd
import torch
from insightface.app import FaceAnalysis
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

MIVOLO_MODEL_ID = "iitolstykh/mivolo_v2"
AGE_ADULT_THRESHOLD = 20.0

_MIVOLO_READY = False
_MIVOLO_DISABLED = False
_MIVOLO_CFG = None
_MIVOLO_MODEL = None
_MIVOLO_PROCESSOR = None
_FACE_APP = None
_MIVOLO_CACHE = {}


def _init_mivolo_once():
    global _MIVOLO_READY, _MIVOLO_DISABLED
    global _MIVOLO_CFG, _MIVOLO_MODEL, _MIVOLO_PROCESSOR, _FACE_APP

    if _MIVOLO_READY or _MIVOLO_DISABLED:
        return

    try:
        _FACE_APP = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))

        _MIVOLO_CFG = AutoConfig.from_pretrained(MIVOLO_MODEL_ID, trust_remote_code=True)
        _MIVOLO_MODEL = AutoModelForImageClassification.from_pretrained(
            MIVOLO_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _MIVOLO_PROCESSOR = AutoImageProcessor.from_pretrained(MIVOLO_MODEL_ID, trust_remote_code=True)
        _MIVOLO_READY = True
        print("MiVOLO initialized for fallback gender/age labeling.")
    except Exception as e:
        _MIVOLO_DISABLED = True
        print(f"MiVOLO initialization failed, fallback disabled: {e}")


def _predict_gender_age_with_mivolo(image_path):
    if not image_path or not os.path.isfile(image_path):
        return None, None

    if image_path in _MIVOLO_CACHE:
        return _MIVOLO_CACHE[image_path]

    _init_mivolo_once()
    if not _MIVOLO_READY:
        _MIVOLO_CACHE[image_path] = (None, None)
        return None, None

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            _MIVOLO_CACHE[image_path] = (None, None)
            return None, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = _FACE_APP.get(img)

        crop = img
        if len(faces) > 0:
            face = faces[0]
            h, w = img.shape[:2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]

        inputs = _MIVOLO_PROCESSOR(images=[crop])["pixel_values"]
        inputs = inputs.to(dtype=_MIVOLO_MODEL.dtype, device=_MIVOLO_MODEL.device)
        dummy_body = torch.zeros_like(inputs)

        with torch.no_grad():
            output = _MIVOLO_MODEL(faces_input=inputs, body_input=dummy_body)

        age = float(output.age_output[0].item())
        gender_idx = int(output.gender_class_idx[0].item())
        gender_label = str(_MIVOLO_CFG.gender_id2label[gender_idx]).lower()
        _MIVOLO_CACHE[image_path] = (age, gender_label)
        return age, gender_label
    except Exception:
        _MIVOLO_CACHE[image_path] = (None, None)
        return None, None


def _map_gender_and_age_to_role(age, gender_label):
    if age is None or gender_label is None:
        return "unknown", "their"

    is_adult = age >= AGE_ADULT_THRESHOLD
    g = gender_label.lower()
    if g in {"male", "man", "boy"}:
        return ("man", "his") if is_adult else ("boy", "his")
    if g in {"female", "woman", "girl"}:
        return ("woman", "her") if is_adult else ("girl", "her")
    return "unknown", "their"

def generate_row_outputs(row):
    # 1. 解析性別 (根據路徑中是否有 'MEN' 判斷)
    image_path = row['image_path']
    pred_age = None
    pred_gender = None
    if 'WOMEN' in str(image_path).upper(): # 轉大寫比對較保險
        gender = "woman"
        pronoun = "her"
    elif 'MEN' in str(image_path).upper():
        gender = "man"
        pronoun = "his"
    else:
        pred_age, pred_gender = _predict_gender_age_with_mivolo(str(image_path))
        gender, pronoun = _map_gender_and_age_to_role(pred_age, pred_gender)

    # 2. 解析 Yaw (左右偏轉)
    yaw = row['sam3d_head_body_yaw']
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
        
    # 3. 解析 Pitch (上下抬頭)
    pitch = row['sam3d_head_pitch']
    if pitch > 25:
        pitch_desc = "chin up"
    elif pitch < -25:
        pitch_desc = "chin down"
    else:
        pitch_desc = "looking straight"

    return pd.Series({
        'caption': f"A {gender} {yaw_desc} and {pitch_desc}",
        'mivolo_age': pred_age,
        'mivolo_gender': pred_gender,
    })

# --- 核心修改部分 ---
from tqdm import tqdm # 1. 引入 tqdm

# 讀取 CSV 檔案
csv_input_path = '/media/ee303/4TB/sam3-body/sam-3d-body/pitch_corrected_with_person.csv' 
df = pd.read_csv(csv_input_path)

# 2. 開啟 tqdm 的 pandas 擴充功能
tqdm.pandas(desc="處理圖片中")

# 3. 產生 caption，並把 MiVOLO 推論出的 age / gender 一起存下來
df[['caption', 'mivolo_age', 'mivolo_gender']] = df.progress_apply(generate_row_outputs, axis=1)

# 檢查結果 (印出前 5 筆)
print(df[['image_path', 'caption', 'mivolo_age', 'mivolo_gender']].head())

# 存檔成新的 CSV
df.to_csv('/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age.csv', index=False)
print("處理完成，結果已儲存至 pitch_corrected_labeled.csv")
