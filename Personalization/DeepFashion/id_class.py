import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增：用於特徵向量的 L2 正規化
import cv2
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import yaml
from math import cos, sin, atan2, asin
import math
import shutil
from collections import defaultdict

from ms1m_ir50.model_irse import IR_50
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX

# ----------------- 3DDFA-V2 初始化 -----------------
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)

# 定義 ArcFace 的標準影像預處理流程 (轉 Tensor + 正規化到 [-1, 1])
arcface_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def P2sRt(P):
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def matrix2angle(R):
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    return x, y, z

def calculate_rotation(image):
    # 注意：FaceBoxes 通常直接吃 OpenCV 讀進來的 BGR 圖片即可
    boxes = face_boxes(image)
    if len(boxes) == 0:
        return None
    param_lst, roi_box_lst = tddfa(image, boxes)
    param = param_lst[0]
    P1 = param[:12].reshape(3, -1).copy()
    s, R1, t3d = P2sRt(P1)
    angle = matrix2angle(R1)
    yaw, pitch, roll = angle
    return yaw * (180 / math.pi)

# ----------------- ArcFace 模型初始化 -----------------
def load_arcface_models():
    arcface_models = {
        'FF': './POE/FF/Backbone_IR_50_Epoch_80.pth',
        'FS': './POE/FS/Backbone_IR_50_Epoch_80.pth',
        'FP': './POE/FP/Backbone_IR_50_Epoch_120.pth',
        'SS': './POE/SS/Backbone_IR_50_Epoch_100.pth',
        'SP': './POE/SP/Backbone_IR_50_Epoch_150.pth',
        'PP': './POE/PP/Backbone_IR_50_Epoch_100.pth'
    }
    arcface_instances = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for name, path in arcface_models.items():
        arcface_model = IR_50([112, 112])
        if os.path.isfile(path):
            arcface_model.load_state_dict(torch.load(path))
            print(f"Loading Arcface Model Checkpoint '{path}' for {name}")
        arcface_instances[name] = arcface_model.to(device)
        arcface_instances[name].eval()
    return arcface_instances

def select_model_by_angle(arcface_models, real_angle, source_angle):
    if abs(real_angle) <= 30 and abs(source_angle) <= 30:
        return arcface_models['FF']
    elif abs(real_angle) >= 60 and abs(source_angle) >= 60:
        return arcface_models['PP']
    elif (abs(real_angle) >= 30 and abs(real_angle) <= 60) and (abs(source_angle) >= 30 and abs(source_angle) <= 60):
        return arcface_models['SS']
    elif (abs(real_angle) <= 30 and abs(source_angle) >= 60) or (abs(source_angle) <= 30 and abs(real_angle) >= 60):
        return arcface_models['FP']
    elif (abs(real_angle) <= 30 and 30 <= abs(source_angle) <= 60) or (abs(source_angle) <= 30 and 30 <= abs(real_angle) <= 60):
        return arcface_models['FS']
    else:
        return arcface_models['SP']

# ----------------- 核心比對與合併邏輯 -----------------
def merge_duplicate_identities(source_folder, output_base_folder, threshold=0.55, log_file_path='merge_log.txt'):
    arcface_models = load_arcface_models()
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_base_folder, exist_ok=True)
    
    print("\n--- Phase 1: 尋找所有資料夾的代表圖 (Anchors) 並萃取正確特徵 ---")
    folder_anchors = {}
    subfolders = sorted([f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f)) and f != 'output' and not f.startswith('.')])

    for subfolder in tqdm(subfolders, desc="Extracting Anchors"):
        subfolder_path = os.path.join(source_folder, subfolder)
        files = sorted(os.listdir(subfolder_path))
        
        for file in files:
            img_path = os.path.join(subfolder_path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            angle = calculate_rotation(img)
            if angle is not None:
                # 【修正1】將 BGR 轉為 RGB，因為 PyTorch 訓練的模型通常預期輸入為 RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 【修正2】使用 arcface_transform 進行正規化 (範圍轉為 [-1, 1])
                img_resized = cv2.resize(img_rgb, (112, 112))
                img_tensor = arcface_transform(img_resized).to(device).unsqueeze(0)
                
                folder_anchors[subfolder] = {
                    'tensor': img_tensor,
                    'angle': angle
                }
                break # 找到第一張有效人臉就當作代表，換下一個資料夾

    print(f"\n成功提取 {len(folder_anchors)} 個資料夾的代表特徵。")
    print("\n--- Phase 2: 跨資料夾 N x N 兩兩比對 (尋找重複模特兒) ---")
    
    folders = list(folder_anchors.keys())
    parent = {f: f for f in folders}

    # Union-Find 演算法
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    with open(log_file_path, 'w') as log_file:
        for i in tqdm(range(len(folders)), desc="Cross Comparing"):
            folder_i = folders[i]
            anchor_i = folder_anchors[folder_i]
            
            for j in range(i + 1, len(folders)):
                folder_j = folders[j]
                anchor_j = folder_anchors[folder_j]

                selected_model = select_model_by_angle(arcface_models, anchor_i['angle'], anchor_j['angle'])
                
                with torch.no_grad():
                    emb_i = selected_model(anchor_i['tensor'])
                    emb_j = selected_model(anchor_j['tensor'])
                    
                    # 【修正3】計算 Cosine Similarity 之前，必須將特徵向量進行 L2 正規化
                    emb_i = F.normalize(emb_i, p=2, dim=1)
                    emb_j = F.normalize(emb_j, p=2, dim=1)
                    
                csim = cos_sim(emb_i, emb_j).item()
                
                if csim >= threshold:
                    log_file.write(f"MATCH: {folder_i} <==> {folder_j} | CSIM: {csim:.4f}\n")
                    union(folder_i, folder_j)

    print("\n--- Phase 3: 重新命名並複製到輸出資料夾 ---")
    groups = defaultdict(list)
    for f in folders:
        groups[find(f)].append(f)

    print(f"總共整理出 {len(groups)} 個獨立的身分 (Identities)。")
    
    # 給予乾淨的流水號 ID_00001, ID_00002...
    for idx, (root_folder, group_folders) in enumerate(tqdm(groups.items(), desc="Merging and Copying")):
        new_identity_id = f"ID_{idx + 1:05d}"
        target_folder_path = os.path.join(output_base_folder, new_identity_id)
        os.makedirs(target_folder_path, exist_ok=True)
        
        for folder in group_folders:
            src_dir = os.path.join(source_folder, folder)
            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                if os.path.isfile(src_file):
                    # 檔名前綴加上原本資料夾名稱，避免合併時檔名衝突
                    new_file_name = f"{folder}_{file}"
                    dst_file = os.path.join(target_folder_path, new_file_name)
                    
                    if not os.path.exists(dst_file):
                        shutil.copy(src_file, dst_file)
                    
    print(f"\n🎉 任務完成！重新命名後的資料夾已儲存至: {output_base_folder}")

# ----------------- 執行設定 -----------------
source_folder = '/media/avlab/data/DeepFashion_Training_Final'
output_base_folder = '/media/avlab/data/DeepFashion_Training_Final_Merged'

# 執行跨資料夾合併 (加入正規化後，建議閾值設定在 0.5 到 0.6 之間)
merge_duplicate_identities(source_folder, output_base_folder, threshold=0.3)