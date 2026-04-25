import argparse
import csv
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from notebook.utils import setup_sam_3d_body
    HAS_SAM3D = True
except ImportError:
    HAS_SAM3D = False

# ================= 關鍵點索引 =================
IDX_NOSE, IDX_LEFT_EYE, IDX_RIGHT_EYE = 0, 1, 2
IDX_LEFT_EAR, IDX_RIGHT_EAR = 3, 4
IDX_LEFT_SHOULDER_SAM, IDX_RIGHT_SHOULDER_SAM = 5, 6
IDX_NECK = 69

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif"}
EPS = 1e-8
FACE_CONF_MIN = 0.03
SHOULDER_CONF_MIN = 0.03

# v4: yaw 採用 v2，pitch 採用 v3
PITCH_OFFSET_DEG = 35.0

# ================= 幾何與數學函數 =================
def limit_angle(angle):
    while angle < -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def _safe_float(v):
    return float(v) if v is not None else ""


def _unit_vec(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    return np.zeros_like(vec) if norm < eps else vec / norm


def _is_valid_point(pt):
    return pt is not None and np.asarray(pt).shape[0] >= 3 and np.all(np.isfinite(np.asarray(pt)[:3]))


def _orient_normal_for_consistency(unit_vec):
    return -unit_vec if unit_vec[2] < 0 else unit_vec


def _signed_angle_yaw_deg(vec_head, vec_body, eps=1e-8):
    head_xz = np.array([vec_head[0], vec_head[2]])
    body_xz = np.array([vec_body[0], vec_body[2]])
    norm_h, norm_b = np.linalg.norm(head_xz), np.linalg.norm(body_xz)
    if norm_h < eps or norm_b < eps:
        return 0.0

    head_xz, body_xz = head_xz / norm_h, body_xz / norm_b
    det = body_xz[0] * head_xz[1] - body_xz[1] * head_xz[0]
    dot = np.clip(body_xz[0] * head_xz[0] + body_xz[1] * head_xz[1], -1.0, 1.0)
    return float(np.degrees(np.arctan2(det, dot)))


# ================= 核心推論邏輯 =================
def compute_sam3d_angles(image_path, estimator):
    if not image_path or not os.path.exists(image_path):
        return {"head_body_yaw": None, "head_pitch": None, "status": "Not_Found", "person_count": 0}

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"head_body_yaw": None, "head_pitch": None, "status": "Read_Failed", "person_count": 0}

        outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        person_count = len(outputs) if outputs else 0
        if not outputs:
            return {"head_body_yaw": None, "head_pitch": None, "status": "No_Person", "person_count": person_count}

        kpts_3d = outputs[0]["pred_keypoints_3d"]
        nose = kpts_3d[IDX_NOSE][:3]
        left_eye, right_eye = kpts_3d[IDX_LEFT_EYE][:3], kpts_3d[IDX_RIGHT_EYE][:3]
        left_ear, right_ear = kpts_3d[IDX_LEFT_EAR][:3], kpts_3d[IDX_RIGHT_EAR][:3]
        left_shoulder, right_shoulder = kpts_3d[IDX_LEFT_SHOULDER_SAM][:3], kpts_3d[IDX_RIGHT_SHOULDER_SAM][:3]
        neck = kpts_3d[IDX_NECK][:3]

        # ===== Yaw: 採用 v2 =====
        cross_face = np.cross(left_eye - nose, right_eye - nose)
        cross_shoulder = np.cross(right_shoulder - neck, left_shoulder - neck)
        if np.linalg.norm(cross_face) / (np.linalg.norm(left_eye - nose) * np.linalg.norm(right_eye - nose) + EPS) < FACE_CONF_MIN:
            return {
                "head_body_yaw": None,
                "head_pitch": None,
                "status": "Degenerate_Face_Geometry",
                "person_count": person_count,
            }
        if np.linalg.norm(cross_shoulder) / (
            np.linalg.norm(right_shoulder - neck) * np.linalg.norm(left_shoulder - neck) + EPS
        ) < SHOULDER_CONF_MIN:
            return {
                "head_body_yaw": None,
                "head_pitch": None,
                "status": "Degenerate_Shoulder_Geometry",
                "person_count": person_count,
            }

        head_center = (nose + left_eye + right_eye) / 3.0
        body_center = (neck + left_shoulder + right_shoulder) / 3.0
        use_ears = _is_valid_point(left_ear) and _is_valid_point(right_ear)
        ref_left, ref_right = (left_ear, right_ear) if use_ears else (left_eye, right_eye)

        head_vec = _orient_normal_for_consistency(_unit_vec(np.cross(ref_right - ref_left, nose - head_center)))
        body_vec = _orient_normal_for_consistency(_unit_vec(np.cross(right_shoulder - left_shoulder, neck - body_center)))

        head_body_yaw = _signed_angle_yaw_deg(head_vec, body_vec)
        if abs(head_body_yaw) > 95.0:
            head_body_yaw += 180.0
        head_body_yaw = limit_angle(head_body_yaw)

        # ===== Pitch: 採用 v3 =====
        mid_eye = (left_eye + right_eye) / 2.0
        v_x = _unit_vec(left_eye - right_eye)
        v_y_temp = _unit_vec(neck - mid_eye)
        v_z = _unit_vec(np.cross(v_y_temp, v_x))

        v_nose_ref = _unit_vec(nose - mid_eye)
        pitch_flip_corrected = np.dot(v_z, v_nose_ref) < 0
        if pitch_flip_corrected:
            v_z = -v_z

        head_pitch_abs = float(np.degrees(np.arctan2(-v_z[1], np.abs(v_z[2]) + EPS)))
        head_pitch = limit_angle(head_pitch_abs + PITCH_OFFSET_DEG)

        status = "OK" if use_ears else "Fallback_Eyes_For_Yaw"
        if pitch_flip_corrected:
            status += "|PitchFlipCorrected"

        return {
            "head_body_yaw": head_body_yaw,
            "head_pitch": head_pitch,
            "status": status,
            "person_count": person_count,
        }

    except Exception as e:
        return {"head_body_yaw": None, "head_pitch": None, "status": f"Error: {e}", "person_count": 0}


# ================= 批次執行 =================
def run_sam3d_inference(input_dir, output_csv, sam3d_repo_id, max_images=0):
    if not HAS_SAM3D:
        raise ImportError("Cannot import setup_sam_3d_body.")

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                image_paths.append(os.path.join(root, f))
    image_paths.sort()

    if max_images > 0:
        image_paths = image_paths[:max_images]

    done_paths = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "image_path" in reader.fieldnames:
                done_paths.update(row.get("image_path", "") for row in reader)

    remaining_paths = [p for p in image_paths if p not in done_paths]
    print(f"📊 總計: {len(image_paths)}, 已完成: {len(done_paths)}, 剩餘: {len(remaining_paths)}")
    if not remaining_paths:
        return 0

    estimator = setup_sam_3d_body(hf_repo_id=sam3d_repo_id)

    mode = "a" if os.path.exists(output_csv) and len(done_paths) > 0 else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "sam3d_head_body_yaw",
                "sam3d_head_pitch",
                "sam3d_status",
                "sam3d_person_count",
            ],
        )
        if mode == "w":
            writer.writeheader()

        for img_path in tqdm(remaining_paths, desc="提取 SAM3D 角度 (v4: yaw=v2, pitch=v3)"):
            res = compute_sam3d_angles(img_path, estimator)
            writer.writerow(
                {
                    "image_path": img_path,
                    "sam3d_head_body_yaw": _safe_float(res["head_body_yaw"]),
                    "sam3d_head_pitch": _safe_float(res["head_pitch"]),
                    "sam3d_status": res["status"],
                    "sam3d_person_count": int(res["person_count"]),
                }
            )
            f.flush()

    return len(remaining_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="輸入圖片資料夾")
    parser.add_argument("--output_csv", type=str, default="sam3d_v4_results.csv", help="輸出的 CSV 檔名")
    args = parser.parse_args()

    run_sam3d_inference(args.input_dir, args.output_csv, "facebook/sam-3d-body-dinov3")
    print(f"\n✅ 執行完畢，結果已儲存至: {args.output_csv}")
