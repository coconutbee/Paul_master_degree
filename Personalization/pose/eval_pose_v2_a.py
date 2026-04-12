import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm


DEFAULT_SAM3D_MODULE = "/media/ee303/4TB/sam3-body/sam-3d-body/infer_v2.py"
DEFAULT_SAM3D_REPO_ID = "facebook/sam-3d-body-dinov3"
POSE_EVAL_LABELS = {"Frontal", "Head_Turn_Left", "Head_Turn_Right", "Back_Over_Shoulder"}


def _normalize_prompt_text(text):
    return str(text).replace("’", "'").replace("‘", "'").strip().lower()


def normalize_gt_pose_label(raw_label_or_text):
    text = _normalize_prompt_text(raw_label_or_text)
    if not text:
        return "Unknown"

    # Already in eval label space.
    for label in POSE_EVAL_LABELS:
        if text == label.lower():
            return label

    # Legacy / compatible labels.
    if text in {"head_slight_left", "side_view_left", "body_turn_left_face_front"}:
        return "Head_Turn_Left"
    if text in {"head_slight_right", "side_view_right", "body_turn_right_face_front"}:
        return "Head_Turn_Right"
    if text in {
        "head_tilt_left",
        "head_tilt_right",
        "head_tilt_down",
        "body_lean_left",
        "body_lean_right",
        "back_view_straight",
    }:
        return "Frontal"
    if text in {"back_view_side_looking_away"}:
        return "Back_Over_Shoulder"

    # Prompt phrase fallback.
    if "over the shoulder" in text or "looking back over" in text or "turns" in text and "back" in text and "shoulder" in text:
        return "Back_Over_Shoulder"

    if re.search(r"head turned to (?:his|her|his/her) left", text):
        return "Head_Turn_Left"
    if re.search(r"head turned to (?:his|her|his/her) right", text):
        return "Head_Turn_Right"
    if "looking to the left" in text or "looks to his left" in text or "looks to her left" in text:
        return "Head_Turn_Left"
    if "looking to the right" in text or "looks to his right" in text or "looks to her right" in text:
        return "Head_Turn_Right"

    if "head facing forward straight" in text or "head tilted up" in text or "head tilted down" in text:
        return "Frontal"
    if "looks straight" in text or "faces downward" in text or "looks upward" in text or "tilted back" in text:
        return "Frontal"

    return "Unknown"


def find_ref_image_by_id(ref_dir, ref_id_str):
    if not os.path.exists(ref_dir):
        return None

    for ext in [".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tif"]:
        exact_path = os.path.join(ref_dir, f"{ref_id_str}{ext}")
        if os.path.exists(exact_path):
            return exact_path

    for filename in os.listdir(ref_dir):
        if filename.startswith(f"{ref_id_str}_") or filename.startswith(f"{ref_id_str}."):
            return os.path.join(ref_dir, filename)

    return None


def find_t2i_image_by_prompt(t2i_dir, prompt):
    if not prompt or not os.path.exists(t2i_dir):
        return None

    target = _normalize_prompt_text(prompt)

    for filename in os.listdir(t2i_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif")):
            continue

        name_no_ext = os.path.splitext(filename)[0]

        if _normalize_prompt_text(name_no_ext) == target:
            return os.path.join(t2i_dir, filename)

        if "_" in name_no_ext:
            parts = name_no_ext.split("_", 1)
            if len(parts) > 1 and _normalize_prompt_text(parts[1]) == target:
                return os.path.join(t2i_dir, filename)

    return None


def load_sam3d_module(module_path):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"sam3d module not found: {module_path}")

    module_dir = os.path.dirname(os.path.abspath(module_path))
    # Ensure local 'notebook' and 'sam_3d_body' packages are resolved first.
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("sam3d_infer_v2", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Fallback: if infer_v2 failed to import setup_sam_3d_body due package
    # shadowing, attach a lazy loader that imports local notebook.utils only
    # when setup is actually invoked.
    if not hasattr(module, "setup_sam_3d_body"):
        def _lazy_setup_sam_3d_body(*args, **kwargs):
            local_utils_path = os.path.join(module_dir, "notebook", "utils.py")
            if not os.path.exists(local_utils_path):
                raise ImportError("infer_v2.py missing setup_sam_3d_body")

            utils_spec = importlib.util.spec_from_file_location("sam3d_notebook_utils", local_utils_path)
            if utils_spec is None or utils_spec.loader is None:
                raise ImportError("cannot load local notebook/utils.py")

            utils_module = importlib.util.module_from_spec(utils_spec)
            utils_spec.loader.exec_module(utils_module)

            if not hasattr(utils_module, "setup_sam_3d_body"):
                raise ImportError("local notebook/utils.py missing setup_sam_3d_body")

            return utils_module.setup_sam_3d_body(*args, **kwargs)

        module.setup_sam_3d_body = _lazy_setup_sam_3d_body

    if not hasattr(module, "setup_sam_3d_body"):
        raise ImportError("infer_v2.py missing setup_sam_3d_body")
    if not hasattr(module, "compute_sam3d_angles"):
        raise ImportError("infer_v2.py missing compute_sam3d_angles")

    return module


def classify_base_class(yaw, pitch):
    """Apply the original Y/P threshold rule provided by user."""
    y = float(yaw)
    p = float(pitch)

    if -20 <= y <= 20:
        if -10 <= p <= 10:
            return "head facing forward straight"
        if p > 10:
            return "head tilted up"
        return "head tilted down"

    if 20 < y <= 40:
        if -10 <= p <= 10:
            return "head turned to his/her left"
        if p > 10:
            return "head turned to his/her left and tilted up"
        return "head turned to his/her left and tilted down"

    if -40 <= y < -20:
        if -10 <= p <= 10:
            return "head turned to his/her right"
        if p > 10:
            return "head turned to his/her right and tilted up"
        return "head turned to his/her right and tilted down"

    if y > 40:
        if -10 <= p <= 10:
            return "head turned to his/her left over the shoulder"
        if p > 10:
            return "head turned to his/her left over the shoulder and tilted up"
        return "head turned to his/her left over the shoulder and tilted down"

    if y < -40:
        if -10 <= p <= 10:
            return "head turned to his/her right over the shoulder"
        if p > 10:
            return "head turned to his/her right over the shoulder and tilted up"
        return "head turned to his/her right over the shoulder and tilted down"

    return "Error!!!!!!!"


def map_angles_to_pose_label(head_body_yaw, head_pitch, status):
    """
    Convert infer_v2 angles to existing evaluation label space, while keeping
    the original base-class decision text for debugging.
    """
    if status in {"Not_Found", "Read_Failed", "No_Person", "Degenerate_Face_Geometry", "Degenerate_Shoulder_Geometry"}:
        return "No_Body", "No_Body"
    if head_body_yaw is None or head_pitch is None:
        return "Error", "Error!!!!!!!"

    base_class = classify_base_class(head_body_yaw, head_pitch)

    if "over the shoulder" in base_class:
        return "Back_Over_Shoulder", base_class
    if "turned to his/her left" in base_class:
        return "Head_Turn_Left", base_class
    if "turned to his/her right" in base_class:
        return "Head_Turn_Right", base_class

    # forward / tilted up / tilted down stay in frontal family per original setup.
    return "Frontal", base_class


def predict_pose_label_with_sam3d(image_path, estimator, infer_module):
    result = infer_module.compute_sam3d_angles(image_path, estimator)
    label, base_class = map_angles_to_pose_label(
        result.get("head_body_yaw"),
        result.get("head_pitch"),
        result.get("status", "Error"),
    )
    result["base_class"] = base_class
    return label, result


def process_pose_evaluation(
    task_name,
    json_path,
    t2i_dir,
    ref_dir,
    mode="t2i",
    swap_dir="",
    sam3d_module_path=DEFAULT_SAM3D_MODULE,
    sam3d_repo_id=DEFAULT_SAM3D_REPO_ID,
):
    print(f"\nProcessing Task (Pose Accuracy): [{task_name}] | Mode: {mode.upper()}")

    if not os.path.exists(json_path):
        print(f"Error: JSON not found: {json_path}")
        return None

    infer_module = load_sam3d_module(sam3d_module_path)
    estimator = infer_module.setup_sam_3d_body(hf_repo_id=sam3d_repo_id)

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"Loaded JSON with {len(data_list)} items.")

    stats = {
        "correct_t2i": 0,
        "total_t2i": 0,
        "correct_swap": 0,
        "total_swap": 0,
        "ref_not_found": 0,
        "t2i_not_found": 0,
        "swap_not_found": 0,
    }

    ref_pose_cache = {}

    for item in tqdm(data_list, desc=f"Running {task_name}"):
        img_filename = item.get("image")
        prompt_text = item.get("prompt", "")
        ref_id = str(item.get("id"))

        if not ref_id:
            continue

        item.setdefault("t2i_pose_match", 0)
        item.setdefault("t2i_pose", "Img_Missing")
        if mode == "full":
            item.setdefault("swap_pose_match", 0)
            item.setdefault("swap_pose", "Img_Missing")

        t2i_path = find_t2i_image_by_prompt(t2i_dir, prompt_text)
        if not t2i_path and img_filename:
            fallback = os.path.join(t2i_dir, img_filename)
            if os.path.exists(fallback):
                t2i_path = fallback

        gt_pose_label = None
        gt_pose_meta = {
            "head_body_yaw": None,
            "head_pitch": None,
            "status": "Prompt_GT",
            "base_class": "Prompt_GT",
        }

        if ref_dir:
            ref_path = find_ref_image_by_id(ref_dir, ref_id)
            if not ref_path:
                stats["ref_not_found"] += 1
                item["ref_pose"] = "Ref_Missing"
                item["pose_match"] = 0
                item["t2i_pose_match"] = 0
                if mode == "full":
                    item["swap_pose_match"] = 0
                continue

            if ref_id in ref_pose_cache:
                gt_pose_label, gt_pose_meta = ref_pose_cache[ref_id]
            else:
                gt_pose_label, gt_pose_meta = predict_pose_label_with_sam3d(ref_path, estimator, infer_module)
                ref_pose_cache[ref_id] = (gt_pose_label, gt_pose_meta)
        else:
            gt_pose_label = normalize_gt_pose_label(item.get("gt_pose", "Unknown"))
            if gt_pose_label == "Unknown":
                gt_pose_label = normalize_gt_pose_label(prompt_text)

        item["ref_pose"] = gt_pose_label
        item["ref_head_body_yaw"] = gt_pose_meta.get("head_body_yaw")
        item["ref_head_pitch"] = gt_pose_meta.get("head_pitch")
        item["ref_pose_status"] = gt_pose_meta.get("status")
        item["ref_pose_base_class"] = gt_pose_meta.get("base_class")

        if not t2i_path or not os.path.exists(t2i_path):
            stats["t2i_not_found"] += 1
            item["t2i_pose"] = "Img_Missing"
            item["t2i_pose_match"] = 0
            item["pose_match"] = 0
        else:
            t2i_pose_label, t2i_pose_meta = predict_pose_label_with_sam3d(t2i_path, estimator, infer_module)
            item["t2i_pose"] = t2i_pose_label
            item["t2i_head_body_yaw"] = t2i_pose_meta.get("head_body_yaw")
            item["t2i_head_pitch"] = t2i_pose_meta.get("head_pitch")
            item["t2i_pose_status"] = t2i_pose_meta.get("status")
            item["t2i_pose_base_class"] = t2i_pose_meta.get("base_class")

            is_correct_t2i = 0
            if gt_pose_label not in ["Error", "Not_Found", "No_Body"] and t2i_pose_label not in ["Error", "Not_Found", "No_Body"]:
                if str(gt_pose_label).lower() == str(t2i_pose_label).lower():
                    is_correct_t2i = 1

            item["t2i_pose_match"] = is_correct_t2i
            item["pose_match"] = is_correct_t2i

            if gt_pose_label not in ["Error", "Not_Found", "No_Body"]:
                stats["correct_t2i"] += is_correct_t2i
                stats["total_t2i"] += 1

        if mode == "full" and swap_dir:
            swap_path = os.path.join(swap_dir, img_filename) if img_filename else None
            if not swap_path or not os.path.exists(swap_path):
                stats["swap_not_found"] += 1
                item["swap_pose"] = "Img_Missing"
                item["swap_pose_match"] = 0
            else:
                swap_pose_label, swap_pose_meta = predict_pose_label_with_sam3d(swap_path, estimator, infer_module)
                item["swap_pose"] = swap_pose_label
                item["swap_head_body_yaw"] = swap_pose_meta.get("head_body_yaw")
                item["swap_head_pitch"] = swap_pose_meta.get("head_pitch")
                item["swap_pose_status"] = swap_pose_meta.get("status")
                item["swap_pose_base_class"] = swap_pose_meta.get("base_class")

                is_correct_swap = 0
                if gt_pose_label not in ["Error", "Not_Found", "No_Body"] and swap_pose_label not in ["Error", "Not_Found", "No_Body"]:
                    if str(gt_pose_label).lower() == str(swap_pose_label).lower():
                        is_correct_swap = 1

                item["swap_pose_match"] = is_correct_swap

                if gt_pose_label not in ["Error", "Not_Found", "No_Body"]:
                    stats["correct_swap"] += is_correct_swap
                    stats["total_swap"] += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    accuracy_t2i = (stats["correct_t2i"] / stats["total_t2i"] * 100) if stats["total_t2i"] > 0 else 0.0
    accuracy_swap = (stats["correct_swap"] / stats["total_swap"] * 100) if stats["total_swap"] > 0 else 0.0

    return {
        "name": task_name,
        "mode": mode,
        "accuracy_t2i": accuracy_t2i,
        "valid_samples_t2i": stats["total_t2i"],
        "accuracy_swap": accuracy_swap,
        "valid_samples_swap": stats["total_swap"],
        "ref_missing": stats["ref_not_found"],
        "t2i_missing": stats["t2i_not_found"],
        "swap_missing": stats["swap_not_found"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2I/Swap Pose Evaluation against Reference (SAM3D infer_v2 backend)")
    parser.add_argument("--json", type=str, required=True, help="Input/Output JSON file")
    parser.add_argument("--t2i", type=str, required=True, help="T2I images folder")
    parser.add_argument("--ref", type=str, default="", help="Optional reference images folder")
    parser.add_argument("--swap", type=str, default="", help="Swap images folder")
    parser.add_argument("--mode", type=str, default="t2i", choices=["full", "t2i"], help="Evaluation mode")
    parser.add_argument("--name", type=str, default="pose_eval", help="Task name")
    parser.add_argument("--sam3d_module", type=str, default=DEFAULT_SAM3D_MODULE, help="Path to infer_v2.py")
    parser.add_argument("--sam3d_repo_id", type=str, default=DEFAULT_SAM3D_REPO_ID, help="HF repo id for setup_sam_3d_body")

    args = parser.parse_args()

    res = process_pose_evaluation(
        task_name=args.name,
        json_path=args.json,
        t2i_dir=args.t2i,
        ref_dir=args.ref,
        mode=args.mode,
        swap_dir=args.swap,
        sam3d_module_path=args.sam3d_module,
        sam3d_repo_id=args.sam3d_repo_id,
    )

    if res:
        print("\n" + "=" * 80)
        print(f"Task: {res['name']}")
        pose_target = "Ref" if args.ref else "Prompt GT"
        print(f"Pose Accuracy (T2I vs {pose_target}): {res['accuracy_t2i']:.2f}%")
        print(f"Valid T2I Comparisons: {res['valid_samples_t2i']}")
        if args.mode == "full":
            print(f"Pose Accuracy (Swap vs {pose_target}): {res['accuracy_swap']:.2f}%")
            print(f"Valid Swap Comparisons: {res['valid_samples_swap']}")
        if args.ref and res["ref_missing"] > 0:
            print(f"Warning: Reference images missing: {res['ref_missing']}")
        if res["t2i_missing"] > 0:
            print(f"Warning: T2I images missing: {res['t2i_missing']}")
        if args.mode == "full" and res["swap_missing"] > 0:
            print(f"Warning: swapped images missing: {res['swap_missing']}")
        print("=" * 80)
        print(f"Results updated in: {args.json}")
