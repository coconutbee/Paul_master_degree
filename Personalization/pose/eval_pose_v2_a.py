import argparse
import importlib.util
import json
import os
import re
import sys

from tqdm import tqdm


DEFAULT_SAM3D_MODULE = "/media/ee303/4tb/Will/Paul_master_degree/sam3-body/sam-3d-body/infer_v2.py"
DEFAULT_SAM3D_REPO_ID = "facebook/sam-3d-body-dinov3"
CANONICAL_POSE_LABELS = {
    "head facing forward straight",
    "head tilted up",
    "head tilted down",
    "head turned to his/her left",
    "head turned to his/her left and tilted up",
    "head turned to his/her left and tilted down",
    "head turned to his/her right",
    "head turned to his/her right and tilted up",
    "head turned to his/her right and tilted down",
    "head turned to his/her left over the shoulder",
    "head turned to his/her left over the shoulder and tilted up",
    "head turned to his/her left over the shoulder and tilted down",
    "head turned to his/her right over the shoulder",
    "head turned to his/her right over the shoulder and tilted up",
    "head turned to his/her right over the shoulder and tilted down",
}
COARSE_POSE_FAMILIES = {
    "Frontal": {
        "head facing forward straight",
        "head tilted up",
        "head tilted down",
    },
    "Head_Turn_Left": {
        "head turned to his/her left",
        "head turned to his/her left and tilted up",
        "head turned to his/her left and tilted down",
    },
    "Head_Turn_Right": {
        "head turned to his/her right",
        "head turned to his/her right and tilted up",
        "head turned to his/her right and tilted down",
    },
    "Back_Over_Shoulder": {
        "head turned to his/her left over the shoulder",
        "head turned to his/her left over the shoulder and tilted up",
        "head turned to his/her left over the shoulder and tilted down",
        "head turned to his/her right over the shoulder",
        "head turned to his/her right over the shoulder and tilted up",
        "head turned to his/her right over the shoulder and tilted down",
    },
}
POSE_EVAL_LABELS = CANONICAL_POSE_LABELS | set(COARSE_POSE_FAMILIES)
BODY_MISSING_STATUSES = {
    "Not_Found",
    "Read_Failed",
    "No_Person",
    "Degenerate_Face_Geometry",
    "Degenerate_Shoulder_Geometry",
}
NON_COMPARABLE_LABELS = {"Error", "Not_Found", "No_Body", "Img_Missing", "Ref_Missing", "Unknown"}
LEGACY_LABEL_MAP = {
    "head_slight_left": "head turned to his/her left",
    "side_view_left": "head turned to his/her left",
    "body_turn_left_face_front": "head turned to his/her left",
    "head_slight_right": "head turned to his/her right",
    "side_view_right": "head turned to his/her right",
    "body_turn_right_face_front": "head turned to his/her right",
    "head_tilt_up": "head tilted up",
    "head_tilt_down": "head tilted down",
    "head_tilt_left": "Frontal",
    "head_tilt_right": "Frontal",
    "body_lean_left": "Frontal",
    "body_lean_right": "Frontal",
    "back_view_straight": "Frontal",
    "back_view_side_looking_away": "Back_Over_Shoulder",
}


def _normalize_prompt_text(text):
    return str(text).replace("’", "'").replace("‘", "'").strip().lower()


def _text_has_any(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def _canonical_pose_label(direction=None, pitch="straight", over_shoulder=False):
    if direction is None:
        if pitch == "up":
            return "head tilted up"
        if pitch == "down":
            return "head tilted down"
        return "head facing forward straight"

    label = f"head turned to his/her {direction}"
    if over_shoulder:
        label += " over the shoulder"
    if pitch == "up":
        label += " and tilted up"
    elif pitch == "down":
        label += " and tilted down"
    return label


def _extract_direction(text):
    yaw_patterns = {
        "left": [
            r"head turned to (?:his|her|their|his/her) left",
            r"(?:turn|look|face|glance|gaze)\w*[^.]{0,40}\bleft\b",
            r"\bsideways toward the left\b",
            r"\bto (?:his|her|their) left\b",
            r"\bover (?:his|her|their) left shoulder\b",
            r"\bover the left shoulder\b",
            r"\bto the left\b",
        ],
        "right": [
            r"head turned to (?:his|her|their|his/her) right",
            r"(?:turn|look|face|glance|gaze)\w*[^.]{0,40}\bright\b",
            r"\bsideways toward the right\b",
            r"\bto (?:his|her|their) right\b",
            r"\bover (?:his|her|their) right shoulder\b",
            r"\bover the right shoulder\b",
            r"\bto the right\b",
        ],
    }
    roll_only = _text_has_any(
        text,
        [
            r"(?:tilt|lean)\w*[^.]{0,25}\bleft\b",
            r"(?:tilt|lean)\w*[^.]{0,25}\bright\b",
        ],
    ) and not _text_has_any(
        text,
        [
            r"(?:turn|look|face|glance|gaze)\w*[^.]{0,40}\bleft\b",
            r"(?:turn|look|face|glance|gaze)\w*[^.]{0,40}\bright\b",
            r"over (?:his|her|their)? ?shoulder",
        ],
    )
    if roll_only:
        return None

    for direction, patterns in yaw_patterns.items():
        if _text_has_any(text, patterns):
            return direction
    return None


def _extract_pitch(text):
    if _text_has_any(
        text,
        [
            r"head facing forward straight",
            r"looks straight",
            r"looking straight",
            r"faces? forward",
            r"facing forward",
        ],
    ):
        return "straight"

    if _text_has_any(
        text,
        [
            r"head tilted up",
            r"tilted up",
            r"looks upward",
            r"looking upward",
            r"looks up",
            r"looking up",
            r"upward",
            r"tilted back",
            r"tilts? (?:his|her|their)? head backward",
            r"head slightly raised",
            r"face(?:s|d|ing)?[^.]{0,20}upward",
            r"raises? (?:his|her|their)? head",
        ],
    ):
        return "up"

    if _text_has_any(
        text,
        [
            r"head tilted down",
            r"tilted down",
            r"looks downward",
            r"looking downward",
            r"looks down",
            r"looking down",
            r"faces?[^.]{0,20}downward",
            r"facing[^.]{0,20}downward",
            r"glancing downward",
            r"gazing downward",
            r"downward",
            r"lowered gaze",
            r"head bowed",
        ],
    ):
        return "down"

    return "straight"


def _has_roll_only_tilt(text):
    return _text_has_any(
        text,
        [
            r"head tilted left",
            r"head tilted right",
            r"tilts? (?:his|her|their)? head left",
            r"tilts? (?:his|her|their)? head right",
            r"leans? (?:his|her|their)? head left",
            r"leans? (?:his|her|their)? head right",
            r"leans? (?:his|her|their)? head toward (?:his|her|their)? ?left shoulder",
            r"leans? (?:his|her|their)? head toward (?:his|her|their)? ?right shoulder",
            r"with (?:his|her|their)? head tilted left",
            r"with (?:his|her|their)? head tilted right",
        ],
    )


def parse_pose_label(raw_label_or_text):
    text = _normalize_prompt_text(raw_label_or_text)
    if not text:
        return "Unknown"

    for label in POSE_EVAL_LABELS:
        if text == label.lower():
            return label

    legacy_label = LEGACY_LABEL_MAP.get(text)
    if legacy_label:
        return legacy_label

    if re.search(r"head turned to (?:his|her|their|his/her) left", text):
        direction = "left"
    elif re.search(r"head turned to (?:his|her|their|his/her) right", text):
        direction = "right"
    else:
        direction = _extract_direction(text)

    over_shoulder = _text_has_any(
        text,
        [
            r"over the shoulder",
            r"over (?:his|her|their)(?: left| right)? shoulder",
            r"over the (?:left|right) shoulder",
            r"back over (?:his|her|their)? ?shoulder",
            r"looking back over",
        ],
    )
    pitch = _extract_pitch(text)

    if over_shoulder and direction is None:
        return "Back_Over_Shoulder"
    if direction is not None:
        return _canonical_pose_label(direction=direction, pitch=pitch, over_shoulder=over_shoulder)

    if pitch != "straight":
        return _canonical_pose_label(pitch=pitch)

    if _has_roll_only_tilt(text):
        return "head facing forward straight"

    if _text_has_any(
        text,
        [
            r"face(?:s|d|ing)? forward",
            r"looks straight",
            r"looking straight",
            r"head facing forward straight",
            r"frontal",
        ],
    ):
        return "head facing forward straight"

    return "Unknown"


def normalize_gt_pose_label(raw_label_or_text, prompt_fallback=""):
    primary = parse_pose_label(raw_label_or_text)
    fallback = parse_pose_label(prompt_fallback)

    if fallback in CANONICAL_POSE_LABELS:
        return fallback
    if primary in COARSE_POSE_FAMILIES and fallback in COARSE_POSE_FAMILIES[primary]:
        return fallback
    if primary != "Unknown":
        return primary
    if fallback != "Unknown":
        return fallback
    return "Unknown"


def _pose_label_components(label):
    parsed = parse_pose_label(label)
    if parsed not in CANONICAL_POSE_LABELS:
        return None

    if "his/her left" in parsed:
        direction = "left"
    elif "his/her right" in parsed:
        direction = "right"
    else:
        direction = None

    if "tilted up" in parsed:
        pitch = "up"
    elif "tilted down" in parsed:
        pitch = "down"
    else:
        pitch = "straight"

    return {
        "direction": direction,
        "pitch": pitch,
        "over_shoulder": "over the shoulder" in parsed,
    }


def pose_labels_match(lhs, rhs):
    left = parse_pose_label(lhs)
    right = parse_pose_label(rhs)

    if left in NON_COMPARABLE_LABELS or right in NON_COMPARABLE_LABELS:
        return False
    if left == right:
        return True

    left_parts = _pose_label_components(left)
    right_parts = _pose_label_components(right)
    if (
        left_parts
        and right_parts
        and left_parts["direction"] is None
        and not left_parts["over_shoulder"]
        and left_parts["pitch"] in {"up", "down"}
        and right_parts["pitch"] == left_parts["pitch"]
    ):
        return True
    if (
        left_parts
        and right_parts
        and left_parts["direction"] is not None
        and left_parts["direction"] == right_parts["direction"]
        and left_parts["pitch"] == right_parts["pitch"]
        and not left_parts["over_shoulder"]
        and right_parts["over_shoulder"]
    ):
        return True

    return False


def is_valid_pose_label(label):
    return parse_pose_label(label) not in NON_COMPARABLE_LABELS


def normalize_prediction_pose_label(existing_label, base_class=None, status="OK"):
    if existing_label == "Img_Missing":
        return "Img_Missing"
    if existing_label == "Ref_Missing":
        return "Ref_Missing"
    if status in BODY_MISSING_STATUSES:
        return "No_Body"
    if existing_label in {"Error", "Not_Found", "No_Body"}:
        return existing_label

    if base_class and base_class not in {"Prompt_GT", "No_Body", "Error!!!!!!!"}:
        parsed_base = parse_pose_label(base_class)
        if parsed_base != "Unknown":
            return parsed_base

    parsed_label = parse_pose_label(existing_label)
    if parsed_label != "Unknown":
        return parsed_label
    return existing_label or "Unknown"


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
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("sam3d_infer_v2", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    """Match the yaw/pitch thresholds used in sam3-body/label.py."""
    y = float(yaw)
    p = float(pitch)

    if -20 <= y <= 20:
        if -25 <= p <= 25:
            return "head facing forward straight"
        if p > 25:
            return "head tilted up"
        return "head tilted down"

    if 20 < y <= 40:
        if -25 <= p <= 25:
            return "head turned to his/her left"
        if p > 25:
            return "head turned to his/her left and tilted up"
        return "head turned to his/her left and tilted down"

    if -40 <= y < -20:
        if -25 <= p <= 25:
            return "head turned to his/her right"
        if p > 25:
            return "head turned to his/her right and tilted up"
        return "head turned to his/her right and tilted down"

    if y > 40:
        if -25 <= p <= 25:
            return "head turned to his/her left over the shoulder"
        if p > 25:
            return "head turned to his/her left over the shoulder and tilted up"
        return "head turned to his/her left over the shoulder and tilted down"

    if y < -40:
        if -25 <= p <= 25:
            return "head turned to his/her right over the shoulder"
        if p > 25:
            return "head turned to his/her right over the shoulder and tilted up"
        return "head turned to his/her right over the shoulder and tilted down"

    return "Error!!!!!!!"


def map_angles_to_pose_label(head_body_yaw, head_pitch, status):
    """
    Keep the full pose label instead of collapsing into 4 coarse buckets so
    left/right over-shoulder and up/down information are preserved.
    """
    if status in BODY_MISSING_STATUSES:
        return "No_Body", "No_Body"
    if head_body_yaw is None or head_pitch is None:
        return "Error", "Error!!!!!!!"

    base_class = classify_base_class(head_body_yaw, head_pitch)
    return base_class, base_class


def predict_pose_label_with_sam3d(image_path, estimator, infer_module):
    result = infer_module.compute_sam3d_angles(image_path, estimator)
    label, base_class = map_angles_to_pose_label(
        result.get("head_body_yaw"),
        result.get("head_pitch"),
        result.get("status", "Error"),
    )
    result["base_class"] = base_class
    return label, result


def recompute_pose_from_stored_angles(item, prefix):
    status = item.get(f"{prefix}_pose_status", "OK")
    yaw = item.get(f"{prefix}_head_body_yaw")
    pitch = item.get(f"{prefix}_head_pitch")

    if status == "Prompt_GT":
        label = normalize_gt_pose_label(
            item.get("gt_pose", "Unknown"),
            item.get("prompt", ""),
        )
        item[f"{prefix}_pose"] = label
        item[f"{prefix}_pose_base_class"] = label
        return label

    if status in BODY_MISSING_STATUSES:
        item[f"{prefix}_pose"] = "No_Body"
        item[f"{prefix}_pose_base_class"] = "No_Body"
        return "No_Body"

    if yaw is not None and pitch is not None:
        label, base_class = map_angles_to_pose_label(yaw, pitch, status)
        item[f"{prefix}_pose"] = label
        item[f"{prefix}_pose_base_class"] = base_class
        return label

    label = normalize_prediction_pose_label(
        item.get(f"{prefix}_pose"),
        item.get(f"{prefix}_pose_base_class"),
        status,
    )
    item[f"{prefix}_pose"] = label
    if item.get(f"{prefix}_pose_base_class") not in {"Prompt_GT", "No_Body", "Error!!!!!!!"}:
        item[f"{prefix}_pose_base_class"] = label
    return label


def relabel_pose_metadata_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    stats = {
        "name": os.path.splitext(os.path.basename(json_path))[0],
        "mode": "full" if any("swap_pose" in item for item in data_list) else "t2i",
        "accuracy_t2i": 0.0,
        "valid_samples_t2i": 0,
        "accuracy_swap": 0.0,
        "valid_samples_swap": 0,
    }
    correct_t2i = 0
    total_t2i = 0
    correct_swap = 0
    total_swap = 0

    for item in data_list:
        prompt_text = item.get("prompt", "")
        prompt_gt_label = normalize_gt_pose_label(item.get("gt_pose", "Unknown"), prompt_text)

        if item.get("ref_pose_status") == "Prompt_GT":
            ref_pose_label = prompt_gt_label
            item["ref_pose"] = ref_pose_label
            item["ref_pose_base_class"] = ref_pose_label
        else:
            ref_pose_label = recompute_pose_from_stored_angles(item, "ref")
            if ref_pose_label not in NON_COMPARABLE_LABELS:
                item["ref_pose"] = ref_pose_label

        gt_pose_label = prompt_gt_label

        item["gt_pose"] = gt_pose_label

        t2i_pose_label = recompute_pose_from_stored_angles(item, "t2i")
        is_correct_t2i = int(pose_labels_match(gt_pose_label, t2i_pose_label))
        item["t2i_pose_match"] = is_correct_t2i
        item["pose_match"] = is_correct_t2i
        if is_valid_pose_label(gt_pose_label):
            total_t2i += 1
            correct_t2i += is_correct_t2i

        if any(key in item for key in ("swap_pose", "swap_pose_status", "swap_pose_base_class", "swap_pose_match")):
            swap_pose_label = recompute_pose_from_stored_angles(item, "swap")
            is_correct_swap = int(pose_labels_match(gt_pose_label, swap_pose_label))
            item["swap_pose_match"] = is_correct_swap
            if is_valid_pose_label(gt_pose_label):
                total_swap += 1
                correct_swap += is_correct_swap

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    stats["accuracy_t2i"] = (correct_t2i / total_t2i * 100) if total_t2i else 0.0
    stats["valid_samples_t2i"] = total_t2i
    stats["accuracy_swap"] = (correct_swap / total_swap * 100) if total_swap else 0.0
    stats["valid_samples_swap"] = total_swap
    return stats


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
        prompt_gt_label = normalize_gt_pose_label(item.get("gt_pose", "Unknown"), prompt_text)

        if not ref_id:
            continue

        item.setdefault("t2i_pose_match", 0)
        item.setdefault("t2i_pose", "Img_Missing")
        if mode == "full":
            item.setdefault("swap_pose_match", 0)
            item.setdefault("swap_pose", "Img_Missing")

        t2i_path = item.get("image_path") or find_t2i_image_by_prompt(t2i_dir, prompt_text)
        if not t2i_path and img_filename:
            fallback = os.path.join(t2i_dir, img_filename)
            if os.path.exists(fallback):
                t2i_path = fallback

        gt_pose_meta = {
            "head_body_yaw": None,
            "head_pitch": None,
            "status": "Prompt_GT",
            "base_class": "Prompt_GT",
        }
        ref_pose_label = prompt_gt_label

        if ref_dir:
            ref_path = item.get("ref_image") or find_ref_image_by_id(ref_dir, ref_id)
            if not ref_path:
                stats["ref_not_found"] += 1
                item["ref_pose"] = "Ref_Missing"
                item["pose_match"] = 0
                item["t2i_pose_match"] = 0
                if mode == "full":
                    item["swap_pose_match"] = 0
                continue

            if ref_id in ref_pose_cache:
                ref_pose_label, gt_pose_meta = ref_pose_cache[ref_id]
            else:
                ref_pose_label, gt_pose_meta = predict_pose_label_with_sam3d(ref_path, estimator, infer_module)
                ref_pose_cache[ref_id] = (ref_pose_label, gt_pose_meta)

        gt_pose_label = prompt_gt_label

        item["gt_pose"] = gt_pose_label
        item["ref_pose"] = ref_pose_label
        item["ref_head_body_yaw"] = gt_pose_meta.get("head_body_yaw")
        item["ref_head_pitch"] = gt_pose_meta.get("head_pitch")
        item["ref_pose_status"] = gt_pose_meta.get("status")
        item["ref_pose_base_class"] = gt_pose_meta.get("base_class") if ref_dir else ref_pose_label

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

            is_correct_t2i = int(pose_labels_match(gt_pose_label, t2i_pose_label))
            item["t2i_pose_match"] = is_correct_t2i
            item["pose_match"] = is_correct_t2i

            if is_valid_pose_label(gt_pose_label):
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

                is_correct_swap = int(pose_labels_match(gt_pose_label, swap_pose_label))
                item["swap_pose_match"] = is_correct_swap

                if is_valid_pose_label(gt_pose_label):
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
    parser.add_argument("--t2i", type=str, default="", help="T2I images folder")
    parser.add_argument("--ref", type=str, default="", help="Optional reference images folder")
    parser.add_argument("--swap", type=str, default="", help="Swap images folder")
    parser.add_argument("--mode", type=str, default="t2i", choices=["full", "t2i"], help="Evaluation mode")
    parser.add_argument("--name", type=str, default="pose_eval", help="Task name")
    parser.add_argument("--sam3d_module", type=str, default=DEFAULT_SAM3D_MODULE, help="Path to infer_v2.py")
    parser.add_argument("--sam3d_repo_id", type=str, default=DEFAULT_SAM3D_REPO_ID, help="HF repo id for setup_sam_3d_body")
    parser.add_argument("--relabel-only", action="store_true", help="Reuse stored prompt/base_class labels without rerunning SAM3D")

    args = parser.parse_args()

    if args.relabel_only:
        res = relabel_pose_metadata_json(args.json)
    else:
        if not args.t2i:
            parser.error("--t2i is required unless --relabel-only is set")
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
        if args.relabel_only:
            print(f"Relabel-only T2I Pose Accuracy: {res['accuracy_t2i']:.2f}%")
            print(f"Relabel-only valid T2I samples: {res['valid_samples_t2i']}")
            if res['valid_samples_swap']:
                print(f"Relabel-only Swap Pose Accuracy: {res['accuracy_swap']:.2f}%")
                print(f"Relabel-only valid Swap samples: {res['valid_samples_swap']}")
        else:
            pose_target = "Ref" if args.ref else "Prompt GT"
            print(f"Pose Accuracy (T2I vs {pose_target}): {res['accuracy_t2i']:.2f}%")
            print(f"Valid T2I Comparisons: {res['valid_samples_t2i']}")
            if args.mode == "full":
                print(f"Pose Accuracy (Swap vs {pose_target}): {res['accuracy_swap']:.2f}%")
                print(f"Valid Swap Comparisons: {res['valid_samples_swap']}")
            if args.ref and res.get("ref_missing", 0) > 0:
                print(f"Warning: Reference images missing: {res['ref_missing']}")
            if res.get("t2i_missing", 0) > 0:
                print(f"Warning: T2I images missing: {res['t2i_missing']}")
            if args.mode == "full" and res.get("swap_missing", 0) > 0:
                print(f"Warning: swapped images missing: {res['swap_missing']}")
        print("=" * 80)
        print(f"Results updated in: {args.json}")
