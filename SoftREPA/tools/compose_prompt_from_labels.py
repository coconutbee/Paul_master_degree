import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compose final prompt CSV from pose and attribute labels")
    parser.add_argument("--pose_csv", required=True, help="CSV from SAM3D pose inference")
    parser.add_argument("--attr_csv", required=True, help="CSV containing image_path, gender, age, race")
    parser.add_argument("--output_csv", required=True, help="Final CSV path with image_path and prompt")
    return parser.parse_args()


def _to_float(v):
    if pd.isna(v):
        return None
    try:
        val = float(v)
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def build_yaw_desc(yaw, pronoun):
    y = _to_float(yaw)
    if y is None:
        return "facing forward"
        
    # 正值(Positive)為左(Left)，負值(Negative)為右(Right)
    if y > 54:
        return f"turned {pronoun} head to the left over the shoulder"
    if y < -54:
        return f"turned {pronoun} head to the right over the shoulder"
    if y > 20:
        return f"turned {pronoun} head to the left"
    if y < -20:
        return f"turned {pronoun} head to the right"
    else:
        return "facing forward"


def build_pitch_desc(pitch):
    p = _to_float(pitch)
    if p is None:
        return "with unknown pitch"
    if p > 25:
        return "chin up"
    if p < -25:
        return "chin down"
    return "looking straight"


def infer_gender_from_path(path):
    s = str(path).lower()
    if "woman" in s:
        return "woman", "her"
    if "man" in s:
        return "man", "his"
    return "no_gender", "their"


def clean_race(race):
    if pd.isna(race):
        return "unknown"
    r = str(race).strip().lower()
    if not r or r in {"nan", "none", "unknown"}:
        return "unknown"
    return r


def infer_label_from_mivolo(gender, age, image_path):
    g = str(gender).strip().lower() if not pd.isna(gender) else ""
    age_value = _to_float(age)

    if g in {"male", "man", "boy"}:
        if age_value is not None and age_value <= 20:
            return "boy", "his"
        return "man", "his"

    if g in {"female", "woman", "girl"}:
        if age_value is not None and age_value <= 20:
            return "girl", "her"
        return "woman", "her"

    return infer_gender_from_path(image_path)


def format_age_prefix(race_token, pronoun, age_value):
    is_male = pronoun == "his"
    child = "boy" if is_male else ("girl" if pronoun == "her" else "child")
    adult = "man" if is_male else ("woman" if pronoun == "her" else "person")
    
    if age_value is None:
        return f"A {race_token} {adult}"

    if age_value < 3:
        return f"A {race_token} baby {child}"
    elif age_value < 12:
        return f"A {race_token} young {child}"
    elif age_value < 20:
        return f"A {race_token} teenage {child}"
    
    if age_value < 30:
        decade = "twenties"
    elif age_value < 40:
        decade = "thirties"
    elif age_value < 50:
        decade = "forties"
    elif age_value < 60:
        decade = "fifties"
    elif age_value < 70:
        decade = "sixties"
    elif age_value < 80:
        decade = "seventies"
    elif age_value < 90:
        decade = "eighties"
    else:
        decade = "nineties"
        
    return f"A {race_token} {adult} in {pronoun} {decade}"


def build_prompt(image_path, race, gender, age, yaw, pitch):
    race_token = clean_race(race)
    _, pronoun = infer_label_from_mivolo(gender, age, image_path)
    age_value = _to_float(age)

    yaw_desc = build_yaw_desc(yaw, pronoun)
    pitch_desc = build_pitch_desc(pitch)
    prefix = format_age_prefix(race_token, pronoun, age_value)
    
    return f"{prefix}, {yaw_desc} and {pitch_desc}"


def main():
    args = parse_args()

    # 取得絕對路徑，方便準確比對是否為同一個檔案
    pose_path = os.path.abspath(args.pose_csv)
    attr_path = os.path.abspath(args.attr_csv)

    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"pose csv not found: {pose_path}")
    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"attr csv not found: {attr_path}")

    required_pose = {"image_path", "sam3d_head_body_yaw", "sam3d_head_pitch"}
    required_attr = {"image_path", "race", "mivolo_gender", "mivolo_age"}

    # 判斷是否使用同一個 CSV
    if pose_path == attr_path:
        print("Detected same CSV for pose and attributes. Skipping merge.")
        merged = pd.read_csv(pose_path)
        
        # 檢查單一檔案是否具備所有需要的欄位
        required_all = required_pose.union(required_attr)
        if not required_all.issubset(set(merged.columns)):
            raise KeyError(f"CSV missing columns: {required_all - set(merged.columns)}")
            
    else:
        print("Detected different CSVs. Performing merge.")
        pose_df = pd.read_csv(pose_path)
        attr_df = pd.read_csv(attr_path)

        if not required_pose.issubset(set(pose_df.columns)):
            raise KeyError(f"pose csv missing columns: {required_pose - set(pose_df.columns)}")
        if not required_attr.issubset(set(attr_df.columns)):
            raise KeyError(f"attr csv missing columns: {required_attr - set(attr_df.columns)}")

        # 預防性濾除重複欄位
        overlap_cols = ["sam3d_head_body_yaw", "sam3d_head_pitch"]
        for col in overlap_cols:
            if col in attr_df.columns:
                attr_df = attr_df.drop(columns=[col])

        merged = attr_df.merge(
            pose_df[["image_path", "sam3d_head_body_yaw", "sam3d_head_pitch"]],
            on="image_path",
            how="left",
        )

    # 執行 Prompt 生成
    merged["prompt"] = merged.apply(
        lambda r: build_prompt(
            r.get("image_path", ""),
            r.get("race", ""),
            r.get("mivolo_gender", ""), 
            r.get("mivolo_age", None),  
            r.get("sam3d_head_body_yaw", None),
            r.get("sam3d_head_pitch", None),
        ),
        axis=1,
    )

    out_df = merged[["image_path", "prompt"]].copy()
    out_df.to_csv(args.output_csv, index=False)

    print(f"Saved final prompt csv: {args.output_csv}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()