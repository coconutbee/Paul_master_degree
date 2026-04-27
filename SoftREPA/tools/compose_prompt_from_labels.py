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
    try:
        return float(v)
    except Exception:
        return None


def build_yaw_desc(yaw, pronoun):
    y = _to_float(yaw)
    if y is None:
        return "with unknown yaw"
    if y > 40:
        return f"turned {pronoun} head to {pronoun} left over the shoulder"
    if y < -40:
        return f"turned {pronoun} head to {pronoun} right over the shoulder"
    if y > 20:
        return f"turned {pronoun} head to {pronoun} left"
    if y < -20:
        return f"turned {pronoun} head to {pronoun} right"
    else:
        return "facing forward"


def build_pitch_desc(pitch):
    p = _to_float(pitch)
    if p is None:
        return "with unknown pitch"
    if p > 25:
        return "tilted up"
    if p < -25:
        return "tilted down"
    return "looking straight"


def infer_gender_from_path(path):
    s = str(path).lower()
    if "woman" in s:
        return "woman", "her"
    if "man" in s:
        return "man", "his"
    return "no_gender", "their"


def clean_race(race):
    r = str(race).strip().lower()
    if not r or r in {"nan", "none", "unknown"}:
        return "unknown"
    return r


def infer_label_from_mivolo(gender, age, image_path):
    g = str(gender).strip().lower()
    age_value = _to_float(age)

    if g in {"male", "man", "boy"}:
        if age_value is not None and age_value <= 20:
            return "boy", "his"
        return "man", "his"

    if g in {"female", "woman", "girl"}:
        if age_value is not None and age_value <= 20:
            return "girl", "her"
        return "woman", "her"

    # Fallback when MiVOLO gender is invalid.
    return infer_gender_from_path(image_path)


def build_prompt(image_path, race, gender, age, yaw, pitch):
    race_token = clean_race(race)
    gender_token, pronoun = infer_label_from_mivolo(gender, age, image_path)
    age_value = _to_float(age)
    if age_value is None:
        age_phrase = "unknown age"
    else:
        age_phrase = f"{age_value:.1f}"

    yaw_desc = build_yaw_desc(yaw, pronoun)
    pitch_desc = build_pitch_desc(pitch)
    return f"A {race_token} {gender_token} in {age_phrase} years old, {yaw_desc} and {pitch_desc}"


def main():
    args = parse_args()

    if not os.path.exists(args.pose_csv):
        raise FileNotFoundError(f"pose csv not found: {args.pose_csv}")
    if not os.path.exists(args.attr_csv):
        raise FileNotFoundError(f"attr csv not found: {args.attr_csv}")

    pose_df = pd.read_csv(args.pose_csv)
    attr_df = pd.read_csv(args.attr_csv)

    required_pose = {"image_path", "sam3d_head_body_yaw", "sam3d_head_pitch"}
    required_attr = {"image_path", "race", "gender", "age"}
    if not required_pose.issubset(set(pose_df.columns)):
        raise KeyError(f"pose csv missing columns: {required_pose - set(pose_df.columns)}")
    if not required_attr.issubset(set(attr_df.columns)):
        raise KeyError(f"attr csv missing columns: {required_attr - set(attr_df.columns)}")

    merged = attr_df.merge(
        pose_df[["image_path", "sam3d_head_body_yaw", "sam3d_head_pitch"]],
        on="image_path",
        how="left",
    )

    merged["prompt"] = merged.apply(
        lambda r: build_prompt(
            r.get("image_path", ""),
            r.get("race", ""),
            r.get("gender", ""),
            r.get("age", None),
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
