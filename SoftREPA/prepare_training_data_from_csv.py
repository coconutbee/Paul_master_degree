import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


@dataclass
class Sample:
    source_path: Path
    caption: str
    status: str
    person_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert image-caption CSV to COCO-format dataset for SoftREPA training."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root. Script creates output_root/<dataset_name>/{train2017,val2017,annotations}",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="deepfashion",
        help="Dataset folder name under output_root (e.g., deepfashion)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio in [0, 1]. Remaining goes to val",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--caption_col", type=str, default="caption", help="CSV column name for caption"
    )
    parser.add_argument(
        "--image_col", type=str, default="image_path", help="CSV column name for image path"
    )
    parser.add_argument(
        "--status_col",
        type=str,
        default="sam3d_status",
        help="CSV column name for status (optional)",
    )
    parser.add_argument(
        "--person_count_col",
        type=str,
        default="sam3d_person_count",
        help="CSV column name for person count (optional)",
    )
    parser.add_argument(
        "--keep_status",
        type=str,
        default="OK",
        help="Keep rows whose status equals this value; set empty string to disable",
    )
    parser.add_argument(
        "--require_person_count",
        type=int,
        default=1,
        help="Keep rows with this person_count; set negative value to disable",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy", "hardlink"],
        help="How images are materialized into coco/train2017 and coco/val2017",
    )
    parser.add_argument(
        "--min_caption_len",
        type=int,
        default=3,
        help="Drop captions shorter than this many non-space characters",
    )
    return parser.parse_args()


def safe_int(value: str, fallback: int = -1) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return fallback


def read_samples(args: argparse.Namespace) -> Tuple[List[Sample], Dict[str, int]]:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "dropped_missing_image": 0,
        "dropped_empty_caption": 0,
        "dropped_status": 0,
        "dropped_person_count": 0,
    }

    samples: List[Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {args.image_col, args.caption_col}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must include columns {required}. Found: {reader.fieldnames}"
            )

        for row in reader:
            stats["rows_total"] += 1

            image_path = Path((row.get(args.image_col) or "").strip())
            caption = (row.get(args.caption_col) or "").strip()
            status = (row.get(args.status_col) or "").strip() if args.status_col else ""
            person_count = safe_int(row.get(args.person_count_col, ""), fallback=-1)

            if not image_path.exists():
                stats["dropped_missing_image"] += 1
                continue

            if len(caption.replace(" ", "")) < args.min_caption_len:
                stats["dropped_empty_caption"] += 1
                continue

            if args.keep_status and status != args.keep_status:
                stats["dropped_status"] += 1
                continue

            if args.require_person_count >= 0 and person_count != args.require_person_count:
                stats["dropped_person_count"] += 1
                continue

            samples.append(
                Sample(
                    source_path=image_path,
                    caption=caption,
                    status=status,
                    person_count=person_count,
                )
            )
            stats["rows_kept"] += 1

    if not samples:
        raise ValueError("No valid samples remain after filtering.")

    return samples, stats


def ensure_dirs(output_root: Path, dataset_name: str) -> Tuple[Path, Path, Path, Path]:
    dataset_root = output_root / dataset_name
    train_dir = dataset_root / "train2017"
    val_dir = dataset_root / "val2017"
    ann_dir = dataset_root / "annotations"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    return dataset_root, train_dir, val_dir, ann_dir


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def build_coco_split(
    split_samples: List[Sample],
    split_dir: Path,
    split_name: str,
    link_mode: str,
    starting_image_id: int,
    starting_ann_id: int,
) -> Tuple[Dict, int, int, List[Dict[str, str]]]:
    images = []
    annotations = []
    manifest = []

    image_id = starting_image_id
    ann_id = starting_ann_id

    for idx, sample in enumerate(split_samples):
        suffix = sample.source_path.suffix.lower()
        file_name = f"{split_name}_{idx:08d}{suffix}"
        dst_path = split_dir / file_name
        materialize_image(sample.source_path, dst_path, link_mode)

        with Image.open(sample.source_path) as im:
            width, height = im.size

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "caption": sample.caption,
            }
        )

        manifest.append(
            {
                "split": split_name,
                "file_name": file_name,
                "source_path": str(sample.source_path),
                "caption": sample.caption,
            }
        )

        image_id += 1
        ann_id += 1

    coco = {
        "info": {
            "description": "Custom dataset converted from CSV for SoftREPA",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "type": "captions",
    }
    return coco, image_id, ann_id, manifest


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.train_ratio <= 1.0):
        raise ValueError("--train_ratio must be between 0 and 1")

    output_root = Path(args.output_root)
    dataset_root, train_dir, val_dir, ann_dir = ensure_dirs(output_root, args.dataset_name)

    samples, stats = read_samples(args)

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    train_size = int(len(samples) * args.train_ratio)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    # Ensure validation is not empty when enough samples exist.
    if len(samples) > 1 and len(val_samples) == 0:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1]

    train_coco, next_img_id, next_ann_id, train_manifest = build_coco_split(
        train_samples,
        train_dir,
        "train",
        args.link_mode,
        starting_image_id=1,
        starting_ann_id=1,
    )

    val_coco, _, _, val_manifest = build_coco_split(
        val_samples,
        val_dir,
        "val",
        args.link_mode,
        starting_image_id=next_img_id,
        starting_ann_id=next_ann_id,
    )

    train_json_path = ann_dir / "captions_train2017.json"
    val_json_path = ann_dir / "captions_val2017.json"
    manifest_path = dataset_root / "conversion_manifest.jsonl"

    with train_json_path.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with val_json_path.open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in train_manifest + val_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Conversion done ===")
    print(f"CSV: {args.csv_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output: {dataset_root}")
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Val images:   {len(val_coco['images'])}")
    print(f"Train json:   {train_json_path}")
    print(f"Val json:     {val_json_path}")
    print(f"Manifest:     {manifest_path}")
    print("--- Filtering stats ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
