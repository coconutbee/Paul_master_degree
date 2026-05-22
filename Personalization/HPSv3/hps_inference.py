from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from hpsv3 import HPSv3RewardInferencer


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _iter_image_files(folder: Path, recursive: bool) -> List[Path]:
  if not folder.exists():
    raise FileNotFoundError(f"Folder not found: {folder}")
  if not folder.is_dir():
    raise NotADirectoryError(f"Not a folder: {folder}")

  pattern = "**/*" if recursive else "*"
  files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
  files.sort(key=lambda p: p.as_posix())
  return files


def _prompt_from_filename(image_path: Path) -> str:
  # Use file stem as prompt and convert underscores to spaces.
  prompt = image_path.stem.replace("_", " ")
  # Normalize whitespace (e.g., multiple underscores -> multiple spaces).
  prompt = " ".join(prompt.split())
  return prompt


def _batched(seq: Sequence, batch_size: int) -> Iterable[Tuple[int, int]]:
  if batch_size <= 0:
    raise ValueError("batch_size must be > 0")
  for start in range(0, len(seq), batch_size):
    end = min(start + batch_size, len(seq))
    yield start, end


def _score_images(
  inferencer: HPSv3RewardInferencer,
  prompts: Sequence[str],
  image_paths: Sequence[str],
  batch_size: int,
) -> List[float]:
  if len(prompts) != len(image_paths):
    raise ValueError("prompts and image_paths must have the same length")

  scores: List[float] = []
  for start, end in _batched(prompts, batch_size):
    rewards = inferencer.reward(list(prompts[start:end]), image_paths=list(image_paths[start:end]))
    # Each reward is expected to be (mu, sigma) or similar; take mu as scalar.
    scores.extend([float(reward[0].item()) for reward in rewards])
  return scores


def main() -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Run HPSv3 scoring on all images in a folder. "
      "Prompt is derived from filename (stem) with '_' replaced by spaces."
    )
  )
  parser.add_argument(
    "folder",
    type=str,
    help="Folder containing images (filenames are prompts).",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on (e.g., cuda, cpu). Default: cuda",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for inference. Default: 16",
  )
  parser.add_argument(
    "--recursive",
    action="store_true",
    help="Recursively search for images under the folder.",
  )
  parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Output CSV path. Default: ./output/hps_scores_<folder>_<timestamp>.csv",
  )
  args = parser.parse_args()

  folder = Path(args.folder)
  image_files = _iter_image_files(folder, recursive=args.recursive)
  if not image_files:
    raise SystemExit(f"No images found in: {folder}")

  prompts = [_prompt_from_filename(p) for p in image_files]
  image_paths = [str(p) for p in image_files]

  inferencer = HPSv3RewardInferencer(device=args.device)
  scores = _score_images(inferencer, prompts, image_paths, batch_size=args.batch_size)

  avg = sum(scores) / len(scores) if scores else float("nan")

  output_dir = Path("output")
  output_dir.mkdir(parents=True, exist_ok=True)

  if args.out:
    out_path = Path(args.out)
  else:
    safe_folder = folder.name or "folder"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hps_scores_{safe_folder}_{ts}.csv"

  out_path.parent.mkdir(parents=True, exist_ok=True)

  with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=["image_name", "image_path", "prompt", "score"],
    )
    writer.writeheader()
    for p, prompt, score in zip(image_files, prompts, scores):
      writer.writerow(
        {
          "image_name": p.name,
          "image_path": str(p),
          "prompt": prompt,
          "score": score,
        }
      )
    writer.writerow({"image_name": "__AVERAGE__", "image_path": "", "prompt": "", "score": avg})

  print(f"Scored {len(scores)} images")
  print(f"Average score: {avg}")
  print(f"CSV saved to: {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())