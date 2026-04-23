#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DeepFashion checklist captions with pose prompts using a "
            "Qwen vision-language model."
        )
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("DeepFashion_Training_Final"),
        help="Root directory of DeepFashion images.",
    )
    parser.add_argument(
        "--sam-csv",
        type=Path,
        default=Path("sam3_labeled.csv"),
        help="CSV file containing image_path and pose caption columns.",
    )
    parser.add_argument(
        "--task-template",
        type=Path,
        default=Path("Task_with_pose.md"),
        help="Prompt template in Task.md checklist format.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("deepfashion_qwen35_task_with_pose.csv"),
        help="CSV file for generated results.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("deepfashion_qwen35_task_with_pose.jsonl"),
        help="JSONL file for generated results.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL,
        help="Hugging Face model id, for example Qwen/Qwen3.5-4B.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images per generation batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=420,
        help="Maximum number of generated tokens per sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for smoke tests.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional row offset before processing.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows that already exist in the output CSV.",
    )
    parser.add_argument(
        "--description-mode",
        choices=["freeform", "confident_only", "confident_fixed"],
        default="confident_fixed",
        help=(
            "How conservative the identity description should be. "
            "'confident_fixed' uses controlled vocabulary and omits low-confidence traits."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to model and processor loading.",
    )
    return parser.parse_args()


def detect_gender(image_path: str) -> str:
    name = Path(image_path).name
    if "WOMEN-" in name:
        return "female"
    if "MEN-" in name:
        return "male"
    return "person"


def normalize_gender_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"female", "woman", "women", "f"}:
        return "female"
    if text in {"male", "man", "men", "m"}:
        return "male"
    if text in {"person", "neutral", "unknown", "unsure", ""}:
        return "person"
    return None


def infer_gender_from_caption(caption: str | None) -> str | None:
    if caption is None:
        return None
    text = str(caption).strip().lower()
    if text.startswith("a woman") or text.startswith("an woman"):
        return "female"
    if text.startswith("a man") or text.startswith("an man"):
        return "male"
    if text.startswith("a person") or text.startswith("an person"):
        return "person"
    return None


def resolve_image_path(image_root: Path, raw_image_path: str) -> Path:
    image_path = Path(raw_image_path)
    if image_path.exists():
        return image_path
    if not image_path.is_absolute():
        candidate = image_root / image_path
        if candidate.exists():
            return candidate
    parts = image_path.parts
    if "DeepFashion_Training_Final" in parts:
        idx = parts.index("DeepFashion_Training_Final")
        candidate = image_root / Path(*parts[idx + 1 :])
        if candidate.exists():
            return candidate
        candidate = image_root.parent / "DeepFashion_Training_Final" / Path(*parts[idx + 1 :])
        if candidate.exists():
            return candidate
    return image_path


def resolve_dtype(name: str, device: str) -> torch.dtype | str:
    if name == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device == "cuda":
            return torch.float16
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_template(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8").strip()


def normalize_pose_prompt(pose_prompt: str, gender: str) -> str:
    text = pose_prompt.strip().rstrip(".")
    text = re.sub(r"^(?:A|An)\s+(?:man|woman|person)\s+", "", text, flags=re.IGNORECASE)
    if gender == "female":
        text = re.sub(r"\bhis\b", "her", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "her", text, flags=re.IGNORECASE)
    elif gender == "male":
        text = re.sub(r"\bhis\b", "his", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "him", text, flags=re.IGNORECASE)
    else:
        text = re.sub(r"\bhis\b", "their", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "them", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    if not text:
        return "Pose unclear"
    return text[:1].upper() + text[1:]


def build_prompt(
    task_template: str,
    gender: str,
    pose_prompt: str,
    description_mode: str,
) -> str:
    rendered = task_template.replace("{gender}", gender)
    extra = (
        "\n\nADDITIONAL POSE CONSTRAINTS:\n"
        f"- Exact pose text to preserve: {pose_prompt}\n"
        "- You must preserve the pose wording exactly as given above.\n"
        "- Do not paraphrase, shorten, reorder, or reinterpret the pose text.\n"
        "- Integrate the pose naturally into the final prompt, but keep the pose phrase itself verbatim.\n"
        "- Write a long, visually rich, generation-ready description.\n"
        "- Avoid checklist formatting, bullet points, or numbered lists in the final answer.\n"
        "- If a detail is unclear, explicitly say it is unclear rather than hallucinating.\n"
        "- Return exactly one final answer wrapped in <final_prompt> and </final_prompt>.\n"
        "- Do not output analysis, planning, or any text outside the <final_prompt> block.\n"
    )
    if description_mode == "confident_only":
        extra += (
            "\nCONFIDENCE RULES:\n"
            "- Only describe identity attributes that are clearly visible in the image.\n"
            "- If an attribute is uncertain, omit it completely instead of guessing.\n"
            "- Do not mention ethnicity.\n"
            "- Do not mention eye color unless it is unmistakably visible.\n"
            "- Do not use hedging phrases such as likely, maybe, appears, seems, or probably.\n"
            "- Prefer stable, broad wording over fragile, overly specific wording.\n"
        )
    elif description_mode == "confident_fixed":
        extra += (
            "\nCONFIDENCE AND CONTROLLED-VOCAB RULES:\n"
            "- Only describe identity attributes that are clearly visible in the image.\n"
            "- If an attribute is uncertain, omit it completely instead of guessing.\n"
            "- Do not mention ethnicity.\n"
            "- Do not mention eye color unless it is unmistakably visible.\n"
            "- Do not use hedging phrases such as likely, maybe, appears, seems, or probably.\n"
            "- Use only these complexion buckets if needed: fair complexion, light complexion, medium complexion, medium-dark complexion, dark complexion.\n"
            "- Use only these hair color buckets if needed: black hair, dark brown hair, brown hair, light brown hair, blonde hair.\n"
            "- Use only these facial hair buckets if needed: clean-shaven, light stubble, beard, mustache.\n"
            "- Prefer broad stable wording over nuanced undertones such as olive, warm, cool, sun-kissed, or rosy.\n"
            "- Keep identity cues concise and stable across similar images.\n"
        )
    return rendered + extra


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.sam_csv)
    df = df.iloc[args.start_index :].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()
    df["image_path"] = df["image_path"].astype(str)
    df["caption"] = df["caption"].fillna("Unclear pose")
    df["raw_caption"] = df["caption"]
    gender_columns = [
        col
        for col in ["gender", "subject_gender", "person_gender"]
        if col in df.columns
    ]
    caption_gender = df["raw_caption"].map(infer_gender_from_caption)
    if gender_columns:
        gender_col = gender_columns[0]
        df["gender"] = df[gender_col].map(normalize_gender_label)
        df["gender"] = df["gender"].fillna(caption_gender)
        df["gender"] = df["gender"].fillna(df["image_path"].map(detect_gender))
    else:
        df["gender"] = caption_gender.fillna(df["image_path"].map(detect_gender))
    df["caption"] = [
        normalize_pose_prompt(caption, gender)
        for caption, gender in zip(df["raw_caption"], df["gender"])
    ]
    return df


def load_processed_paths(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    processed: set[str] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = row.get("image_path")
            if image_path:
                processed.add(image_path)
    return processed


def batched_rows(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(rows), batch_size):
        yield rows[idx : idx + batch_size]


def apply_chat_template(processor, image_path: str, prompt_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a caption writer for image generation. "
                "Return exactly one polished prompt wrapped in <final_prompt> and "
                "</final_prompt>. Do not reveal analysis or intermediate reasoning."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt_text


def ensure_pose_prompt(generated_text: str, pose_prompt: str) -> str:
    stripped = generated_text.strip()
    if pose_prompt in stripped:
        return stripped
    if stripped:
        return f"{stripped} The pose and body orientation should be: {pose_prompt}."
    return f"The pose and body orientation should be: {pose_prompt}."


def stabilize_identity_language(
    text: str,
    description_mode: str,
    gender: str,
) -> str:
    if description_mode == "freeform":
        return text

    substitutions = [
        (r"\b(?:young\s+)?caucasian\s+(man|male)\b", r"young \1"),
        (r"\b(?:young\s+)?caucasian\s+(woman|female)\b", r"young \1"),
        (r"\b(?:young\s+)?asian\s+(man|male)\b", r"young \1"),
        (r"\b(?:young\s+)?asian\s+(woman|female)\b", r"young \1"),
        (r"\b(?:young\s+)?african(?:-american)?\s+(man|male)\b", r"young \1"),
        (r"\b(?:young\s+)?african(?:-american)?\s+(woman|female)\b", r"young \1"),
        (r"\bplatinum blonde hair\b", "blonde hair"),
        (r"\bdirty blonde hair\b", "blonde hair"),
        (r"\bjet-black hair\b", "black hair"),
        (r"\bjet black hair\b", "black hair"),
        (r"\bdeep brown eyes\b", "brown eyes"),
        (r"\blarge dark eyes\b", "dark eyes"),
        (r"\blight olive complexion\b", "light complexion"),
        (r"\bwarm olive complexion\b", "medium complexion"),
        (r"\bolive complexion\b", "medium complexion"),
        (r"\blight olive skin tone\b", "light complexion"),
        (r"\bwarm, medium-dark complexion\b", "medium-dark complexion"),
        (r"\bwarm, olive-toned skin tone\b", "medium complexion"),
        (r"\bwarm, olive-toned complexion\b", "medium complexion"),
        (r"\bwarm skin tone\b", "light complexion"),
        (r"\bsun-kissed complexion\b", "medium complexion"),
        (r"\bsun-kissed tone\b", "medium complexion"),
        (r"\bsoft, natural beard\b", "beard"),
        (r"\bhint of stubble\b", "light stubble"),
        (r"\blight stubble beard\b", "light stubble"),
        (r"\bneatly trimmed goatee\b", "beard"),
    ]
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    if description_mode == "confident_only":
        text = re.sub(r"\blight-colored eyes\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdark almond-shaped eyes\b", "dark eyes", text, flags=re.IGNORECASE)
    if description_mode == "confident_fixed":
        text = re.sub(r"\blight-colored eyes\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bblue eyes\b", "blue eyes", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdark almond-shaped eyes\b", "dark eyes", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwell-groomed beard and mustache\b", "beard and mustache", text, flags=re.IGNORECASE)

    if gender in {"female", "person"}:
        text = re.sub(
            r"\bShe has\s+a\s+clean-shaven face with\b",
            "She has",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bHe has\s+a\s+clean-shaven face with\b",
            "He has",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bThey have\s+a\s+clean-shaven face with\b",
            "They have",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bHer face is clean-shaven with\b",
            "She has",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bHis face is clean-shaven with\b",
            "He has",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bTheir face is clean-shaven with\b",
            "They have",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bfeaturing\s+a\s+clean-shaven face and\s+a\b",
            "featuring a",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bfeaturing\s+a\s+clean-shaven face and\b",
            "featuring",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bfeaturing\s+a\s+clean-shaven face with\b",
            "featuring",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:she|the subject) is clean-shaven with no facial hair visible\.?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\bclean-shaven face\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bclean-shaven jawline\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bclean-shaven\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\blight stubble\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bbeard and mustache\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bbeard\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmustache\b", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"\((?:[^)]*female subject[^)]*|[^)]*male subject[^)]*|[^)]*requested in the task instructions[^)]*)\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bfeaturing\s+a\s+and\s+a\b", "featuring a", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfeaturing\s+a\s+and\b", "featuring", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith\s+a\s+and\s+a\b", "with a", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith\s+a\s+and\b", "with", text, flags=re.IGNORECASE)
    text = re.sub(r"\bThe fabric smooth with\b", "The fabric is smooth with", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bThe fabric of the ([a-z ]+?) smooth with\b",
        r"The fabric of the \1 is smooth with",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([A-Za-z]+) faces forward and looking straight\b",
        r"\1 faces forward and looks straight",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([A-Za-z]+) is facing forward and looking straight\b",
        r"\1 is facing forward and looking straight",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([A-Za-z]+) faces forward and tilted down\b",
        r"\1 faces forward with their head tilted down",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bHer face is with\b",
        "She has",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bHis face is with\b",
        "He has",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bTheir face is with\b",
        "They have",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bShe has\s+a\s+with\b",
        "She has",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bHe has\s+a\s+with\b",
        "He has",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bThey have\s+a\s+with\b",
        "They have",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bShe has\s+with\b",
        "She has",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bHe has\s+with\b",
        "He has",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bThey have\s+with\b",
        "They have",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bwith pocket on the left chest\b",
        r"with a pocket on the left chest",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bwith center part\b",
        r"with a center part",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:likely|maybe|seems|probably)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r",\s*,", ", ", text)
    text = re.sub(r"\bfeaturing\s*,", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith\s*,", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\bwith a,?\s*", "with ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_final_prompt(generated_text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        generated_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = re.findall(
        r"<final_prompt>(.*?)</final_prompt>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match_text in reversed(matches):
        candidate = re.sub(r"\s+", " ", match_text).strip()
        if len(candidate) >= 40 and "`" not in candidate:
            return candidate

    match = re.search(
        r"(?:final prompt|prompt)\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def clean_generated_text(
    generated_text: str,
    description_mode: str,
    gender: str,
) -> str:
    text = extract_final_prompt(generated_text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("the user wants"):
            continue
        if line.lower().startswith("i need to follow"):
            continue
        line = re.sub(r"^\d+\.\s*analyze the image:\s*", "", line, flags=re.IGNORECASE)
        line = line.replace("**", "")
        line = re.sub(r"^\*\s*", "", line)
        line = re.sub(r"^-\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\bAnalyze the Image:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\bDraft the Prompt - Section by Section:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI need to follow specific instructions:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return stabilize_identity_language(text, description_mode, gender)


def prepare_output_files(output_csv: Path, output_jsonl: Path) -> None:
    fieldnames = [
        "image_path",
        "gender",
        "pose_prompt",
        "prompt_text",
        "generated_text",
        "final_text",
        "status",
        "error",
    ]
    if not output_csv.exists():
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
    if not output_jsonl.exists():
        output_jsonl.write_text("", encoding="utf-8")


def append_result(output_csv: Path, output_jsonl: Path, row: dict) -> None:
    fieldnames = [
        "image_path",
        "gender",
        "pose_prompt",
        "prompt_text",
        "generated_text",
        "final_text",
        "status",
        "error",
    ]
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)
    with output_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    image_root = args.image_root.resolve()
    sam_csv = args.sam_csv.resolve()
    task_template_path = args.task_template.resolve()
    output_csv = args.output_csv.resolve()
    output_jsonl = args.output_jsonl.resolve()

    task_template = load_template(task_template_path)
    dataset = load_dataset(args)
    prepare_output_files(output_csv, output_jsonl)

    processed_paths = (
        load_processed_paths(output_csv) if args.skip_existing else set()
    )
    if processed_paths:
        dataset = dataset[~dataset["image_path"].isin(processed_paths)].copy()

    rows = dataset.to_dict(orient="records")
    if not rows:
        print("No rows to process.")
        return

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch_dtype = resolve_dtype(args.dtype, device)

    print(f"Loading model: {args.model_id}")
    print(f"Device: {device}")
    print(f"DType: {torch_dtype}")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    for batch in batched_rows(rows, args.batch_size):
        valid_batch: list[dict] = []
        texts: list[str] = []
        images: list[Image.Image] = []

        for row in batch:
            image_path = resolve_image_path(image_root, row["image_path"])

            prompt_text = build_prompt(
                task_template=task_template,
                gender=row["gender"],
                pose_prompt=row["caption"],
                description_mode=args.description_mode,
            )

            if not image_path.exists():
                append_result(
                    output_csv,
                    output_jsonl,
                    {
                        "image_path": str(image_path),
                        "gender": row["gender"],
                        "pose_prompt": row["caption"],
                        "prompt_text": prompt_text,
                        "generated_text": "",
                        "final_text": "",
                        "status": "missing_image",
                        "error": "Image file not found.",
                    },
                )
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                text = apply_chat_template(processor, str(image_path), prompt_text)
            except Exception as exc:  # pragma: no cover - runtime protection
                append_result(
                    output_csv,
                    output_jsonl,
                    {
                        "image_path": str(image_path),
                        "gender": row["gender"],
                        "pose_prompt": row["caption"],
                        "prompt_text": prompt_text,
                        "generated_text": "",
                        "final_text": "",
                        "status": "image_load_error",
                        "error": str(exc),
                    },
                )
                continue

            row = dict(row)
            row["resolved_image_path"] = str(image_path)
            row["prompt_text"] = prompt_text
            valid_batch.append(row)
            texts.append(text)
            images.append(image)

        if not valid_batch:
            continue

        try:
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                )
        except Exception as exc:  # pragma: no cover - runtime protection
            for row in valid_batch:
                append_result(
                    output_csv,
                    output_jsonl,
                    {
                        "image_path": row["resolved_image_path"],
                        "gender": row["gender"],
                        "pose_prompt": row["caption"],
                        "prompt_text": row["prompt_text"],
                        "generated_text": "",
                        "final_text": "",
                        "status": "generation_error",
                        "error": str(exc),
                    },
                )
            continue

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for batch_idx, row in enumerate(valid_batch):
            trimmed_ids = generated_ids[batch_idx][int(input_lengths[batch_idx]) :]
            generated_text = processor.decode(
                trimmed_ids,
                skip_special_tokens=True,
            ).strip()
            cleaned_text = clean_generated_text(
                generated_text,
                description_mode=args.description_mode,
                gender=row["gender"],
            )
            final_text = ensure_pose_prompt(cleaned_text, row["caption"])
            append_result(
                output_csv,
                output_jsonl,
                {
                    "image_path": row["resolved_image_path"],
                    "gender": row["gender"],
                    "pose_prompt": row["caption"],
                    "prompt_text": row["prompt_text"],
                    "generated_text": generated_text,
                    "final_text": final_text,
                    "status": "ok",
                    "error": "",
                },
            )
            print(f"Processed: {row['resolved_image_path']}")


if __name__ == "__main__":
    main()