import argparse
import csv
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor
from prompt import system_prompt

MODEL_ID = "google/gemma-4-E2B-it"
CACHE_DIR = "/media/ee303/4TB/Gemma/checkpoints"
DEFAULT_INPUT_CSV = "/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age_race_labeled.csv"
DEFAULT_OUTPUT_CSV = "/media/ee303/4TB/Gemma/laion_gender_age_race_long_captioned"
DEFAULT_IMAGE_COLUMN_CANDIDATES = ("person_image_path", "image_path")
DEFAULT_OUTPUT_COLUMN = "long_caption"
DEFAULT_INDEX = 19


def parse_args():
    parser = argparse.ArgumentParser(description="Generate long captions for images listed in a CSV.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to '<base>_prompt<index>.csv'.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_INDEX,
        help="System prompt index to use.",
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help="CSV column containing image paths. If omitted, tries person_image_path then image_path.",
    )
    parser.add_argument(
        "--output-column",
        default=DEFAULT_OUTPUT_COLUMN,
        help="Column name used to store the generated long caption.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Stable key column for resume. Defaults to the selected image column.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV by skipping completed rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate captions even if the output column already exists for a row.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging.")
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = f"{DEFAULT_OUTPUT_CSV}_prompt{args.index}.csv"
    return args


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    return processor, model


def detect_image_column(fieldnames, requested_column=None):
    if requested_column:
        if requested_column not in fieldnames:
            raise ValueError(f"Image column '{requested_column}' not found in CSV.")
        return requested_column

    for column in DEFAULT_IMAGE_COLUMN_CANDIDATES:
        if column in fieldnames:
            return column

    raise ValueError(
        f"Could not find an image column. Tried: {', '.join(DEFAULT_IMAGE_COLUMN_CANDIDATES)}"
    )


def build_user_prompt(row, system_prompt_text):
    hints = {
        "gender": row.get("mivolo_gender", "").strip(),
        "age": row.get("mivolo_age", "").strip(),
        "race": row.get("race", "").strip(),
        "short_caption": row.get("caption_rf", "").strip() or row.get("caption", "").strip(),
    }

    hint_lines = [
        f"- Gender hint: {hints['gender'] or 'unknown'}",
        f"- Age hint: {hints['age'] or 'unknown'}",
        f"- Race hint: {hints['race'] or 'unknown'}",
        f"- Existing short caption hint: {hints['short_caption'] or 'none'}",
    ]

    return (
        f"{system_prompt_text}\n\n"
        "Metadata hints:\n"
        f"{chr(10).join(hint_lines)}\n\n"
        "Generate a dense long caption for this image."
    )


def generate_caption(processor, model, image_path, prompt, max_new_tokens):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return " ".join(response.split())


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def load_completed_ids(output_csv, id_column, output_column):
    completed = set()
    if not os.path.exists(output_csv):
        return completed

    with open(output_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = (row.get(id_column) or "").strip()
            caption = (row.get(output_column) or "").strip()
            if row_id and caption:
                completed.add(row_id)
    return completed


def ensure_output_fieldnames(input_fieldnames, output_column):
    if output_column in input_fieldnames:
        return input_fieldnames
    return input_fieldnames + [output_column]


def should_process_row(row):
    person_count = (row.get("person_count") or "").strip()
    caption_rf = (row.get("caption_rf") or "").strip().lower()
    if caption_rf and "years old" in caption_rf:
        # Apply format rewrite based on age/gender/race
        row["caption_rf"] = transform_caption_prefix(
            row["caption_rf"], 
            row.get("race", ""), 
            row.get("gender", ""), 
            row.get("age", None)
        )
        caption_rf = row["caption_rf"].lower()

    return person_count == "1" and "facing forward" not in caption_rf


def main():
    args = parse_args()
    system_prompt_text = system_prompt(index=args.index).strip()
    rows, input_fieldnames = read_rows(args.input_csv)
    image_column = detect_image_column(input_fieldnames, args.image_column)
    id_column = args.id_column or image_column
    output_fieldnames = ensure_output_fieldnames(input_fieldnames, args.output_column)

    if id_column not in output_fieldnames:
        raise ValueError(f"ID column '{id_column}' not found in CSV.")

    completed_ids = set()
    file_exists = os.path.exists(args.output_csv)
    if args.resume and file_exists:
        completed_ids = load_completed_ids(args.output_csv, id_column, args.output_column)

    filtered_rows = [row for row in rows if should_process_row(row)]
    if args.limit is not None:
        filtered_rows = filtered_rows[: args.limit]

    processor, model = load_model()

    mode = "a" if args.resume and file_exists else "w"
    write_header = not (args.resume and file_exists)

    progress = tqdm(total=len(filtered_rows), desc="Generating long captions", initial=0)

    if args.resume and completed_ids:
        already_done = 0
        for row in filtered_rows:
            row_id = (row.get(id_column) or "").strip()
            if row_id in completed_ids and not args.overwrite:
                already_done += 1
        if already_done:
            progress.update(already_done)

    with open(args.output_csv, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        if write_header:
            writer.writeheader()

        for row in filtered_rows:
            row_id = (row.get(id_column) or "").strip()
            image_path = (row.get(image_column) or "").strip()

            if args.resume and not args.overwrite and row_id in completed_ids:
                continue

            output_row = dict(row)
            try:
                if not image_path:
                    raise ValueError(f"Missing image path in column '{image_column}'.")
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                prompt = build_user_prompt(row, system_prompt_text)
                output_row[args.output_column] = generate_caption(
                    processor=processor,
                    model=model,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                output_row[args.output_column] = f"[ERROR] {exc}"

            writer.writerow(output_row)
            f.flush()
            progress.update(1)

    progress.close()
    print(f"Saved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
