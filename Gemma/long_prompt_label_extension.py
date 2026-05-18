import argparse
import csv
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
import transformers

MODEL_ID = "google/gemma-4-E2B-it"
CACHE_DIR = "/media/ee303/4TB/Gemma/checkpoints"
DEFAULT_INPUT_CSV = "/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age_race_labeled.csv"
DEFAULT_OUTPUT_CSV = "/media/ee303/4TB/Gemma/laion_gender_age_race_long_captioned_extension"
DEFAULT_IMAGE_COLUMN_CANDIDATES = ("person_image_path", "image_path")
DEFAULT_INPUT_COLUMN = "caption_rf"
DEFAULT_OUTPUT_COLUMN = "long_caption"
DEFAULT_SENTENCE_COUNT = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use caption_rf as Gemma input and save the generated continuation to long_caption."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to '<base>.csv'.",
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help="CSV column containing image paths. If omitted, tries person_image_path then image_path.",
    )
    parser.add_argument(
        "--input-column",
        default=DEFAULT_INPUT_COLUMN,
        help="CSV column containing the seed caption to continue.",
    )
    parser.add_argument(
        "--output-column",
        default=DEFAULT_OUTPUT_COLUMN,
        help="Column name used to store Gemma output. Defaults to long_caption.",
    )
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=DEFAULT_SENTENCE_COUNT,
        help="Number of new scenario sentences to add.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Stable key column for resume. Defaults to the selected image column.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length.")
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
        args.output_csv = f"{DEFAULT_OUTPUT_CSV}.csv"
    return args


def get_model_class():
    for class_name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
    raise ImportError(
        "No supported multimodal model auto class found in transformers. "
        "Please upgrade transformers."
    )


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model_class = get_model_class()
    model = model_class.from_pretrained(
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


def build_user_prompt(seed_caption, sentence_count):
    return (
        "Continue the provided image caption by adding scenario context grounded in the image.\n"
        f"Seed caption: {seed_caption}\n\n"
        f"Write exactly {sentence_count} additional English sentences.\n"
        "Describe the visible scenario, environment, activity, objects, and spatial context.\n"
        "Do not rewrite or repeat the seed caption.\n"
        "Do not add demographics, age, race, gender, or pose labels unless they are already in the seed caption.\n"
        "Do not use bullet points, numbering, quotation marks, or line breaks.\n"
        "Return only the additional sentences."
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


def split_sentences(text):
    sentences = []
    start = 0
    for idx, char in enumerate(text):
        if char in ".!?":
            sentence = text[start : idx + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = idx + 1
    trailing = text[start:].strip()
    if trailing:
        sentences.append(trailing.rstrip(".!?") + ".")
    return sentences


def clean_continuation(seed_caption, generated_text, sentence_count):
    continuation = generated_text.strip().strip('"').strip("'")
    if continuation.lower().startswith(seed_caption.lower()):
        continuation = continuation[len(seed_caption) :].strip()
    sentences = split_sentences(continuation)
    return " ".join(sentences[:sentence_count]).strip()


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


def should_process_row(row, input_column):
    return bool((row.get(input_column) or "").strip())


def main():
    args = parse_args()
    rows, input_fieldnames = read_rows(args.input_csv)
    image_column = detect_image_column(input_fieldnames, args.image_column)
    id_column = args.id_column or image_column
    output_fieldnames = ensure_output_fieldnames(input_fieldnames, args.output_column)

    if args.input_column not in input_fieldnames:
        raise ValueError(f"Input caption column '{args.input_column}' not found in CSV.")
    if id_column not in output_fieldnames:
        raise ValueError(f"ID column '{id_column}' not found in CSV.")

    completed_ids = set()
    file_exists = os.path.exists(args.output_csv)
    if args.resume and file_exists:
        completed_ids = load_completed_ids(args.output_csv, id_column, args.output_column)

    filtered_rows = [row for row in rows if should_process_row(row, args.input_column)]
    if args.limit is not None:
        filtered_rows = filtered_rows[: args.limit]

    processor, model = load_model()

    mode = "a" if args.resume and file_exists else "w"
    write_header = not (args.resume and file_exists)

    progress = tqdm(total=len(filtered_rows), desc="Generating scenario captions", initial=0)

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

                seed_caption = (row.get(args.input_column) or "").strip()
                prompt = build_user_prompt(seed_caption, args.sentence_count)
                generated_text = generate_caption(
                    processor=processor,
                    model=model,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                continuation = clean_continuation(
                    seed_caption=seed_caption,
                    generated_text=generated_text,
                    sentence_count=args.sentence_count,
                )
                output_row[args.output_column] = continuation
            except Exception as exc:
                output_row[args.output_column] = f"[ERROR] {exc}"

            writer.writerow(output_row)
            f.flush()
            progress.update(1)

    progress.close()
    print(f"Saved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
