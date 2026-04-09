import argparse
import csv
import json
import re
from pathlib import Path

import torch


def read_prompts(input_path: Path, prompt_key: str) -> list[str]:
    prompts: list[str] = []
    suffix = input_path.suffix.lower()
    prompt_key_norm = prompt_key.strip().lower()

    if suffix == ".csv":
        # utf-8-sig strips BOM so "\ufeffprompt" won't break header matching.
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            field_map = {name.strip().lower(): name for name in fieldnames if name is not None}

            selected_key = field_map.get(prompt_key_norm)
            if selected_key is None:
                for candidate in ("prompt", "caption", "text", "sentence"):
                    if candidate in field_map:
                        selected_key = field_map[candidate]
                        print(f"[WARN] prompt key '{prompt_key}' not found, fallback to '{selected_key}'")
                        break

            if selected_key is None:
                raise KeyError(f"CSV missing prompt column: {prompt_key}. Available: {fieldnames}")

            for row in reader:
                prompt = (row.get(selected_key) or "").strip()
                if prompt:
                    prompts.append(prompt)
    elif suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e
                selected_key = None
                for key in obj.keys():
                    if str(key).strip().lower() == prompt_key_norm:
                        selected_key = key
                        break
                if selected_key is None:
                    for candidate in ("prompt", "caption", "text", "sentence"):
                        for key in obj.keys():
                            if str(key).strip().lower() == candidate:
                                selected_key = key
                                break
                        if selected_key is not None:
                            break
                prompt = str(obj.get(selected_key, "") if selected_key is not None else "").strip()
                if prompt:
                    prompts.append(prompt)
    else:
        raise ValueError("Input file must be .csv or .jsonl")

    if not prompts:
        raise ValueError("No valid prompts found in input file")
    return prompts


def prompt_to_filename(prompt: str, max_len: int = 120) -> str:
    # Keep filenames portable: replace unsafe chars, collapse spaces, and cap length.
    name = prompt.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        name = "prompt"
    return name[:max_len].rstrip("._-") or "prompt"


def main() -> None:
    from diffusers import Lumina2Pipeline

    parser = argparse.ArgumentParser(description="Generate images with Lumina from CSV/JSONL prompts")
    parser.add_argument("--input", type=str, required=True, help="Path to .csv or .jsonl prompt file")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Prompt field name")
    parser.add_argument("--output_dir", type=str, default="generated/lumina", help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic generation")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_trunc_ratio", type=float, default=0.25)
    parser.add_argument("--cfg_normalization", action="store_true", default=True)
    parser.add_argument("--no_cfg_normalization", action="store_false", dest="cfg_normalization")
    parser.add_argument("--model_id", type=str, default="Alpha-VLLM/Lumina-Image-2.0")
    parser.add_argument("--disable_cpu_offload", action="store_true", help="Disable CPU offload")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(input_path, args.prompt_key)
    print(f"[INFO] Loaded {len(prompts)} prompts from {input_path}")

    pipe = Lumina2Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    if not args.disable_cpu_offload:
        pipe.enable_model_cpu_offload()

    used_names: dict[str, int] = {}

    for i, prompt in enumerate(prompts):
        generator = torch.Generator("cpu").manual_seed(args.seed + i)
        image = pipe(
            prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            cfg_trunc_ratio=args.cfg_trunc_ratio,
            cfg_normalization=args.cfg_normalization,
            generator=generator,
        ).images[0]

        base_name = prompt_to_filename(prompt)
        suffix_count = used_names.get(base_name, 0)
        used_names[base_name] = suffix_count + 1
        file_name = f"{base_name}.png" if suffix_count == 0 else f"{base_name}_{suffix_count}.png"
        output_path = output_dir / file_name
        image.save(output_path)
        print(f"[OK] {output_path}")


if __name__ == "__main__":
    main()
