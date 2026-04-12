#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import statistics


def run_cmd(cmd, dry_run=False):
    printable = " ".join(cmd)
    print(f"\n[RUN] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_fga(raw_score):
    if raw_score is None:
        return 0.0
    val = float(raw_score)
    return min(max((val - 1.0) / 4.0, 0.0), 1.0)


def _safe_float(raw):
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def append_metrics_summary_jsonl(json_path, summary_jsonl_path, exp_name, mode):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metric_keys = [
        "t2i_pose_match",
        "swap_pose_match",
        "expression_correct_t2i",
        "expression_correct",
        "t2i_scenario_score",
        "swap_scenario_score",
        "t2i_clip_t2i",
        "swap_clip_t2i",
        "t2i_clip_id_i2i",
        "swap_clip_id_i2i",
        "t2i_dino_id_i2i",
        "swap_dino_id_i2i",
        "t2i_id_similarity",
        "swap_id_similarity",
        "t2i_final_score",
        "swap_final_score",
    ]

    metrics = {}
    for key in metric_keys:
        values = []
        for item in data:
            raw = item.get(key)
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue

        if values:
            metrics[key] = {
                "mean": round(statistics.fmean(values), 6),
                "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
                "count": len(values),
            }

    fga_t2i_vals = []
    fga_swap_vals = []
    for item in data:
        t2i_raw = _safe_float(item.get("fga_orig_score"))
        if t2i_raw is not None:
            fga_t2i_vals.append(normalize_fga(t2i_raw))

        swap_raw = _safe_float(item.get("fga_swap_score"))
        if swap_raw is not None:
            fga_swap_vals.append(normalize_fga(swap_raw))

    if fga_t2i_vals:
        metrics["t2i_evalmuse_norm"] = {
            "mean": round(_mean(fga_t2i_vals), 6),
            "std": round(_std(fga_t2i_vals), 6),
            "count": len(fga_t2i_vals),
        }
    if fga_swap_vals:
        metrics["swap_evalmuse_norm"] = {
            "mean": round(_mean(fga_swap_vals), 6),
            "std": round(_std(fga_swap_vals), 6),
            "count": len(fga_swap_vals),
        }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": exp_name,
        "mode": mode,
        "json": os.path.abspath(json_path),
        "metrics": metrics,
    }

    summary_dir = os.path.dirname(summary_jsonl_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def export_named_metrics_csv(json_path, csv_path, exp_name):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def collect_mean(key, transform=None):
        vals = []
        for item in data:
            v = _safe_float(item.get(key))
            if v is None:
                continue
            vals.append(transform(v) if transform else v)
        return _mean(vals) if vals else None

    row = {
        "name": exp_name,
        "t2i_clip_t2i": collect_mean("t2i_clip_t2i"),
        "swap_clip_t2i": collect_mean("swap_clip_t2i"),
        "t2i_clip_i2i": collect_mean("t2i_clip_id_i2i"),
        "swap_clip_i2i": collect_mean("swap_clip_id_i2i"),
        "t2i_dino": collect_mean("t2i_dino_id_i2i"),
        "swap_dino": collect_mean("swap_dino_id_i2i"),
        "t2i_evalmuse": collect_mean("fga_orig_score", transform=normalize_fga),
        "swap_evalmuse": collect_mean("fga_swap_score", transform=normalize_fga),
        "t2i_sa": collect_mean("t2i_scenario_score"),
        "swap_sa": collect_mean("swap_scenario_score"),
    }

    fields = list(row.keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation pipeline: required folder, optional ref/swap"
    )
    parser.add_argument("--folder", required=True, help="T2I image folder path")
    parser.add_argument("--ref", default="", help="Optional reference ID image folder path")
    parser.add_argument("--swap", default="", help="Optional swapped image folder path")
    parser.add_argument("--name", default="", help="Experiment name (default: folder basename)")
    parser.add_argument("--output", default="", help="Metadata json output path")
    parser.add_argument("--summary-jsonl", default="metrics_summary.jsonl", help="Append metric mean/std to this JSONL file")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--pose-python", default="", help="Python executable for pose stage only")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    ref = os.path.abspath(args.ref) if args.ref else ""
    swap = os.path.abspath(args.swap) if args.swap else ""

    ensure_exists(folder, "folder")
    if ref:
        ensure_exists(ref, "ref")
    if swap:
        ensure_exists(swap, "swap")

    exp_name = args.name.strip() or Path(folder).name
    json_output = args.output.strip() or f"{exp_name}_metadata.json"
    json_output = os.path.abspath(json_output)
    pose_python = args.pose_python.strip() or args.python

    mode = "full" if swap else "t2i"

    print("=" * 88)
    print("Unified Personalization Evaluation")
    print(f"name   : {exp_name}")
    print(f"folder : {folder}")
    print(f"ref    : {ref if ref else '(not provided)'}")
    print(f"swap   : {swap if swap else '(not provided)'}")
    print(f"python : {args.python}")
    print(f"posepy : {pose_python}")
    print(f"mode   : {mode}")
    print(f"json   : {json_output}")
    print("=" * 88)

    # 1) Build metadata
    # In full mode, IDs should come from swapped filenames (usually {id}_{prompt}).
    # In t2i-only mode, keep using the T2I folder as metadata source.
    metadata_source = swap if swap else folder
    run_cmd([
        args.python,
        "json_maker_a.py",
        "--folder", metadata_source,
        "--output", json_output,
    ], dry_run=args.dry_run)

    # 2) ID similarity (requires ref)
    if ref:
        id_cmd = [
            args.python,
            "AdaFace/inference_v2_a.py",
            "--json", json_output,
            "--name", exp_name,
            "--t2i", folder,
            "--ref", ref,
            "--mode", mode,
        ]
        if swap:
            id_cmd.extend(["--swap", swap])
        run_cmd(id_cmd, dry_run=args.dry_run)
    else:
        print("\n[SKIP] AdaFace/inference_v2_a.py (no --ref provided)")

    # 3) Generate GT labels
    run_cmd([
        args.python,
        "gt_maker.py",
        "--input", json_output,
        "--output", json_output,
    ], dry_run=args.dry_run)

    # 4) Pose evaluation (T2I always, Swap in full mode)
    pose_cmd = [
        pose_python,
        "pose/eval_pose_v2_a.py",
        "--t2i", folder,
        "--name", exp_name,
        "--json", json_output,
        "--mode", mode,
    ]
    if ref:
        pose_cmd.extend(["--ref", ref])
    if swap:
        pose_cmd.extend(["--swap", swap])
    run_cmd(pose_cmd, dry_run=args.dry_run)

    # Shared args for scripts that support mode/swap
    mode_base = [
        "--name", exp_name,
        "--json", json_output,
        "--t2i", folder,
        "--mode", mode,
    ]
    if ref:
        mode_base.extend(["--ref", ref])
    if swap:
        mode_base.extend(["--swap", swap])

    # 5) Expression + scenario
    run_cmd([
        args.python,
        "exps_scenario_v2_a.py",
        *mode_base,
    ], dry_run=args.dry_run)

    # 6) CLIP + DINO
    # Without ref, script still computes CLIP-T2I (and swap prompt alignment where available).
    run_cmd([
        args.python,
        "general_scoring_v2_a.py",
        *mode_base,
    ], dry_run=args.dry_run)

    # 7) Weighted final score
    # scoring_v2_a can still run without ref; missing fields default to 0.
    run_cmd([
        args.python,
        "scoring_v2_a.py",
        *mode_base,
    ], dry_run=args.dry_run)

    # 8) EvalMuse
    evalmuse_cmd = [
        args.python,
        "EvalMuse/eval_v2_a.py",
        "--json", json_output,
        "--name", exp_name,
        "--t2i", folder,
        "--mode", mode,
    ]
    if swap:
        evalmuse_cmd.extend(["--swap", swap])

    run_cmd(evalmuse_cmd, dry_run=args.dry_run)

    # 9) Append mean/std metrics summary
    if not args.dry_run:
        summary_jsonl = os.path.abspath(args.summary_jsonl)
        append_metrics_summary_jsonl(json_output, summary_jsonl, exp_name, mode)
        print(f"Metrics summary jsonl: {summary_jsonl}")

        named_csv = os.path.abspath(f"{exp_name}.csv")
        export_named_metrics_csv(json_output, named_csv, exp_name)
        print(f"Named metrics csv: {named_csv}")

    print("\nDone.")
    print(f"Final json: {json_output}")


if __name__ == "__main__":
    main()
