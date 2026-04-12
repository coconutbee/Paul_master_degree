#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/ee303/anaconda3/envs/pslz/bin/python"
POSE_PYTHON_BIN="/home/ee303/anaconda3/envs/sam3d/bin/python"
SUMMARY_JSONL="summary/metrics_summary.jsonl"

mkdir -p summary

run_one() {
  local folder="$1"
  local name="$2"

  bash run_unified_v3.sh \
    --folder "$folder" \
    --name "$name" \
    --output "summary/${name}_metadata.json" \
    --summary-jsonl "$SUMMARY_JSONL" \
    --python "$PYTHON_BIN" \
    --pose-python "$POSE_PYTHON_BIN"
}

# SoftREPA fine-tuning
run_one "/media/ee303/4TB/SoftREPA/generated/PP_with_dc_t" "SoftREPA_FT_PP_noref"
run_one "/media/ee303/4TB/SoftREPA/generated/GP_with_dc_t" "SoftREPA_FT_GP_noref"

# SD3 vanilla
run_one "/media/ee303/4TB/SoftREPA/generated/GP_vanilla" "SD3_GP_noref"
run_one "/media/ee303/4TB/SoftREPA/generated/PP_vanilla" "SD3_PP_noref"

# SoftREPA
run_one "/media/ee303/4TB/SoftREPA/generated/GP_SoftREPA" "SoftREPA_GP_noref"
run_one "/media/ee303/4TB/SoftREPA/generated/PP_SoftREPA" "SoftREPA_PP_noref"

# Lumina
run_one "/media/ee303/4TB/SoftREPA/generated/lumina/GP" "Lumina_GP_noref"
run_one "/media/ee303/4TB/SoftREPA/generated/lumina/PP" "Lumina_PP_noref"
