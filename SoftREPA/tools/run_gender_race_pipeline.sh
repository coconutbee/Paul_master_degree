#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_gender_race_pipeline.sh <input_image_folder> <output_csv> [method]"
  exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_CSV="$2"
METHOD="${3:-custom}"
SAM3D_ENV="${SAM3D_ENV:-sam3d}"
MIVOLO_ENV="${MIVOLO_ENV:-mivolo}"
INFINITY_ENV="${INFINITY_ENV:-infinity}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIVOLO_DIR="${SCRIPT_DIR}/MiVOLO"
DEEPFACE_SCRIPT="${SCRIPT_DIR}/deepface/race_label.py"
POSE_SCRIPT="/media/ee303/4TB/sam3-body/sam-3d-body/infer_v2_for_custom.py"
SAM3_ROOT="/media/ee303/4TB/sam3-body/sam-3d-body"
PROMPT_COMPOSE_SCRIPT="${SCRIPT_DIR}/compose_prompt_from_labels.py"
INFINITY_ROOT="/media/ee303/4TB/Infinity"
INFINITY_RUN_NAME="generated/Judy_test"

POSE_CSV="${OUTPUT_CSV%.csv}_pose.csv"
GENDER_CSV="${OUTPUT_CSV%.csv}_gender_age.csv"
GENDER_RACE_CSV="${OUTPUT_CSV%.csv}_gender_race.csv"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found. Please install/initialize conda first."
  exit 1
fi

# Load conda shell hooks for non-interactive bash.
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[INFO] Activating conda env: ${SAM3D_ENV}"
conda activate "${SAM3D_ENV}"

if ! (
  cd "${SAM3_ROOT}" && \
  PYTHONPATH="${SAM3_ROOT}:${PYTHONPATH:-}" python "${POSE_SCRIPT}" \
    --input_dir "${INPUT_FOLDER}" \
    --output_csv "${POSE_CSV}"
); then
  echo "[ERROR] SAM3D pose step failed."
  echo "Hint: ensure SAM3D deps are installed in current env (e.g., braceexpand)."
  echo "Try: pip install braceexpand"
  exit 1
fi

echo "[INFO] Activating conda env: ${MIVOLO_ENV}"
conda activate "${MIVOLO_ENV}"

python "${MIVOLO_DIR}/gender_label.py" \
  --method "${METHOD}" \
  --input_folder "${INPUT_FOLDER}" \
  --json "" \
  --csv_out "${GENDER_CSV}"

python "${DEEPFACE_SCRIPT}" \
  --input_csv "${GENDER_CSV}" \
  --output_csv "${GENDER_RACE_CSV}" \
  --image_col image_path

python "${PROMPT_COMPOSE_SCRIPT}" \
  --pose_csv "${POSE_CSV}" \
  --attr_csv "${GENDER_RACE_CSV}" \
  --output_csv "${OUTPUT_CSV}"

echo "[INFO] Activating conda env: ${INFINITY_ENV}"
conda activate "${INFINITY_ENV}"

cd "${INFINITY_ROOT}"
bash scripts/batch_infer.sh "${OUTPUT_CSV}" "${INFINITY_RUN_NAME}"

echo "Pipeline done. Final CSV: ${OUTPUT_CSV}"


#bash /media/ee303/4TB/SoftREPA/tools/run_gender_race_pipeline.sh  /media/ee303/4TB/SoftREPA/celeb_imgs /media/ee303/4TB/SoftREPA/tools/final_prompt.csv