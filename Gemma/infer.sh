#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/media/ee303/4TB/Gemma"
INPUT_CSV="/media/ee303/4TB/sam3-body/sam-3d-body/laion_gender_age_race_labeled.csv"
RUN_SECONDS=2500

for index in {6..15}; do
    echo "Starting prompt ${index} at $(date)"
    timeout "${RUN_SECONDS}" python "${SCRIPT_DIR}/long_prompt_label.py" \
        --input-csv "${INPUT_CSV}" \
        --index "${index}" \
        --resume \
        > "${SCRIPT_DIR}/prompt_${index}.out" 2>&1 || status=$?

    status="${status:-0}"
    if [[ "${status}" -eq 124 ]]; then
        echo "Prompt ${index} reached ${RUN_SECONDS}s timeout; switching to next prompt."
    elif [[ "${status}" -ne 0 ]]; then
        echo "Prompt ${index} exited with status ${status}; switching to next prompt."
    else
        echo "Prompt ${index} completed before timeout."
    fi
    unset status
done
