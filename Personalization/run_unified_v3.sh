#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_unified_v3.sh --folder <T2I_FOLDER> [--ref <REF_FOLDER>] [--swap <SWAP_FOLDER>] [--name <EXP_NAME>] [--output <JSON_PATH>] [--summary-jsonl <JSONL_PATH>] [--python <PYTHON_BIN>] [--pose-python <POSE_PYTHON_BIN>] [--dry-run]

Required:
  --folder   Folder containing T2I generated images

Optional:
  --ref      Folder containing reference ID images (if omitted, skip ref-based metrics)
  --swap     Folder containing swapped images (if omitted, pipeline runs in t2i mode)
  --name     Experiment name (default: basename of folder)
  --output   Metadata/result json path (default: <name>_metadata.json)
  --summary-jsonl  Append metric mean/std to this JSONL (default: metrics_summary.jsonl)
  --python   Python executable (default: python)
  --pose-python  Python executable for pose stage only (default: same as --python)
  --dry-run  Print commands only
EOF
}

FOLDER=""      # Input: T2I 圖片資料夾
REF=""         # Input: Reference ID 圖片資料夾
SWAP=""        # Input (optional): 換臉結果資料夾
NAME=""        # Input (optional): 實驗名稱
OUTPUT=""      # Output (optional): 輸出的 metadata/result JSON 路徑
SUMMARY_JSONL=""  # Output (optional): 統計摘要 JSONL 路徑
PYTHON_BIN="python"  # Input (optional): Python 執行檔
POSE_PYTHON_BIN=""  # Input (optional): Pose 階段專用 Python 執行檔
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder)
      FOLDER="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --swap)
      SWAP="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --summary-jsonl)
      SUMMARY_JSONL="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --pose-python)
      POSE_PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$FOLDER" ]]; then
  echo "Error: --folder is required."
  usage
  exit 1
fi

# Output fallback: 若未提供 --output，會在 Python 端輸出為 <name>_metadata.json

CMD=(
  "$PYTHON_BIN" "unified_pipeline_v3.py"  # 執行整合流程
  "--folder" "$FOLDER"                    # Input: T2I 圖片資料夾
)

if [[ -n "$REF" ]]; then
  CMD+=("--ref" "$REF")  # Input (optional): Reference ID 圖片資料夾
fi

if [[ -n "$SWAP" ]]; then
  CMD+=("--swap" "$SWAP")  # Input (optional): 換臉結果資料夾
fi
if [[ -n "$NAME" ]]; then
  CMD+=("--name" "$NAME")  # Input (optional): 實驗名稱
fi
if [[ -n "$OUTPUT" ]]; then
  CMD+=("--output" "$OUTPUT")  # Output: 指定 metadata/result JSON 路徑
fi
if [[ -n "$SUMMARY_JSONL" ]]; then
  CMD+=("--summary-jsonl" "$SUMMARY_JSONL")  # Output: 指定統計摘要 JSONL 路徑
fi
if [[ -n "$POSE_PYTHON_BIN" ]]; then
  CMD+=("--pose-python" "$POSE_PYTHON_BIN")  # Input: Pose 階段專用 Python 執行檔
fi
if [[ $DRY_RUN -eq 1 ]]; then
  CMD+=("--dry-run")  # 僅顯示流程指令，不實際執行模型
fi

"${CMD[@]}"
