# Personalization Evaluation Pipeline

This directory contains the unified evaluation pipeline for Personalization models. It provides an automated, end-to-end framework to calculate essential image generation and person-centric metrics, including **CLIP (T2I/I2I)**, **DINOv2**, **AdaFace (ID)**, **FGA/EvalMuse (Normalized)**, **Scenario**, **Expression**, and **Pose (SAM3D)**.

---

## 核心執行腳本：`run_unified_v3.sh`

`run_unified_v3.sh` 是整個分析流程的直接入口（Bash wrapper），負責將參數向下傳遞給核心排程器 `unified_pipeline_v3.py`，自動完成各階段的推理、評分與統計彙整。

### 設計亮點
1. **雙環境支援 (Dual Environment Routing)**：由於不同模型（如 SAM3D Pose 評估）可能有套件版本衝突（如 `roma` 套件），您可以透過 `--python` 指定主依賴環境（如 `pslz`），並以 `--pose-python` 指定 Pose 專用環境（如 `sam3d`）。
2. **條件式參考圖評估 (Reference-Free Support)**：把 `--ref` 設為選填。如果未提供 Reference 資料夾，系統會聰明地跳過 AdaFace ID 等 I2I 測試，僅執行 CLIP T2I 與其他純 Prompt-dependent 測試。
3. **FGA 正規化與統計輸出**：內部自動將 EvalMuse (FGA) 分數進行 `clip((val-1)/4, 0, 1)` 正規化，並支援將每輪實驗的 Mean / Std 結果統整匯出至獨立的 CSV 與 JSONL 中。
4. **一致的 Pose 標籤映射**：整合了 `gt_maker.py` 與 `eval_pose_v2_a.py`，透過 Regex Fallback 將 Prompt 嚴格對應到 4 大類別 (`Frontal`, `Head_Turn_Left`, `Head_Turn_Right`, `Back_Over_Shoulder`)，確保 Ground Truth 與 Prediction 評分標準一致。

### 參數清單 (Usage)

```bash
bash run_unified_v3.sh --folder <T2I_FOLDER> [OPTIONS]
```

#### Required Arguments
* `--folder <path>`: 包含生成的 T2I (Text-to-Image) 圖片目錄。

#### Optional Arguments
* `--ref <path>`: 參考 ID 圖片目錄。如果**不指定**，則忽略 AdaFace 等 ID 綁定之評估指標。
* `--swap <path>`: 換臉結果目錄（如有特定 Swap 流程可傳入）。
* `--name <string>`: 實驗名稱，如果沒給，預設會使用 `--folder` 路徑的 basename。
* `--output <path>`: 輸出的 Metadata JSON 檔案路徑。預設為 `<name>_metadata.json`。
* `--summary-jsonl <path>`: 指定要匯總結果的 JSONL 檔案（Append 模式），會將平均值和標準差匯入此檔，非常適合進行多次實驗比較（如 `metrics_summary.jsonl`）。
* `--python <path>`: 指定主要執行的 Python Bin 路徑（預設：`python`）。
* `--pose-python <path>`: 特殊指定執行 Pose 評估時的 Python Bin 路徑（可解決 `sam3d` 環境隔離需求）。
* `--dry-run`: 印出將要執行的指令，但不實際調用模型推理。

---

## 執行範例 (Examples)

### 1. 完整流程評估 (T2I + Reference ID + JSONL 統整)
指定兩組虛擬環境，並且輸出整合報表到 `summary_all.jsonl`。
```bash
bash run_unified_v3.sh \
  --folder path/to/generated_images \
  --ref path/to/reference_images \
  --name "My_Experiment_1" \
  --python /path/to/envs/pslz/bin/python \
  --pose-python /path/to/envs/sam3d/bin/python \
  --summary-jsonl summary_all.jsonl
```

### 2. 無參考圖評估 (Reference-Free Evaluation)
不需要比對人物 ID，只評估純生成品質與文本相似度。
```bash
bash run_unified_v3.sh \
  --folder path/to/generated_images \
  --name "NoID_Exp" \
  --python /path/to/envs/pslz/bin/python
```

### 3. 批次執行多個資料夾
若想自動跑完多個資料夾，請搭配 `series_run.sh`，指令內部會使用 `run_one` function 來迭代呼叫 `run_unified_v3.sh`：
```bash
bash series_run.sh
```

---

## 輸出結構說明

執行完畢後，一般會產生以下幾種輸出格式：
1. **[EXP_NAME]_metadata.json**：紀錄每一張圖片 (Image-level) 的所有指標細部分數（包含 Prompt、Pose 取向等）。
2. **Metrics CSV (如 [EXP_NAME]_metrics.csv)**：列出各指標彙整後的易讀格式，方便貼上 Excel 整理。
3. **Summary JSONL**：每次執行結束後新增一列 (一行 JSON) 至 `--summary-jsonl`，紀錄如 `clip_t2i_mean`, `fga_norm_std` 等全局統計數據。
