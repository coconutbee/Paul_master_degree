# T2I Personalization & Pose Alignment Workspace

This file serves as the main directory for data processing, pose alignment, and Text-to-Image (T2I) multi-model benchmarks.

## 1. Data Processing Pipeline (Before Refinement)
Tools used to filter datasets, detect humans, and establish base metrics.
- Computes `person_count` and `person_detected`: `python ./laion_HR/filter_person.py`
- Plot distribution of counts: `python ./laion_HR/plot_person_pie.py`
- Analyzes DeepFashion Data Captions: `python ./sam3-body/analyze_captions.py`

## 2. Pose & Labeling Pipeline 
Predicts head poses and maps them to descriptive prompts (e.g. *turned his head to his left*).
*   **Predict Yaw and Pitch**: `python ./sam3-body/sam-3d-body/infer_v4.py` (or `infer_v2.py` / `sam3_results.csv`)
*   **Angle rules**:
    *   `yaw > 40` or `yaw < -40`: turn head left/right over shoulder
    *   `yaw > 20` or `yaw < -20`: turn head left/right
    *   `-25 < yaw < 25`: face forward
    *   `pitch > 25`: chin up
    *   `pitch < -25`: chin down
*   **Map Angles to Prompts**: `python ./sam3-body/label.py`

## 3. Prompt Refinement (Gender, Age, Race)
Updates subject info via DeepFace & Qwen to refine captions properly.
*   **Standard Auto Pipeline**: `bash ./SoftREPA/tools/run_gender_race_pipeline.sh ./SoftREPA/celeb_imgs ./SoftREPA/tools/final_prompt.csv`
*   **Famous Celebrities Pipeline**: `bash ./SoftREPA/tools/run_gender_race_pipeline.sh ./SoftREPA/more_famous ./SoftREPA/tools/final_prompt.csv`
*   **DeepFace labeler**: `python ./SoftREPA/tools/deepface/race_label.py`
*   **Format Rewrite**: `python ./SoftREPA/tools/rewrite_caption_rf.py`
*   **Qwen Utils**: `./sam3-body/qwen_infer.py` & `qwen_batch_infer.py`

## 4. Multi-Model Inference (T2I Generation)
All models read carefully aligned prompts from: 
`./SoftREPA/tools/final_prompt_test_result_all_pose.csv`

### 4.1 Environment Execution Map
| Model | Directory | Est. Latency |
|-------|-----------|--------------|
| **SoftREPA** | `./SoftREPA` | ~4.0s |
| **Lumina** | `./SoftREPA/Lumina` | ~13.0s |
| **Flux2** | `./flux2` | ~3.0s |
| **Sana1.5** | `./Sana` | ~3.0s |
| **Emu3.5** | `./Emu3.5` | ~2.0s |
| **Janus-Pro 7B** | `/media/ee303/disk1/Janus` | ~7.0s |
| **Infinity** | `./Infinity` | ~0.5s |
| **Hart** | `./hart` | ~0.4s |

### 4.2 Explicit Inference Commands
**SoftREPA T2I**
```bash
python sample.py \
    --model sd3 --use_dc --use_dc_t True \
    --n_dc_tokens 4 --n_dc_layers 5 \
    --img_size 1024 \
    --NFE 28 --cfg_scale 4 \
    --load_dir "tokens/sd3" \
    --save_dir "generated/SoftREPA" \
    --datadir "./Generic_prompts"  # or Posture_prompts
```
**Lumina T2I**
```bash
python Lumina_inference.py --input ./SoftREPA/Posture_prompts/pose_prompts.jsonl --output_dir generated/lumina/PP
```
**Infinity T2I**
```bash
cd ./Infinity
bash scripts/batch_infer.sh ./SoftREPA/tools/final_prompt_test_result_all_pose.csv generated/Posture_prompts/
```

**Hart T2I**
```bash
cd ./hart
python sample.py \
    --model_path "hart-0.7b-1024px/llm" \
    --text_model_path "hart-0.7b-1024px/llm" \
    --shield_model_path "shieldgemma-2b" \
    --prompt_csv "/media/ee303/4TB/SoftREPA/tools/8prompt.csv" \
    --sample_folder_dir "8prompt/"
```
** Emu3.5 T2I**
```bash
cd ./Emu3.5
python inference.py
```
** Janus-Pro 7B T2I**
```bash
conda activate diffusers
cd /media/ee303/disk1/Janus
python inference.py
```
** Flux2 T2I**
```bash
conda activate paul
cd /media/ee303/4TB/flux2
python inference.py
```
** Sana1.5 T2I**
```bash
conda activate paul
cd /media/ee303/4TB/Sana
python inference.py
```
**Note: For Hart, Emu3.5, Janus-Pro 7B, Flux2, and Sana1.5, ensure the `prompt_csv` or input paths are correctly set to read from the aligned prompts CSV.*

## 5. Benchmarking, Analytics & Visualization
Run pipeline metrics across generated samples (CLIP, DINO, HPS, ImageReward, FID with COCO-val 1K, LPIPS).
*   **Execute Evaluator**: 
    ```bash
    conda activate sam3d
    cd ./Personalization
    bash run_unified_v3.sh \
      --folder /media/ee303/disk2/JACK/ECCV_DATA/T2I_20_prompts \
      --swap /media/ee303/disk2/JACK/ECCV_DATA/Infinity_20prompts \
      --name infinity_noref --output infinity_noref_metadata.json \
      --summary-jsonl metrics_summary.jsonl
    ```
*   **Execute Evaluator**: 
    ```bash
    conda activate sam3d
    cd ./Personalization
    bash series_run.sh
    ```

*   **Visualize Image Grids**:
    ```bash
    conda activate sam3d
    python SoftREPA/show_image.py -f generated/SoftREPA -c 15
    python SoftREPA/show_image.py -f generated/lumina/PP -c 15
    ```
*   **Interactive Streamlit Review UI**:
    ```bash
    cd ./Personalization
    conda activate pslz
    streamlit run pose_result_analyzer_app.py
    ```

## 6. (Legacy) SoftREPA Training Details & Reward Functions

### 6.1 Architectures & Formulations
SoftREPA is trained with paired COCO data (118K images) and DeepFashion data (25K images in `./DeepFashion_Training_Final`).

It uses Diffusion-DPO (Direct Preference Optimization for Diffusion) and DDPO (Denoising Diffusion Policy Optimization).
$$ \mathcal{L}_{DPO}(\theta; \theta_{\text{ref}}) = -\mathbb{E}_{(x_w, x_l, c)} \left[ \log \sigma \left( \beta \cdot (\text{err}(x_l, \theta) - \text{err}(x_l, \theta_{\text{ref}})) - \beta \cdot (\text{err}(x_w, \theta) - \text{err}(x_w, \theta_{\text{ref}})) \right) \right] $$

**Diagram References (from documentation):**
- System Arch: `![Architecture](Architecture.png)` & `![Scoring Module](Scoring_Module.png)`
- DPO Details: `![Diffusion DPO](image.png)`, `![DPO Diagram 1](image-1.png)`, `![DPO Diagram 2](image-2.png)`
- Reward Visuals: `![Reward Model Selection](image-3.png)`, `![Reward Score](image-4.png)`, `![Mean Reward Score](image-5.png)`

### 6.2 Data Preparation & Training Usage
*   **Prepare training data**:
    `python ./SoftREPA/prepare_training_data_from_csv.py` (Outputs to `./sam3-body/sam3_labeded_training/deepfashion`)
*   **Execute Training (Single GPU)**:
    `bash ./SoftREPA/run_train_single_gpu.sh` (Outputs to `./SoftREPA/data/deepfashion`)

## 7. LAION-17K Dataset & Auxiliary Scripts
*   **Dataset Path**: `./Personalization/laion_gender_age_race_long_captioned_prompt19_bible.csv`
*   **Long Caption Labeling with Gemma**:
    ```bash
    cd ./Gemma
    conda activate paul
    python long_prompt_label.py
    ```
    *(Note: Modify `system prompt` directly from `./Gemma/prompt.py` when using this).*
