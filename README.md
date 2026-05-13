# T2I Personalization & Pose Alignment Workspace

This file serves as the main directory for data processing, pose alignment, and Text-to-Image (T2I) multi-model benchmarks.

## 1. Data Processing Pipeline (Before Refinement)
Tools used to filter datasets, detect humans, and establish base metrics.
- Computes `person_count` and `person_detected`: `python /media/ee303/4TB/laion_HR/filter_person.py`
- Plot distribution of counts: `python /media/ee303/4TB/laion_HR/plot_person_pie.py`
- Analyzes DeepFashion Data Captions: `python /media/ee303/4TB/sam3-body/analyze_captions.py`

## 2. Pose & Labeling Pipeline 
Predicts head poses and maps them to descriptive prompts (e.g. *turned his head to his left*).
*   **Predict Yaw and Pitch**: `python /media/ee303/4TB/sam3-body/sam-3d-body/infer_v4.py` (or `infer_v2.py` / `sam3_results.csv`)
*   **Angle rules**:
    *   `yaw > 40` or `yaw < -40`: turn head left/right over shoulder
    *   `yaw > 20` or `yaw < -20`: turn head left/right
    *   `-25 < yaw < 25`: face forward
    *   `pitch > 25`: chin up
    *   `pitch < -25`: chin down
*   **Map Angles to Prompts**: `python /media/ee303/4TB/sam3-body/label.py`

## 3. Prompt Refinement (Gender, Age, Race)
Updates subject info via DeepFace & Qwen to refine captions properly.
*   **Standard Auto Pipeline**: `bash /media/ee303/4TB/SoftREPA/tools/run_gender_race_pipeline.sh /media/ee303/4TB/SoftREPA/celeb_imgs /media/ee303/4TB/SoftREPA/tools/final_prompt.csv`
*   **Famous Celebrities Pipeline**: `bash /media/ee303/4TB/SoftREPA/tools/run_gender_race_pipeline.sh /media/ee303/4TB/SoftREPA/more_famous /media/ee303/4TB/SoftREPA/tools/final_prompt.csv`
*   **DeepFace labeler**: `python /media/ee303/4TB/SoftREPA/tools/deepface/race_label.py`
*   **Format Rewrite**: `python /media/ee303/4TB/SoftREPA/tools/rewrite_caption_rf.py`
*   **Qwen Utils**: `/media/ee303/4TB/sam3-body/qwen_infer.py` & `qwen_batch_infer.py`

## 4. Multi-Model Inference (T2I Generation)
All models read carefully aligned prompts from: 
`/media/ee303/4TB/SoftREPA/tools/final_prompt_test_result_all_pose.csv`

### Environment execution map:
| Model | Directory | Execution Details | Est. Latency |
|-------|-----------|-------------------|--------------|
| **SoftREPA** | `/media/ee303/4TB/SoftREPA` | `python sample.py --model sd3 --use_dc --use_dc_t True...` | ~4.0s |
| **Lumina** | `/media/ee303/4TB/SoftREPA/Lumina` | `python Lumina_inference.py` | ~13.0s |
| **Flux2** | `/media/ee303/4TB/flux2` | `conda activate t2i; python inference.py` | ~3.0s |
| **Sana1.5** | `/media/ee303/4TB/Sana` | `conda activate t2i; python inference.py` | ~3.0s~ |
| **Emu3.5** | `/media/ee303/4TB/Emu3.5` | `conda activate t2i; python sample.py --cfg configs/example_config_t2i.py` | ~2.0s |
| **Janus-Pro 7B** | `/media/ee303/disk1/Janus` | `conda activate t2i; python inference.py` | ~7.0s |
| **Infinity** | `/media/ee303/4TB/Infinity` | `conda activate t2i; bash scripts/batch_infer.sh` | ~0.5s |
| **Hart** | `/media/ee303/4TB/hart` | `conda activate hart; python sample.py` | ~0.4s |

## 5. Benchmarking & Analytics (Personalization)
Run pipeline metrics across generated samples.
*   **Execute Evaluator**: 
    ```bash
    cd /media/ee303/4TB/Personalization
    bash series_run.sh
    ```
*   **Interactive Streamlit Review UI**:
    ```bash
    cd /media/ee303/4TB/Personalization
    conda activate pslz
    streamlit run pose_result_analyzer_app.py
    ```
*   **Streamlit (Before vs After Refinement)**: `streamlit run /media/ee303/4TB/sam3-body/app.py`

## 6. (Legacy) SoftREPA Training Details
- Paired COCO dataset (118K) & DeepFashion dataset (25K in `4TB/DeepFashion_Training_Final`)
- Uses Diffusion-DPO and DDPO for policy optimization.

## 7. LAION-17K Dataset
- /media/ee303/4TB/Personalization/laion_gender_age_race_long_captioned_prompt19_bible.csv

## Long caption label
```bash
cd /media/ee303/4TB/Gemma
conda activate paul
python long_prompt_label.py
```
Noted: modify 'system prompt' from '/media/ee303/4TB/Gemma/prompt.py'