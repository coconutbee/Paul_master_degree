# Experiments Log — Stop the Ghosts (flux2)

Snapshot of this session's pipeline + results, so work can resume later.
Paper source: `paper/main.tex` (+ `paper_overleaf.zip`). Method = **Untangle** (B1+B2) + learned **structure gate**.

---

## 1. Scripts created / modified this session

| Script | Env | What it does |
|---|---|---|
| `make_paper_figures_clean.py` | any (PIL) | Rebuilds `paper/figures/teaser.png` + `qualitative.png` from plain-bg `ghost_demo` picks (red/green badges). |
| `make_blur_tradeoff_fig.py` | any (mpl) | Blur trade-off fig, recolored red=leakage / green=good. |
| `smplest_x_angles_kit/annotate_laion17k_smplx.py` | smplestx | SMPLest-X pose annotation of all 17,710 LAION imgs → `Gemma/laion17k_smplx_pose.csv` (resumable). |
| `regen_captions_smplx.py` | any (pandas) | Re-filter to turned subset (\|head_to_torso_yaw\|>20) + regen captions from SMPLest-X → `Gemma/laion10k_smplx_turned.csv`. |
| `prepare_pose_lora_subset.py` (existing) | pandas | Built `training_data_pose2k_smplx/metadata.csv` (2k, from LAION-10K via `Gemma/laion10k_smplx_for_prep.csv`). |
| `run_big_eval.sh` | softrepa+sam3d | 380-pair benchmark @1MP: stock/b1b2/person_only/full/gate → `cluttered_eval_big/`. |
| `run_gate_retrain.sh` | softrepa | Retrain gate on LAION-10K → `flux2-posegate-pose2k-gate-spatial-v3/`. |
| `run_gate_v3_eval.sh` | softrepa+sam3d | gate_v3 @1MP + stock/b1b2/gate_v3 @0.3MP (`cluttered_eval_big_03/`). |
| `run_baselines_eval.sh` | softrepa+sam3d | OminiControl γ + OminiControl2 asym baselines → `cluttered_eval_big/{omini_gamma,omini2_asym}`. |
| `clip_eval_big.py` | softrepa (CPU) | CLIP-T/CLIP-I over 380. **VARIANTS edited per-run** (currently `stock,omini_gamma,omini2_asym`). |
| `pose_guardrail_big.py` | softrepa | med\|Δyaw\| vs stock + same-class, merges CLIP. Reads `YAW_JSON`. **VARIANTS edited per-run.** |
| `measure_yaw.py` (existing) | smplestx | head-to-torso yaw per dir → json. |
| **`flux2_ghosting_attn.py`** | — | **Added** `enable_gamma`/`gamma_strength` (OminiControl const γ on output→depth) + `enable_asym` (OminiControl2 depth↛output mask) to `GhostingController`. |
| **`batch_pose_transfer.py`** | — | **Added** flags `--ghost-gamma --gamma-strength --ghost-asym`; wired into `maybe_install_ghosting`. |

**Reproduce note:** `clip_eval_big.py` / `pose_guardrail_big.py` `VARIANTS` were rewritten in place for each run; the saved `cluttered_eval_big/clip_summary.csv` + `guardrail_summary.csv` currently hold the **baseline** run, not gate_v3. Reset VARIANTS to reproduce a specific row.

## 2. Key data CSVs

- `Gemma/laion17k_smplx_pose.csv` — SMPLest-X annotation, 17,710 rows (adds `smplx_*`).
- `Gemma/laion10k_smplx_turned.csv` — **9,806** turned-head subset + regenerated captions (`long_caption_smplx`).
- `training_data_pose2k_smplx/metadata.csv` — 2k gate-training subset (1844 unique imgs + `person_masks/`).

## 3. Results — 380-pair cluttered benchmark (`cluttered_eval_big`)

### Ghosting (bg_corr_hp ↓, n=380)
| Method | bg_corr_hp | bg_corr_lum | bg_bright |
|---|---|---|---|
| stock | +0.0557 ± 0.1431 | +0.112 | 0.270 |
| (A) person-only depth | +0.0180 ± 0.0882 | −0.047 | 0.193 |
| **B1+B2 (Untangle)** | +0.0042 ± 0.0291 | −0.050 | 0.288 |
| (A+B1+B2) full | +0.0057 ± 0.0368 | −0.048 | 0.208 |
| gate (old spatial-v2, SAM3D) | +0.0168 ± 0.0736 | −0.012 | 0.280 |
| **gate_v3 (retrained, LAION-10K)** | **+0.0090 ± 0.0467** | −0.034 | — |
| *OminiControl γ (const 2.5)* | +0.0017 ± 0.0240 | −0.059 | 0.293 |
| *OminiControl2 asym mask* | +0.0025 ± 0.0248 | −0.045 | 0.291 |

→ stock→B1+B2 = ~92% mean reduction, ~5× variance collapse. **The two in-context-control baselines match B1+B2** (no clear winner on this benchmark).

### Resolution (gate_v3, `cluttered_eval_big_03` @0.3MP, n=380)
| Method | @1MP | @0.3MP |
|---|---|---|
| stock | +0.056±0.143 | +0.047±0.128 |
| B1+B2 | +0.004±0.029 | +0.002±0.026 |
| **gate_v3** | **+0.009±0.047** | **+0.010±0.047** |

→ gate_v3 **resolution-stable** (gap of old SAM3D gate closed). Retraining on corrected poses, not higher-res, was the binding constraint.

### Guardrail (n=380): pose vs stock / CLIP
| Method | med\|Δyaw\| / same-cls | CLIP-T | CLIP-I |
|---|---|---|---|
| stock | 0.0 / 100% | 0.289 | 0.940±0.072 |
| B1+B2 | 1.75° / 79.9% | 0.287 | 0.978±0.026 |
| gate_v3 | 1.22° / 82.3% | 0.288 | 0.968±0.038 |
| OminiControl γ | 1.86° / 78.1% | 0.287 | 0.987±0.012 |
| OminiControl2 asym | 1.77° / 79.9% | 0.287 | 0.976±0.029 |

→ all preserve pose (med\|Δyaw\|<2°), keep CLIP-T flat, improve CLIP-I. Baselines comparable.

### LAION-10K composition (SMPLest-X, 9,806 turned)
yaw: turn-L 35.9 / OTS-L 17.6 / turn-R 32.0 / OTS-R 14.5 (L 53.5/R 46.5, OTS 32.2%).
pitch (\|p\|≤90): chin-up **1.8** / level 84.5 / chin-down 13.7. race white 73 / black 14 / asian 7. gender F 64. age adult 88.
SAM3D bug: head_body_yaw has a frontal dead zone (0 values in \|yaw\|≤20) → 51% of captions were relabelled.

## 4. Paper state
- ✅ Figures 1/3 redone; blur fig recolored.
- ✅ `tab:main` n=56→380; `tab:guardrail` n=380; `tab:learned` n=380 + gate_v3 (resolution-stable framing); abstract/intro/contributions/conclusion gate claims updated (~0.006 of B1+B2, resolution-stable).
- ✅ LAION-17K→LAION-10K everywhere (SMPLest-X, turned filter).
- ✅ Related Work: added OminiControl2/ContextDrag/ChronoEdit/VisualCloze cites + "Relation to in-context control mechanisms" para + 2 future-work extensions (KV-cache, grid-layout).
- ⏳ **OPEN:** the two empirical baselines (γ, asym) are computed but **not yet written into `tab:main`** — pending a framing decision (they match B1+B2; honest options: lead with diagnosis+metric+gate / dig for fine-pose differences / table-only). Bib for ContextDrag/ChronoEdit/VisualCloze uses placeholder "et al." authors — verify against arXiv 2512.08477 / 2510.04290 / 2504.07960.
- ⏳ `tab:learned` LoRA + λ-sweep rows dropped to text (still n=56 pilot).

---

# 2026-06-05 — Cross-ID pose-transfer: which feature map (control) best re-angles the SAME identity?

**Question.** A = source image (RGB, the identity/appearance to keep), B = a *different* target image (the wanted angle). Feed FLUX.2-klein `image=[control_B, rgb_A]` **and** rewrite the prompt's pose clause to B's angle (keep A's pronoun). Does the output = A's id at B's angle? Compare 4 control feature maps: `normal`, `mesh_depth`(depth), `skeleton`(SMPL-X), `sam3d`.

## Scripts created this thread
| Script | Env | What it does |
|---|---|---|
| `sam3-body/sam-3d-body/render_sam3d_control.py` | sam3d | Render FLUX-consumable SAM3D 70-kpt 2D skeleton control `{stem}_sam3d.png` (original image frame, person-only black bg) + `{stem}_sam3d3d.png` (3-view 3D, display only). |
| `sam3-body/sam-3d-body/extract_3d_skeleton.py` | sam3d | General per-image 70-kpt 3D skeleton extractor → npy + 3-panel viz. |
| `flux2_control_test.py` (modified) | softrepa | Added `--src-dir` mode: plain image folder = RGB, controls from `--ctrl-dir`, `image=[control,rgb]`. (legacy `--dir`/`_input.png` mode kept.) |
| `flux2_pose_transfer.py` | softrepa | The cross-pairing generator. Parses `<id> turned <pron> head <dir>[ over the shoulders] and chin <up/down>. <scene>` from filenames; builds prompt = **A id+scene, B angle, A pronoun**. `--src-idx/--tgt-idx` (or `all`), `--skip-self`, **resume (skip existing outputs)**. |
| `make_ctrlgen_grid.py`, `make_control_grid.py`, `make_pose_transfer_grid.py` | any (PIL) | Comparison grids. `make_pose_transfer_grid.py`: rows=source ID(RGB), cols=target angle(control thumb), one grid per control. |
| `sam3-body/sam-3d-body/eval_pose_angles.py` | sam3d | (superseded) SAM3D yaw/pitch eval. |
| `smplest_x_angles_kit/eval_angles_resume.py` | smplestx | **Angle eval used for metrics.** Imports `CAngleExtractor` from `extract_angles_c.py`; resume + per-row flush. |
| `flux2/make_pose_metrics.py` | any | Joins manifest + source/output angle CSVs → per-control yaw/pitch MAE, baseline, gain, dir-acc. |
| `Personalization/AdaFace/eval_adaface_id.py` | adaface | ID fidelity: AdaFace ir_101 cosine(output, source-A); caches 40 source feats; resume. |

## Adjustments / details (IMPORTANT — easy to get wrong)
- **Largest-person selection.** SAM3D/`infer_v4` use `outputs[0]` = *first* detection, which can be a spurious edge bbox (e.g. a 21px-wide person) → near-empty skeleton. Fixed in `render_sam3d_control.py` + `extract_3d_skeleton.py` (pick `max bbox area`, matching `render_controls.py`). `eval_pose_angles.py`/`extract_angles_c.py` already pick largest.
- **Used `extract_angles_c.py` (SMPLest-X), NOT SAM3D, for the metric** (user directive). It gives absolute `head_yaw` (+=camera-right), `head_pitch` (+=chin up), plus `head_to_torso_yaw`, `body_yaw`. Original CLI writes once at end / no resume → wrote `eval_angles_resume.py` wrapper.
- **Sign convention validated** on the 40 sources: left→mean yaw −50°, right→+16°; chin-up→+21°, chin-down→−4°; label-vs-measured sign agreement yaw 85% / pitch 82% (that 85% is the *source generations'* own prompt fidelity = a fair reference baseline).
- **Convention-free metric:** output and target B both measured by the same SMPLest-X extractor → compare directly, no sign mapping needed.
- **Control thumbnail in grids** = the *exact* 2D image fed to FLUX (`{stem}_sam3d.png`), not the 3-view `_sam3d3d.png` (which is display-only) — fixed earlier to avoid "is it feeding 3 images?" confusion. FLUX gets 1 image (control) or 2 (`[control, rgb]`), never 3.
- **Filename stems** end in a stray `.` (`...gi..png`); all scripts `rstrip(".")` consistently so control/output names line up.
- **Source crops are half-body** → skeleton legs are model-extrapolated; head/upper-body angle (what we measure) is reliable.

## Data layout (`FT_test_40TP/batch_inference/`)
- `pretrained/` — 40 source RGBs (`<prompt>.png`). `pretrained_controls/` — `{stem}_{normal,mesh_depth,skeleton,sam3d}.png` (+ `_sam3d3d`, `_skeleton3d`, `_input`).
- `pose_transfer_full/` — **6400** outputs `src{A:02d}__tgt{B:02d}__{ctrl}.png` (40×40×4) + `manifest.csv` (src/tgt/control/prompt) + `pose_metrics_{summary,detail}.csv`.
- `angles_sources_smplx.csv` / `angles_outputs_smplx.csv` — SMPLest-X angles. `id_adaface.csv` — AdaFace ID cosine (running).
- `pose_transfer/` — earlier 5×4×4=80 pilot (src 0,9,15,21,29 × tgt 6,10,14,22). `spotcheck.png`, `sam3d_wins.png` — montages.

## Results — yaw transfer accuracy (cross-pairs only, n=1560/control)
| control | yawMAE↓ | baseline(do-nothing) | gain↑ | pitchMAE | yaw dir-acc | pitch dir-acc |
|---|---|---|---|---|---|---|
| **sam3d** | **28.2°** | 52.5° | **24.3°** | 10.1° | **80.3%** | 75.6% |
| skeleton (SMPL-X) | 34.9° | 52.5° | 17.6° | 10.1° | 72.1% | 72.6% |
| normal | 39.3° | 52.5° | 13.2° | 10.3° | 68.7% | 72.4% |
| mesh_depth | 39.4° | 52.5° | 13.1° | 10.2° | 68.6% | 72.4% |

**By target-angle magnitude (yawMAE; no target had |yaw|<20):**
| bin | normal | mesh_depth | skeleton | sam3d |
|---|---|---|---|---|
| mid 20–45° (n=507) | 38.5 | 38.0 | 37.2 | **27.7** |
| big ≥45° (n=1053) | 39.7 | 40.0 | 33.7 | **28.4** |

**Takeaways.** (1) Same-ID angle transfer works — all controls beat the do-nothing baseline (52.5°). (2) **sam3d wins decisively** (28° MAE, 80% dir-acc): sparse 70-kpt skeleton carries angle without B's body shape. (3) Ranking sam3d > skeleton > normal ≈ depth; dense maps don't improve at big angles (normal/depth flat ~40°), skeleton improves, sam3d is angle-magnitude-robust. (4) pitch ≈ equal (~10°) across controls. Best sam3d-advantage cases (`sam3d_wins.png`): e.g. asian woman → +57° target, sam3d err 3° vs other maps ≥117°. Supports the thesis that the sparse structure gate is the right pose signal.
## ID fidelity (AdaFace ir_101 cos, output vs source-A) — `id_adaface.csv`, `make_id_vs_angle.py`, `make_id_align_panels.py`
- **Face-alignment caveat:** MTCNN can't align extreme-profile faces → no CSIM. **12/40 source-A images themselves are un-alignable** (babies + extreme crops) → 1920/6400 rows have `src_aligned=0`; exclude those (not the feature map's fault). Conditioned on source alignable (n=1120/control):

| control | out-align rate | CSIM (aligned) | CSIM (fail=0, honest) |
|---|---|---|---|
| normal | **91.8%** | **0.600** | **0.551** |
| mesh_depth | 91.9% | 0.593 | 0.545 |
| skeleton | 88.8% | 0.556 | 0.494 |
| sam3d | 84.5% | 0.563 | 0.476 |

- `out-align rate` = fraction of outputs with an AdaFace-detectable face. sam3d turns most → 7–8 pp more often produces an un-detectable (extreme-profile) face. CSIM(fail=0) counts those as ID 0.
- `id_vs_angle_{target,output}.png` + `id_align_panels.png` (2×2: CSIM & align-rate × target/output yaw, 5° bins). Once a face aligns, ID holds ~0.55–0.65 up to 50–65°; align-rate drops to 0.4–0.6 only in the 75–85° tail (skeleton/sam3d only).
- **Clean Pareto trade-off (accuracy ↔ ID):** sam3d = best angle (yawMAE 28°) / worst honest ID (0.476); normal,depth = worst angle (39°) / best ID (0.55); skeleton in the middle on both. Dense maps "keep ID" largely by under-rotating (staying frontal → easy to align). The right operating point depends on whether the task weights pose accuracy or identity more.

---

# 2026-06-05 — T2I model comparison: pose accuracy across 6 generators (40 prompts) [Janus-Pro added 2026-06-08]

6 text-to-image models (FLUX.2-klein, Sana, SoftREPA/SD3, Infinity, HART, **Janus-Pro-7B**), each generating the SAME 40 pose prompts (1 img/prompt). Measure pose accuracy = does the generated head pose match the prompt's intended pose. `t2i_pose_compare/` (`SUMMARY.csv` + `SUMMARY_with_janus.csv`, `{model}_pose_acc.csv`, `{model}_angles.json`, `pose_compare.png` + `angle_hist_with_janus.png`). Runner `_run_t2i_pose_compare.sh` (stage A) + `_run_t2i_poseB.sh` (stage B). **Janus-Pro added via `_run_janus_pose.sh` (folder `/media/ee303/disk1/Janus/prompt_test_512/Janu_40TP`); 6-model summary + histogram rebuilt by `make_t2i_compare.py`.**

**Pipeline (2 stage, 2 env):** A) `extract_angles_c.py <folder> --out _angles.json` (env smplestx) → SMPLest-X head_yaw/pitch/head_to_torso. B) `eval_smplestx_vs_prompt.py --folder --angles --out` (env paul) → T5Gemma classifies the prompt's intended pose, maps angles→labels, computes yaw/pitch match.

**Thresholds (user-specified, uniform across all 6):** yaw >15° = left/right, pitch >±15° = up/down, |head_to_torso_yaw| >25° = "over the shoulder". (Note: the older `pretrained/pose_acc_smplestx.csv` used the docstring rule pitch±25/OS35 — different; the `t2i_pose_compare/` numbers use 15/15/25.)

| model | yaw acc | pitch acc | mean \|yaw\| | mean \|pitch\| | SMPLest-X fail |
|---|---|---|---|---|---|
| **flux2** | **67.5%** | 40.0% | 50.4° | 14.0° | 0/40 |
| sana | 55.0% | 42.5% | 53.7° | 15.2° | 0/40 |
| softrepa | 42.5% | 37.5% | 59.6° | 15.8° | 0/40 |
| infinity | 35.0% | 37.5% | 30.2° | 13.8° | 0/40 |
| hart | 32.5% | 42.5% | 38.2° | 16.8° | 0/40 |
| **janus-pro-7B** | **5.0%** | 25.0% | **5.5°** | 14.1° | 0/40 |

**Takeaways.** flux2 best yaw fidelity (67.5%), sana 2nd (55%); infinity/hart worst (~33%). mean|yaw| shows *willingness to turn*: softrepa turns most (59.6°) but imprecise (42.5%); **infinity under-rotates (30°) → lowest yaw acc**; flux2 turns a lot (50°) AND accurately. **Janus-Pro is by far the worst (yaw 5.0%): it essentially does NOT turn the head at all — mean |yaw| 5.5°, median 4°, near-frontal for nearly every prompt** (only 2/40 yaw-correct), and its pitch is the lowest too (25%, mostly wrong-signed). Pitch ~37–42% for the others (chin is hard for every T2I, model-independent). All 6 had a person detected in 40/40. All prompts specify chin → pitch denominator = 40.

### Angle distributions (`angle_hist.png` / `angle_hist_with_janus.png`, 5° bins): |yaw|, pitch split by prompt chin up/down
Balanced design: each model has exactly 20 chin-up + 20 chin-down prompts. Median measured angle:

| model | median \|yaw\| | chin-UP median pitch | chin-DOWN median pitch |
|---|---|---|---|
| flux2 | 50.8° | +20.2 | −4.5 |
| sana | 52.9° | +18.1 | −2.8 |
| infinity | 30.7° | +15.8 | −5.8 |
| softrepa | 59.5° | +18.5 | **+8.2** ⚠ |
| hart | 43.3° | +24.2 | **+8.7** ⚠ |
| **janus-pro-7B** | **4.0°** ⚠ | +14.9 | **+11.7** ⚠ |

**The pitch failure is almost entirely chin-DOWN.** Chin-up is handled well by all (measured pitch peaks at +15–25°, clears the +15° threshold). Chin-down should be negative but the distributions straddle 0: flux2/sana/infinity only weakly negative (−3 to −6°), and **softrepa/hart/janus actually go positive (median +8 to +12°) — they raise the chin when asked to lower it.** This is why pitch acc sits at ~40% (25% for janus); splitting up/down exposes it (the aggregate pitch_acc hid the asymmetry). |yaw| panel confirms infinity under-rotates (mass at 0–35°) and **janus collapses to ~0° (it does not turn at all)**, while softrepa/sana/flux2 sit at 40–65°. → reinforces adopting **flux2** as the backbone (best yaw fidelity *and* willingness to turn).


# 2026-06-06 — FLUX.2 ID-personalization + pose (60 IDs × 40 prompts, 2400/exp)

ID images = `SoftREPA/ID_image` (60). Prompts = the 40 pose prompts, **kept UNCHANGED incl. identity words** (user directive). Two experiments, same 60×40 pairing:
- **exp1** `idpose_exp1/` — `image=[ID]` + prompt (no pose control).
- **exp2** `idpose_exp2_sam3d/` — `image=[sam3d_control_i, ID]` + prompt; sam3d control = the SAM3D skeleton of `pretrained/{prompt_i}` (the flux2 t2i of that prompt).

Scripts: `flux2_id_pose.py` (gen, `--control none|sam3d|...`, `--ids all`, resume), `Personalization/AdaFace/eval_adaface_idref.py` (CSIM vs the ID image), `make_idpose_poseacc.py` (manifest-driven pose acc, reuses `eval_smplestx_vs_prompt` classifier+mapping), `make_idpose_analysis.py` (CSIM + |yaw|/pitch hists `idpose_angle_hist.png`). Thresholds yaw15/pitch15/os25. Runners `_run_idpose.sh`, `_run_idpose_eval.sh`.

| metric | exp1 (ID only) | exp2 (sam3d+ID) |
|---|---|---|
| pose YAW acc | **73.7%** (1769/2399) | 58.5% (1405/2400) |
| pose PITCH acc | 46.8% | **51.1%** |
| ID CSIM (aligned) | **0.153** | 0.122 |
| ID CSIM (fail=0) | 0.134 | 0.101 |
| output face-align rate | 87.6% | 82.5% |
| median \|yaw\| | 45.2° | 48.7° |

**Findings.** (1) **ID is essentially NOT preserved (CSIM ~0.12–0.15 ≈ different person;** AdaFace same-person is 0.3–0.5+). The unchanged prompt's identity words ("A black man…") override the ID image (e.g. Marilyn ID → black-man output). Direct cost of keeping the prompt fixed. (2) **Adding the sam3d control HURT yaw (73.7→58.5%)**: the control is the SAM3D skeleton of the pretrained flux2 image (own yaw acc only 67.5%), so exp2 inherits that pose error; prompt-only (exp1) is more accurate. Pitch improved slightly (46.8→51.1%). (3) `idpose_angle_hist.png` (5° bins): |yaw| peaks 40–55° both; chin-UP fine (med +21/+22°); **chin-DOWN still the weakness** (med −1°/−3°, mass straddles 0).
- **Gotcha:** relaunching `_run_idpose.sh` (10→60 IDs) without killing the first nohup caused a concurrency leak — exp2 `id_csim.csv` accumulated exp1's `__none` rows (raw kept as `id_csim_RAW_with_leak.csv`; cleaned to sam3d-only). Analysis joins by the clean 2400-row manifest so the table above is unaffected. Kill prior runs before relaunch.
## Follow-up pilots (small batch: 4 IDs [Marilyn,Oprah,Lionel,Shohei] × 5 prompts [0,6,15,21,29])
New script flags: `flux2_id_pose.py --prompt-idx` (subset), `--strip-id` (replace the prompt's identity descriptor "A black man"→"A person"). New gen `flux2_id_pose_rgb.py` (排程1: `image=[raw pose RGB, ID]`, prompt rewritten to the pose image's MEASURED angle). Rotated controls via `render_sam3d_control.py --yaw ±20 --suffix _yp20/_ym20`. Eval reuses `eval_angles_resume.py` + `eval_adaface_idref.py`. Runners `_run_small_preview.sh`, `_run_small_eval.sh`. Outputs: `idpose_small_{rgb,yp20,ym20,noidword}/`. Montages `preview_shohei.png`, `id_strip_compare.png`, `rotation_compare.png`.

### (#2) Identity word is the dominant ID killer — strip it
Same `image=[ID]` only (no pose control); the ONLY variable is the prompt's identity word. Matched-pair mean ID CSIM:

| setting | image fed | prompt | mean CSIM |
|---|---|---|---|
| keeps identity (`idpose_exp1` subset) | `[ID]` | unchanged, "A black man…" | 0.087 (≈ different person), n=19 |
| **identity STRIPPED** (`idpose_small_noidword`) | `[ID]` | "A person…" (`--strip-id`) | **0.228 (ID emerges)**, n=14 |
| 排程1 RGB-pose (`idpose_small_rgb`) | `[raw pose RGB, ID]` | rewritten to pose img's measured angle, keeps identity | 0.030 (worst), n=14 |

→ Stripping the identity word **~2.6× the CSIM** (0.087→0.228); the ID's gender/race visibly emerges (`id_strip_compare.png`). Still only ~0.23 (moderate) — FLUX.2-klein reference is a weak personalizer. **排程1 worst (0.030):** the raw RGB pose image is a *full person* whose identity competes with the ID image (two identities + prompt identity = triple conflict). (Note 0.087 here is the small matched subset; full 2400-img exp1 CSIM was 0.153.)

### (#1) Rotating the sam3d skeleton ±20° does NOT controllably re-angle the output — negative
For matched (prompt,ID), output head_yaw at sam3d −20 / orig / +20 (controls 40° apart):
- mean (+20)−(−20) output-yaw gap = **−1.9°** and **sign-inconsistent across prompts** (p00 −8°, p06 +8°, p15 −12°). Visually indistinguishable (`rotation_compare.png`).
- **FLUX largely ignores the 20° skeleton rotation.** Recommend dropping 排程2 (or test a much larger angle like ±40° before scaling).

**Decisions going forward:** (a) ID-personalization MUST strip the prompt identity word (else prompt dominates → CSIM ~0.09). (b) ±20° skeleton rotation is ineffective as a fine pose-control knob. (c) adding a pose reference (sam3d control in exp2, or raw RGB in 排程1) HURTS both yaw acc and ID vs prompt-only — the extra reference fights the prompt/ID.

### Full-scale 2×2 (60 IDs × 40 prompts, 2400/cell; seed 42 / 4 steps / 1024px, identical settings)
Factors: {ID only | sam3d+ID} × {prompt keeps identity | "A person" stripped}. Runners `_run_strip_full.sh`, `_run_exp2strip_full.sh`. Dirs `idpose_strip/` (exp1-strip), `idpose_exp2_strip/` (exp2-strip).

Consolidated per-cell table (pose acc + ID CSIM + align + angle medians; `idpose4_summary.csv` + `idpose_angle_hist`):

| cell | YAW acc | PITCH acc | CSIM(aln) | CSIM(fail0) | align% | med \|yaw\| | chin-UP pitch | chin-DOWN pitch |
|---|---|---|---|---|---|---|---|---|
| exp1 (ID,+id) | **73.7%** | 46.8% | 0.153 | 0.134 | 87.6% | 45° | +21 | −1 |
| exp1-strip (ID,strip) | 68.8% | 52.6% | **0.263** | **0.184** | 69.9% | 53° | +25 | −6 |
| exp2 (sam3d+ID,+id) | 58.5% | 51.1% | 0.122 | 0.101 | 82.5% | 49° | +22 | −3 |
| exp2-strip (sam3d+ID,strip) | 56.3% | **58.9%** | 0.211 | 0.131 | 62.3% | 54° | +26 | **−8** |

(honest CSIM fail=0: exp1 0.134, exp1-strip **0.184**, exp2 0.101, exp2-strip 0.131.)

**Two clean main effects (both consistent across the other factor):**
- **Stripping identity** (→): CSIM **+0.09–0.11** (0.153→0.263, 0.122→0.211), PITCH +6–8, but YAW −2 to −5 and align −18 to −20 (stripped IDs are varied faces that turn more → harder to MTCNN-align).
- **Adding sam3d** (↓): YAW **−12 to −15** (73.7→58.5, 68.8→56.3), CSIM −0.03 to −0.05, only PITCH +4 to +6.

**Answer to "does sam3d help on the stripped basis?": NO.** exp1-strip→exp2-strip: ID CSIM 0.263→0.211 (hurts), YAW 68.8→56.3 (hurts), only PITCH 52.6→58.9 (helps). Same trade as with-identity. **Best ID = exp1-strip (ID-only + stripped, 0.263); best YAW = exp1 (ID-only + identity, 73.7%); best PITCH = exp2-strip (58.9%).** sam3d always trades yaw+ID for pitch; the extra pose reference fights identity. The pitch gain is the one real benefit of adding sam3d (consistent with chin-down being the prompt's weak axis).

### ID CSIM vs #pose-criteria-correct (`make_csim_by_posecorrect.py`, `csim_poscorrect_fig.png`)
Per output, count pose criteria met vs prompt: **yaw direction / over-the-shoulders (only if prompt asks) / pitch** (thresholds 15/15/25). Non-OTS prompts have 2 applicable (max 2), OTS prompts 3. Group cumulatively: ≥1 / ≥2(incl 3) / =3. `=3` only comes from OTS prompts.

**Use CSIM(fail=0) — NOT aligned-only — as the metric here.** Aligned-only is **survivorship-biased**: more pose criteria correct ⇒ head turns more ⇒ MTCNN align rate falls (e.g. exp2-strip: ≥1 63% → ≥2 54% → =3 48%), so the aligned average is taken over an ever-cleaner surviving subset and *spuriously rises* with strictness. Counting un-alignable faces as 0 removes this.

CSIM(fail=0) / align% by group:
| group | exp1 (ID,+id) | exp1-strip | exp2 (sam3d,+id) | exp2-strip |
|---|---|---|---|---|
| ≥1 | 0.135 / 88% | 0.185 / 70% | 0.102 / 84% | 0.133 / 63% |
| ≥2 | 0.130 / 87% | 0.186 / 68% | 0.095 / 78% | 0.113 / 54% |
| =3 | 0.160 / 83% | 0.194 / 63% | 0.100 / 72% | 0.111 / 48% |

→ Under the honest metric the "stricter→higher CSIM" illusion **collapses**: **exp2-strip actually DECREASES with pose-correctness (0.133→0.113→0.111)** — i.e. nailing the full pose turns the head away and *costs* ID, the expected trade-off. exp1-strip is flat (~0.19), exp1/exp2(+id) only weakly up. The earlier aligned-only "=3 highest" was alignment survivorship + the cumulative ≥1 being diluted by the partial-correct majority. Lesson: any "ID vs pose" cut must use fail=0 (or report align%), because turning the head is exactly what makes the face undetectable.

Group sizes (n for ≥1 / ≥2 / =3; `=3` only from OTS prompts): exp1 2310/1509/235, exp1-strip 2326/1468/217, exp2 2273/1250/183, exp2-strip 2281/1381/157. Figure `csim_poscorrect_fig.png` (fail=0 + align%, n labelled).

### Angle distributions, all 4 cells (`idpose4_angle_hist.png`, 5° bins): |yaw|, pitch split by prompt chin up/down
(Medians folded into the consolidated table above.) → All turn 40–55° (|yaw|); **stripped+sam3d (exp2-strip) turns most (54°)** — consistent with its lowest align rate / most ID loss. Chin-up is handled well by all (+21–26°); **chin-down stays the weak axis** (median near 0) but **exp2-strip pushes it most negative (−8°)** — re-confirming the only thing adding sam3d reliably helps is the chin-down/pitch axis. `make_idpose_analysis.py --exps ...` (4 experiments), `idpose4_{angle_hist.png,summary.csv}`.

### WHY exp2 < exp1: the sam3d controls are wrong, not the mechanism (`sam3d_dir_bug.png`)
Per-prompt exp1 vs exp2 yaw (official yaw_match=dir+OTS, aggregated over 60 IDs). The sam3d control for prompt i = SAM3D skeleton of `pretrained/{i}` (the flux2 t2i image), so a bad source image ⇒ a bad control for all 60 IDs.

**sam3d HURT 15 prompts, HELPED 7, neutral 18.** The 15 hurt split into two control-quality faults:
- **A. wrong direction (5):** the source t2i image turned the *opposite* way to the prompt (all 6 wrong-dir prompts ask "right" but the source rendered "left", yaw −51…−69°). exp2 faithfully follows the flipped skeleton → flips. e.g. p20 0.97→0.17, p29 1.00→0.37, p27/p25/p5 ~0.55→~0. (p8 is wrong-dir but survived.) `sam3d_dir_bug.png` shows source(left)→skeleton(left)→exp2 out(left, red) vs exp1 out(right, green).
- **B. over-the-shoulders not encoded (10):** direction correct but **all 10 are OTS prompts** whose control/output doesn't reach |head_to_torso|>25 → OTS criterion fails. e.g. p22 0.83→0, p37 0.82→0, p31 0.80→0, p7 0.78→0.02.

**sam3d HELPED 7** (text alone failed, skeleton fixed it): p6 **0.10→1.00**, p17 0.43→0.92, p0 0.15→0.47, p39 0.08→0.38, p23 0.80→1.00, p15, p36.

**Counterfactual (the key result):** restore the 15 hurt prompts to exp1 level ⇒ **exp2 yaw acc = 80.5%, which BEATS exp1 (73.7%)**. (Fixing only the 6 wrong-direction prompts → 69.6%.) So sam3d's ceiling is *above* prompt-only; the current 58.5% is entirely **control-quality bottleneck** (source images that flip direction / under-rotate the shoulders), not a flaw in conditioning on sam3d. Fix: source the pose controls from images whose direction+OTS actually match the prompt (or filter source images by `sign(measured yaw)==prompt dir` and `|head_to_torso|>25` for OTS prompts), then rerun exp2.

**The error is the source image, not sam3d's reading (evidence).** The sam3d control is predicted FROM the flux2 t2i image; if that image is posed wrong, the skeleton is wrong — sam3d itself has no L/R flip:
1. *Image is wrong, confirmed by an estimator unrelated to sam3d:* SMPLest-X measures the 6 wrong-dir sources' absolute head_yaw as all negative (left: −65/−51/−62/−69/−53/−45°) though the prompts say "right". flux2 t2i's own yaw acc is only 67.5%.
2. *sam3d faithfully carries the source pose (no flip):* per-prompt **corr(source_yaw, exp2_output_yaw)=0.928**; **exp2 output direction follows the SOURCE 98%** but the prompt only 88%. Since the only pose input to exp2 is the sam3d skeleton, output matching source ⇒ the skeleton encodes the source's absolute direction correctly (a flip would make output track the *opposite* of source).
3. *Caveat:* the SAM3D `eval_pose_angles.py` `head_body_yaw` is head-relative-to-torso, NOT absolute camera yaw, so it is the wrong axis to cross-check skeleton direction (corr with SMPLest-X abs yaw only 0.37 — expected, different quantity). Use absolute head_yaw (= the basis of points 1–2). Data: `sam3d_angles_pretrained.csv`, `sam3d_dir_bug.png`.

### FIX — category-matched skeletons, FULL SCALE (60 IDs × 40 prompts = 2400/cell; seed 42 / 4 steps / 1024px)
The diagnosis above said the bottleneck is *control quality*, not the sam3d mechanism. Fix without filtering (keeps all 2400 pairs): classify each of the 40 source skeletons by its **measured** (direction, over-the-shoulders) category (SMPLest-X angles, `flux2_angles.json`), and for each prompt pick a **random skeleton from the matching category** (`flux2_id_pose_matched.py`, `build_matching()`, seed 42). Chin can't be matched (the 40-skeleton bank has ~no chin-down). Ran both strip / keep-identity (`_run_matched_both_full.sh` → `idpose_exp2_matched/`, `idpose_exp2_matched_strip/`).

Full 6-cell comparison (canonical `yaw_match`=dir+OTS, `pitch_match`; CSIM aligned / fail=0):

| cell | YAW acc | PITCH acc | CSIM(aln) | CSIM(f0) | align% |
|---|---|---|---|---|---|
| exp1 (ID,+id) | **73.7%** | 46.8% | 0.153 | 0.134 | 87.6% |
| exp1-strip | 68.8% | 52.6% | **0.263** | **0.184** | 69.9% |
| exp2 own-skel (+id) | 58.5% | 51.1% | 0.122 | 0.101 | 82.5% |
| exp2 own-skel strip | 56.3% | **58.9%** | 0.211 | 0.131 | 62.3% |
| **matched (+id)** | **72.0%** | 45.1% | 0.119 | 0.099 | 83.2% |
| **matched-strip** | **69.6%** | 53.0% | 0.199 | 0.134 | 67.1% |

**Pose: the fix works, at full scale.** Matched skeletons recover YAW from **58.5% → 72.0%** (+13.4, +id) and **56.3% → 69.6%** (+13.3, strip) — essentially closing the gap to prompt-only (exp1 73.7% / exp1-strip 68.8%). This confirms the counterfactual prediction (≈80% ceiling) was right in mechanism: feeding a *correctly-posed* skeleton makes exp2's yaw track the prompt instead of a flipped source. PITCH drops vs own-skel (51.1→45.1, 58.9→53.0) exactly as expected — the matched bank has no chin-down skeletons, so the one axis sam3d used to help is now un-helpable.

**ID: not recovered.** Matched CSIM(f0) = 0.099 (+id) / 0.134 (strip) — still *below* the corresponding ID-only cells (0.134 / 0.184) and essentially tied with own-skel exp2. **Fixing the pose did NOT fix identity.** So adding any sam3d reference still costs ~0.03–0.05 CSIM regardless of whether its pose is right — the loss is the *extra reference competing with the ID image for the high-freq budget*, not the control being mis-posed. This is the crux motivating the schedule work: pose is solved by control quality; ID needs a different lever (more steps / step-dependent reference weighting), not a better control.

**Net:** matched = best pose-with-control (72% yaw, ≈ prompt-only) but ID still the weak axis. Use **matched-strip** as the working setting for the schedule sweep (correct pose + stripped identity so the ID image is free to define the face).

### WHY matched-strip yaw (69.6%) looks below exp1 (73.7%) — it is NOT a direction failure
Splitting yaw_match into its two sub-criteria (direction / over-the-shoulder), on matched-strip vs exp1 (per-category, `pose_acc.csv`):

| | matched-strip | exp1 |
|---|---|---|
| **direction-only** | **94.7%** | 93.7% |
| yaw_match (dir+OTS folded) | 69.6% | 73.7% |

**On direction — the thing the skeleton controls — matched-strip (94.7%) BEATS exp1 (93.7%).** The headline gap is two things unrelated to direction alignment:
- **(A) the over-the-shoulder sub-criterion** (`|head_to_torso_yaw|>25`, required for the 20 OTS prompts). See below.
- **(B) the strip penalty:** stripping the identity word alone costs ~5 pts yaw (exp1 73.7→exp1-strip 68.8). So matched-**strip** vs exp1-**keep** is a cross-condition compare. Fair: matched-strip 69.6 **>** exp1-strip 68.8; matched-keep 72.0 ≈ exp1-keep 73.7.

### OTS ceiling: FLUX clamps head-vs-torso at ~25° (the 2D-skeleton fidelity limit)
Measured source-skeleton `|head_to_torso|` → output `|head_to_torso|` for OTS prompts (matched-strip, `*_angles.csv`):

| | src \|hbt\| | out \|hbt\| | OTS hit |
|---|---|---|---|
| left-OTS (random matched) | 37.4° | **21.3°** | 25.9% |
| right-OTS (random matched) | 30.8° | **26.1°** | 62.4% |

FLUX **systematically shrinks** the torso-relative head turn (left −16°, right −5°), landing right on the 25° threshold → OTS hit collapses, esp. left. Note exp1 (text-only) actually scores *higher* left-OTS (58%) than matched (26%): without a skeleton pinning the body, FLUX makes a freer head-only turn. The 2D skeleton can't force a stationary-torso configuration (in 2D projection an over-shoulder pose ≈ a milder turn, and the distilled model regularizes toward a frontal torso → it turns the whole body to follow the head).

**Strongest-skeleton test** (`flux2_id_pose_matched.py --ots-strongest`, `_run_otsmax_test.sh`, 200 imgs: 20 OTS prompts × 10 IDs, left→p24 src 60°, right→p12 src 39°):

| | src \|hbt\| | out \|hbt\| | OTS hit |
|---|---|---|---|
| left (random→strongest) | 37.5→**60.3°** | 20.8→**24.7°** | 22%→**42%** |
| right (random→strongest) | 30.8→38.9° | 25.1→25.0° | 58%→56% |
| **overall** | | | 40%→**49%** |

→ **Hypothesis partly confirmed but reveals a hard ceiling.** A 60° source skeleton lifts left-OTS 22%→42%, but the **output `|hbt|` saturates at ~25°**: src 37→60° (+23) yields out only 20.8→24.7° (+4) — FLUX eats ~35° of a 60° skeleton. Right doesn't improve (the bank's strongest right-OTS is only 39°, and output clamps at 25° anyway). So OTS hit hovers in a 40–60% noise band *because the model's output `|hbt|` ceiling sits exactly on the 25° criterion*. **OTS cannot be fully fixed by skeleton strength** — it's a fidelity limit of in-context 2D-skeleton conditioning; needs a stronger mechanism (explicit body-pose ControlNet / 3D-aware conditioning).

### Schedule lever #1 (more steps) is INEFFECTIVE — actually hurts (negative result)
matched-strip, fixed 10 IDs × 40 prompts, only #denoising-steps varied (`_run_step_sweep.sh`, `idpose_matched_strip_s{08,16,28}/`; 4-step = subset of the full matched-strip run):

| steps | YAW | PITCH | **CSIM(f0)** | CSIM(aln) | align% |
|---|---|---|---|---|---|
| 4 | 68.8% | 55.5% | **0.170** | 0.239 | 71.0% |
| 8 | 68.0% | 55.8% | **0.170** | 0.236 | 72.0% |
| 16 | 65.5% | 56.2% | **0.156** | 0.210 | 74.2% |
| 28 | 65.8% | 57.2% | **0.141** | 0.209 | 67.2% |

→ **More steps does NOT recover ID — CSIM falls monotonically 0.170→0.141 (4→28), and YAW drifts down too.** klein is a *distilled few-step* model; running 28 steps with default `linspace(1,1/steps,steps)` sigmas pushes it off its training distribution and ID detail degrades. **Refutes the "more steps = finer high-freq budget = better ID" hypothesis.** Schedule work must instead use *step-dependent reference weighting* (up-weight the ID reference in the late/low-sigma steps), not a longer schedule.

**Combined take-away of the three tests:** direction = solved (matched skeleton, 94.7%); OTS = capped by a ~25° model fidelity ceiling (skeleton strength only partly helps); ID = not recoverable by adding steps. Both naive "more control" (stronger skeleton) and "more compute" (more steps) hit the distilled model's ceilings → the remaining levers are a stronger pose-conditioning mechanism (for OTS) and step-dependent ID weighting (for identity).

# 2026-06-08 — Cracking OTS by skeleton SELECTION: verified-OTS controls beat exp1 (yaw 84.4%)

**This section partly OVERTURNS the "~25° hard ceiling / yaw ceiling = exp1" conclusion above.** That ceiling was a property of the *matched/exag* skeletons, not the model. Hand-picking the specific skeletons that FLUX *reads* as over-the-shoulder breaks it, and the resulting full-scale yaw (84.4%) **beats prompt-only exp1 (73.7%)** for the first time. New gen flags on `flux2_id_pose_matched.py`: `--ots-strongest`, `--ots-exag-left/right` (override OTS-prompt control by direction). Synthesis tool `sam3-body/sam-3d-body/make_exaggerated_ots.py` (rotates head joints {nose,eyes,ears} about the vertical axis through neck, keeps body pixel-exact, reprojects). All runs keep-identity / 4 steps / seed 42 unless noted.

### Metric reconciliation (resolves "sam3d 80.3% vs my 73% — am I worse?")
The 2026-06-05 cross-transfer table's **yaw dir-acc 80.3%** = pure L/R sign-match vs a *target image*, **no OTS, no threshold**. Its comparable quantity in the ID-personalization runs is **dir-only**, which for matched is **~98%** (full) — *higher* than 80.3%. The headline `yaw_match` (73%/84%) is a *stricter* metric that folds in the over-the-shoulder sub-criterion (`|head_to_torso|>25`), which the 80.3% never measured. So nothing regressed; OTS is simply an extra, harder axis. (Cross-transfer dir-acc is lower because its targets are arbitrary continuous angles incl. near-frontal; matched prompts are clean L/R.)

### The OTS bottleneck is LEFT/RIGHT-asymmetric and skeleton-SELECTION-solvable (200-pair sweeps, 40 prompts × 5 IDs, keep-id)
Per-skeleton OTS-drive (from full matched run): output `|hbt|` and OTS-hit are **non-monotonic in source angle** — the skeleton's *2D readability* as OTS matters, not its magnitude. Best drivers: **left p24** (lying-down, out 35°/88% hit) and **right p15** (upright, src only 29° but out 33°/97% hit). Five left-OTS strategies tried:

| left-OTS control | left-OTS dir | left-OTS yaw_match | note |
|---|---|---|---|
| matched random | ~99% | ~26% | under-drives OTS |
| exaggerated (head→−101°, past profile) | 94% | 28% | over-push backfires |
| mirror of p15 (upright) | 96% | 34% | flip breaks OTS read |
| exag-p08 (upright, head→−78°) | 94% | 28% | upright left won't drive OTS |
| **p24 (lying-down)** | 66% | **56%** | only thing that drives left-OTS |

**Key asymmetry:** upright skeletons drive RIGHT-OTS fine (p15 → 80% yaw_match) but NOT LEFT-OTS (mirror/exag-p08 → 28–34%). The *only* control that forces left-OTS is the lying-down p24 — horizontal body makes head-vs-torso unambiguous in 2D — but lying-down costs direction reliability (66%). So **FLUX.2-klein has a systematic bias: it renders right-over-shoulder readily, resists left-over-shoulder** (consistent with random/strongest data throughout: left-OTS always ≪ right-OTS). Exaggerating head rotation does NOT help (and past-profile backfires); mirroring an upright skeleton fixes direction but kills the OTS read.

### 200-pair comparison (same 5 IDs, keep-id) — verified config wins
| config (OTS control) | YAW_match | dir-only | OTS | PITCH |
|---|---|---|---|---|
| exp1 (prompt-only) | 76.0% | 97.5% | 57.0% | 48.0% |
| matched | 73.0% | 98.5% | 48.0% | 47.5% |
| exag (left −101 / right +80) | 70.5% | 91.0% | 43.0% | 45.5% |
| mirror-p15 / p15 | 77.5% | 98.0% | 57.0% | 48.0% |
| exag-p08 / p15 | 76.0% | 97.5% | 54.0% | 46.5% |
| **verified (p24 / p15)** | **83.0%** | 90.5% | **71.0%** | **51.5%** |

### FULL SCALE — `idpose_otsverified_full/` (2400; left-OTS→p24, right-OTS→p15, non-OTS→matched; `_run_otsverified_full.sh`)
| cell | YAW_match | dir-only | OTS hit | PITCH | CSIM(aln) | CSIM(f0) | align% |
|---|---|---|---|---|---|---|---|
| exp1 (ID,+id) | 73.7% | ~94% | — | 46.8% | 0.153 | 0.134 | 87.6% |
| matched (+id) | 72.0% | ~98% | 48% | 45.1% | 0.119 | 0.099 | 83.2% |
| **verified-OTS (+id)** | **84.4%** | 90.2% | **72.9%** | **50.8%** | 0.111 | 0.093 | 84.0% |

Per-category yaw_match: left **100%**, left-OTS 58.0%, right 94.2%, right-OTS **85.3%**. → **verified-OTS yaw 84.4% beats exp1 73.7% by +10.7 and matched 72.0% by +12.4**, with best PITCH too. The only remaining yaw weakness is left-OTS (58%, the model asymmetry). **Revised conclusion: in-context 2D-skeleton CAN beat prompt-only on yaw — but only by curating controls FLUX reads as OTS, and left-OTS stays model-capped.** CSIM still ~0.09 (keep-identity → prompt word dominates; this is the ID-enhance target, next section).

# 2026-06-08 — ID-enhance: duplicating the ID reference is a tunable CSIM↔pose knob

Goal: raise ID CSIM (the weak axis: keep-id ~0.09–0.14 f0) **without dropping pose much**. Holding pose with the verified-OTS controls (left p24 / right p15) so yaw stays high, sweep ID-side levers in **100-img batches** (20 balanced prompts × 5 IDs), always reporting CSIM **and** yaw/pitch. New `flux2_id_pose_matched.py` flags: `--dup-id N` (feed ID ref N×), `--dup-ctrl N`, `--id-first`, `--id-dir` (e.g. face-crop). Helpers `make_id_facecrop.py` (mediapipe → `ID_image_facecrop/`), `normalize_idfile.py` (CSIM always vs original ID), `idenh_report.py`. Runners `_run_idenh_round{1,2,3}.sh`. CSIM = AdaFace fail=0.

Consolidated (100 imgs each; B0 = subset of the keep-id full-2400 for the same prompts/IDs):

| config | CSIM(aln) | CSIM(f0) | align% | YAW | PITCH |
|---|---|---|---|---|---|
| B0 keep-id baseline | 0.155 | 0.138 | 89% | **85.0%** | 50.0% |
| strip | 0.229 | 0.154 | 67% | 78.0% | 66.0% |
| strip + facecrop | 0.239 | 0.153 | 64% | 77.0% | 66.0% |
| strip + id-first | 0.268 | 0.180 | 67% | 78.0% | 66.0% |
| strip + dup-id ×2 | 0.298 | 0.209 | 70% | 80.0% | 61.0% |
| **strip + dup-id ×3** | 0.348 | **0.282** | 81% | 76.0% | 57.0% |
| strip + dup-id ×4 | 0.371 | 0.312 | 84% | **62.0%** | 42.0% |
| strip + dup-id ×2 + id-first | 0.368 | 0.272 | 74% | 77.0% | 59.0% |
| strip + dup-id ×2 + facecrop | 0.279 | 0.187 | 67% | 78.0% | 61.0% |
| **keep-id + dup-id ×3** | 0.284 | 0.253 | 89% | 74.0% | 41.0% |

**Findings.** (1) **Duplicating the ID reference token-block is the dominant ID lever** — CSIM(f0) rises monotonically with dup count. Each reference is a ~1024-token block (VAE-encoded, concatenated to the latents every step, `pipeline_flux2_klein` `torch.cat([latents, image_latents],1)`); duplicating gives the ID more attention budget. (2) **It's a CSIM↔pose Pareto knob** (more dup ⇒ more CSIM, less pose). (3) **id-first ordering is the key pose-retention lever** — putting the ID block *before* the control rescues pose AND lifts CSIM (e.g. strip dup4: 0.312/62% → +id-first 0.385/74%). (4) **dup overrides the identity word — stripping is NOT needed** and is in fact *worse* than keep-id+id-first (strip costs ~5–7 pose pts and align rate). (5) **`--dup-ctrl` REJECTED** (boosting control tokens dilutes ID without recovering pose). **facecrop** neutral-to-negative. Qualitative `idenh_qualitative.png` confirms real identity emergence (keep-id baseline often renders the *prompt's* identity, e.g. a child / wrong race; dup-id recovers the reference face).

**Complete Pareto (100 imgs each; CSIM fail=0):**
| config | CSIM(f0) | align% | YAW | PITCH |
|---|---|---|---|---|
| keep-id baseline (B0) | 0.138 | 89% | 85.0% | 50.0% |
| strip | 0.154 | 67% | 78.0% | 66.0% |
| strip+dup3 | 0.282 | 81% | 76.0% | 57.0% |
| strip+dup3+idfirst | 0.288 | 75% | 80.0% | 51.0% |
| strip+dup4+idfirst | 0.385 | 87% | 74.0% | 27.0% |
| strip+dup5+idfirst | 0.462 | 94% | 59.0% | 41.0% |
| keepid+dup2 | 0.199 | 88% | 82.0% | 45.0% |
| **keepid+dup3+idfirst** ⭐ | **0.327** | 92% | **85.0%** | 39.0% |
| keepid+dup4+idfirst | 0.352 | 89% | 76.0% | 18.0% |

**WINNER = `keep-id + dup-id ×3 + id-first`: CSIM 0.327 (2.4× the 0.138 keep-id baseline) at YAW 85.0% — ZERO pose loss, no stripping.** id-first ordering on the keep-identity path is the sweet spot: the ID block (placed first, ×3) dominates identity while the pose control is undisturbed (pose = baseline 85%). The only cost is PITCH (50→39; over-weighting ID flattens the chin — chin was always the weak axis). For more ID at a pose cost: `strip+dup4+idfirst` 0.385/74%, or `strip+dup5+idfirst` 0.462/59%. **Gen flags: `--dup-id 3 --id-first` (keep identity word). Next: validate the winner at full 2400 scale; the chin/PITCH drop is the remaining open axis.**

**Generalization (fresh 5 IDs `08851,10771,11689,18327,19511`, `_run_idenh_gen.sh`):** keepid+dup3+idfirst = **0.324 f0 / YAW 87%** (vs 0.327/85% on the original 5; baseline 0.128/86%); strip+dup4+idfirst = 0.426/74% (vs 0.385/74%). **The winner is robust across ID sets — ~2.5× CSIM at zero pose loss.**

### FULL-SCALE VALIDATION of the winner — `idpose_winner_full/` (2400; keepid+dup3+idfirst, verified-OTS controls; `_run_winner_full.sh`)
| run (2400) | YAW | dir-only | OTS | PITCH | CSIM(aln) | CSIM(f0) | align% |
|---|---|---|---|---|---|---|---|
| exp1 (prompt-only ID) | 73.7% | ~94% | — | 46.8% | 0.153 | 0.134 | 87.6% |
| verified-OTS (no dup) | **84.4%** | 90.2% | 72.9% | **50.8%** | 0.111 | 0.093 | 84.0% |
| **WINNER keepid+dup3+idfirst** | 82.2% | 91.9% | 69.8% | 40.2% | **0.287** | **0.252** | 87.8% |

Per-category yaw: left 100%, left-OTS 52%, right 98%, right-OTS 78%. → **At full scale the winner holds: CSIM(f0) 0.252 = 2.7× the verified-OTS keep-id baseline (0.093) and ~1.9× exp1 (0.134), while YAW 82.2% still beats exp1 by +8.5** (dup-id ×3 costs only −2.2 yaw vs the no-dup verified config). The 100-img estimate (0.327/85%) was slightly optimistic on 5 IDs; the honest full-2400 number is **0.252 CSIM at 82.2% YAW, keep-identity (no stripping)**. PITCH is the one regression (50.8→40.2 — over-weighting the frontal ID flattens the chin; chin was always the weak axis). **Bottom line: the verified-OTS pose curation (yaw 84.4% > exp1) + `--dup-id 3 --id-first` ID up-weighting (CSIM 2.7×) together give the best joint pose+ID operating point found.** Gen: `flux2_id_pose_matched.py --ids all --dup-id 3 --id-first --ots-exag-left <p24> --ots-exag-right <p15>` (keep identity word).

### PITCH proof-of-concept: chin-rotated controls — chin-DOWN is a model limit (mostly negative)
Can encoding chin in the control recover PITCH (the winner's regression)? Synthesised chin-up/down sam3d controls (`sam3-body/sam-3d-body/make_pitch_control.py` — rotates head joints {nose,eyes,ears} about the LEFT-RIGHT axis through the neck, body pixel-exact, auto-signs by nose-elevation), used via `flux2_id_pose_matched.py --chin-pitch` (loads `{control}_sam3d_{chinup,chindown}.png` for chin prompts). 100 imgs (winner config + `--chin-pitch`), vs the winner-full subset (same 5 IDs × 20 prompts):

| config | YAW | PITCH | chin-UP acc | chin-DOWN acc | median pitch up / down |
|---|---|---|---|---|---|
| winner (no chin-pitch) | 85% | 39% | 88% | **7%** | +18.8° / **+1.9°** |
| + chin-pitch controls | 86% | 43% | **95%** | 8% | +18.5° / **+0.7°** |

→ **Chin-rotated controls lift chin-UP a little (88→95%, overall PITCH 39→43%) but do NOT fix chin-DOWN (7→8%): output median pitch stays ~0° when it should be < −15°.** FLUX won't render a chin-down head even when the 2D skeleton clearly encodes it — the **same model-resistance pattern as left-OTS** (and as every T2I in the 5-model comparison, where chin-down was the universal failure). **Chin-down is a fundamental FLUX.2-klein limit, not a control-quality problem** — the skeleton's chin-down signal isn't read, just as left-over-shoulder isn't. CSIM unaffected (0.327→0.336). Net: PITCH is capped; the winner's PITCH drop is partly recoverable on chin-UP but chin-DOWN is out of reach via in-context 2D skeletons.

### New tooling this session (ID-enhance + pitch)
`flux2_id_pose_matched.py` flags: `--dup-id N`, `--dup-ctrl N` (rejected), `--id-first`, `--id-dir`, `--chin-pitch`, plus the earlier `--ots-strongest`, `--ots-exag-left/right`. Synthesis: `make_exaggerated_ots.py` (head yaw rotation), `make_pitch_control.py` (head pitch rotation), `make_id_facecrop.py` (mediapipe face crops, rejected). Eval helpers: `normalize_idfile.py`, `idenh_report.py`. Runners: `_run_idenh_round{1..5}.sh`, `_run_idenh_gen.sh`, `_run_winner_full.sh`, `_run_pitch_poc.sh`, `_run_otsverified_full.sh`, `_run_step_sweep.sh`.

# 2026-06-09 — Minimal learnable ID-token module (AdaFace→projector→K tokens→concat): NEGATIVE

**Goal (narrow, deliberately minimal).** Test whether projecting a frozen-AdaFace identity feature
into **K learnable tokens, concatenated to the MM-DiT visual-token sequence**, improves identity
preservation — with **nothing else added** (no LoRA, no step-dependent attention bias, no attention
routing, no pose loss, no sigma-schedule change). Only a small projector trains; FLUX.2-klein +
AdaFace are frozen. The minimal member of the IP-Adapter/PhotoMaker family. Design in
`LEARNABLE_ID_TOKEN_design.md`.

### Architecture (confirmed conceptually sound)
```
ID image →(adaface, frozen) AdaFace ir_101 → id_emb[B,512] →(TRAINABLE) IDTokenProjector
  (LN→Linear512→1024→GELU→Linear→K·128) → id_tokens[B,K,128]  →  appended as one more
  reference block to image_latents → frozen x_embedder(128→3072) → joint attention in MM-DiT.
```
Tokens enter in the **packed reference space `in_channels=128`** (same path as VAE reference
images), so **no transformer surgery** — only `image_latents`/`image_latent_ids` are extended
(`append_id_tokens`, RoPE 4-D `(T,H,W,L)` with the next free T-slot `T=10·(n_refs+1)`, `W=0..K-1`).
At inference the injection is a monkeypatch on `pipe.prepare_image_latents` so the entire real
`__call__` (scheduler/mu/decode) is reused unchanged.

### Pipeline (env-split, 3 stages) + scripts created this session
| Stage | Env | Script | Output |
|---|---|---|---|
| A. ID embeddings | `adaface` | `prep_id_embeds.py` | `id_cache_train/` (2000 face-detectable LAION embeds [2000,512] + `index.csv`), `id_cache_eval/` (60 ID embeds + `names.txt`) |
| B. Train projector | `softrepa` | `train_id_tokens.py` (+ `id_token_module.py`) | `id_projector.pt` — frozen FLUX, custom flow-matching loop, **grad-checkpointed** (backprop reaches the projector *through* the frozen 4B transformer) |
| C. Eval | `softrepa`+`smplestx`+`adaface` | `flux2_id_tokens_gen.py` (`--use-id-tokens`), reuses `eval_angles_resume.py`/`make_idpose_poseacc.py`/`eval_adaface_idref.py`/`idenh_report.py` | 4-arm CSIM+pose |
Runners `_run_idtok_train.sh` (waits for the GPU window → 5-step smoke → 4000-step train) +
`_run_idtok_genC.sh`. Training data: `Gemma/laion_gender_age_race_long_captioned_extension_prompt25_race_fixed_viewer.csv`
(2000 `sam3d_status==OK` + face-detectable rows; ID = the target image's **own** face so the only
face signal is the AdaFace token; skeleton = bank-matched by `long_caption` pose). 4000 steps,
1024px, K=8, lr 1e-4.

### Result — 100 imgs/arm (20 balanced prompts × 5 IDs, verified-OTS controls, seed 42, 4 steps)
Baseline here = **skeleton + prompt only, NO ID image** (the true no-ID-conditioning floor; this is
*not* the prior exp1 baseline, which fed the ID image as a VAE block → ~0.13). The only variable is
whether K=8 projected AdaFace tokens are appended.

| arm | n | CSIM(aln) | CSIM(f0) | align% | YAW | PITCH |
|---|---|---|---|---|---|---|
| baseline keep-id | 100 | 0.0152 | 0.0137 | 90% | **85.0%** | 55.0% |
| **+ID-tokens keep-id** | 100 | 0.0120 | 0.0120 | 100% | **85.0%** | 55.0% |
| baseline strip | 100 | 0.0056 | 0.0033 | 60% | 70.0% | 70.0% |
| **+ID-tokens strip** | 100 | 0.0068 | 0.0042 | 61% | 72.0% | 67.0% |

**Verdict: NEGATIVE — the minimal token injection does NOT transfer identity.** CSIM stays at the
no-conditioning floor: keep 0.0137→0.0120 (slightly *down*), strip 0.0033→0.0042 (negligible, within
noise). Pose is untouched (YAW 85% keep, identical to baseline — the tokens don't fight the skeleton).

**Why it failed (mechanism, evidenced):**
1. **Training loss never decreased** — flat ~0.67–0.76 from step 50 to 4000. Among ~8200 visual
   tokens (4096 gen + 4096 skeleton + 8 ID), the 8 ID tokens give the projector negligible leverage
   to reduce the flow-matching reconstruction loss against a frozen backbone → the projector receives
   almost no gradient signal → learns nothing useful.
2. **The injection path is correct and live** (not a wiring bug): +ID-tokens outputs differ from
   baseline by mean |Δpixel| = 1.9/255 — the tokens *do* enter the MM-DiT and perturb the image, but
   the perturbation is **identity-neutral**. The frozen `x_embedder` (trained on VAE-latent stats)
   maps the projector's 128-d vectors to embeddings the attention largely ignores.

**Contrast with what works.** `dup-id` (full 2400: CSIM 0.252) feeds a **full VAE-encoded ID image
block** the model already knows how to attend to; that's a ~1024-token reference, not 8 abstract
tokens, and needs no training. The minimal AdaFace-token route is the cleaner *idea* (identity-pure,
pose-invariant, same space as the CSIM metric) but is too weakly coupled to a frozen distilled
backbone to learn anything from MSE alone.

**Implication for next steps (all beyond the deliberately-minimal scope of this run):** the token
needs stronger coupling and/or a stronger signal — e.g. inject post-`x_embedder` at d_model=3072
(more expressive, needs transformer hooks), many more tokens, an **identity-aligned auxiliary loss**
(decode prediction → AdaFace CSIM term) so the projector gets direct identity gradient instead of
diluted reconstruction gradient, or a cross-attention IP-Adapter rather than soft-prompt concat.
For now, `dup-id 3 + id-first` (CSIM 0.252, no training) remains the best ID lever; this experiment
cleanly rules out the *minimal* projector+concat variant.

## 2026-06-09 (follow-up) — Identity-aligned auxiliary loss ABLATION: gradient flows, CSIM still doesn't move (NEGATIVE)

**One hypothesis, one variable.** The section above diagnosed the null as "the projector gets no
useful gradient from MSE." This ablation adds a **direct identity gradient** and asks only: does CSIM
*start to move*? Everything else is held identical to the null — concat injection unchanged, backbone
frozen, skeleton kept, K=8, no cross-attention, no sigma-schedule change. **Only** `total = mse +
λ·id_loss` is added, where `id_loss = 1 − cos(AdaFace(decode(x0_pred)), id_emb_target)`, gated to
near-clean timesteps (σ<0.3) so the one-step `x0_pred` is sharp enough for AdaFace. 200 IDs, 512px.

**The crux — keeping the id-loss differentiable to the projector.** The whole chain
`proj → transformer → v_pred → x0_pred(=noisy−σ·v_pred) → vae.decode → align → AdaFace → CSIM` must
retain grad, or the projector gets zero ID signal (= the null again). The standard AdaFace alignment
(`align.get_aligned_face`, MTCNN on a file path → PIL crop) is **non-differentiable** and would detach
the graph. **Solution:** precompute each training image's face-alignment transform **offline** (env
`adaface`, which has cv2) and at train time apply it to the *decoded prediction* with a **pure-torch
`grid_sample` warp** (env `softrepa` has no kornia/cv2). The transform is the EXACT cv2 4-DOF
*similarity* `tfm` (a 6-DOF affine lstsq fit does NOT match it — caught by validation), converted to the
`F.affine_grid` convention `theta = normIn⁻¹·inv([tfm;0,0,1])·normOut`; validated vs cv2 warp to
mean-pixel-diff **0.21** → 200/200 cached thetas. (Tooling: `prep_id_embeds.py --save-align`
→ `id_cache_train200/align_theta.npy`; `train_id_tokens.py --id-loss --lambda-id --id-sigma-max`;
runners `_run_idtok_idloss.sh`, `_run_idtok_idloss_genC.sh`.)

**Self-controlled run: λ=0 (MSE only) vs λ=0.5, identical data/skeleton/seed.**

| signal | λ=0 (control) | λ=0.5 |
|---|---|---|
| **in-loop val-CSIM** (held-out 16, fixed σ=0.2) | 0.451 → 0.451 (flat) | 0.451 → 0.451 (flat) |
| projector **grad-norm** (step 250 → 2000) | 0.011 → 0.139 | 0.006 → **0.21 @1750** → 0.177 |
| train `id_loss` (diluted by fire-rate ~0.2) | — | 0.10 → 0.09 (noisy, no clear trend) |

**Decisive end-to-end generation CSIM** (full denoise from noise, identity ONLY from the tokens — no
target leakage; 100 imgs = 20 prompts × 5 eval IDs, verified-OTS controls, 1024px):

| arm | CSIM(aln) | CSIM(f0) | align% |
|---|---|---|---|
| baseline (no token) | 0.0152 | 0.0137 | 90% |
| id-loss **λ=0** projector | 0.0156 | 0.0147 | 94% |
| id-loss **λ=0.5** projector | 0.0150 | 0.0143 | 95% |

**Verdict: NEGATIVE — and it moves the diagnosis.** The auxiliary loss **does** deliver a direct
identity gradient (λ=0.5 grad-norm exceeds λ=0 throughout training — gradient genuinely reaches the
projector through decode→AdaFace, the make-or-break the prior section lacked). **Yet CSIM does not
move:** the in-loop val proxy is flat (0.451 both), and decisively the end-to-end generation CSIM is
**0.0143 (λ=0.5) ≈ 0.0147 (λ=0) ≈ 0.0137 (baseline)** — all indistinguishable at the no-conditioning
floor. So the binding constraint is **NOT** the absence of an identity gradient (the prior hypothesis);
it is the **injection mechanism + frozen guidance-distilled backbone**: 8 abstract tokens concatenated
through the frozen `x_embedder` cannot steer identity even when trained against a direct AdaFace signal.
(The tiny apparent train-`id_loss` wiggle on the *training* faces' one-step decode did not generalize —
flat on held-out val, floor at generation.)

**Caveats.** (i) The σ=0.2 val proxy is contaminated (`x0_pred ≈ 0.8·x0 + …` leaks the true target),
so its flatness is weak evidence on its own — the end-to-end generation CSIM is the load-bearing
result. (ii) Pose was not evaluated for the two id-loss arms (no `pose_acc.csv`); id-tokens don't carry
pose, so this is immaterial to the verdict.

**Implication.** Both levers on the *training signal* are now exhausted (MSE → null; +direct identity
loss → still null). The next experiment must change the **injection**, not the loss: post-`x_embedder`
injection at d_model=3072, many more tokens, or a cross-attention IP-Adapter — i.e. give the identity
condition a pathway the frozen MM-DiT actually attends to, rather than a soft-prompt concat it ignores.

## 2026-06-09 (follow-up 2) — Cross-attention IP-Adapter injection: mechanism works, doesn't generalize at ablation scale (NEGATIVE)

Changes the injection from concat to a **decoupled cross-attention IP-Adapter** on FLUX.2-klein's MM-DiT
(`ip_adapter_flux2.py`): every attention layer (5 double + 20 single = 25) gets a parallel image-prompt
attention `attn_out += ip_scale·SDPA(Q_joint, Wk_ip·id, Wv_ip·id)`; id tokens come from the projector at
d_model=3072 and are routed via `attention_kwargs→joint_attention_kwargs` (FLUX threads them to every
processor; non-reentrant checkpointing keeps grad). Trainable: per-layer `Wk_ip/Wv_ip/norm_k_ip` (fp32)
+ projector = **471.9M + 25.7M**; backbone frozen. New: `--inject ip` in `train_id_tokens.py`,
`flux2_ip_gen.py` (inference via `attention_kwargs`), TensorBoard logging (`train/id_csim`=1−id_loss =
decoded-image ID CSIM, `val/csim`, mse, grad-norm). 1000 IDs (`id_cache_train1000`), 512px.

**Mechanism is verified live** (the precondition concat lacked): install smoke gives output that responds
strongly to `ip_scale` and **IP grad-norm 1.12 / projector 0.80** — gradient reaches the cross-attn K/V
through the frozen checkpointed backbone (≫ the ≤0.05 of any concat run).

### Run A — IP + MSE-only (canonical IP-Adapter training, 3000 steps)
| arm | CSIM(f0) |
|---|---|
| baseline (no token) | 0.0137 |
| ip_scale 0.5 | 0.0158 |
| ip_scale 1.0 | 0.0109 |
| ip_scale 1.5 | **−0.0124** (destructive) |

**Training MSE never moved** (flat ~0.51–0.58/3000 steps); val-CSIM flat (0.437). Root cause = the
**objective, not the mechanism**: reconstructing the target whose own latents already carry its identity
makes the IP condition *redundant* for MSE → no pressure to use it (same failure as concat-MSE). CSIM
goes *negative* at scale 1.5 = the (undertrained, near-random) IP perturbs destructively, not toward ID.

### Run B — IP + id_loss (the synthesis: capable mechanism + a signal that demands identity, λ=0.5, σ<0.3, 2000 steps)
| signal | result |
|---|---|
| `train/id_csim` (training faces, decoded one-step) | **rises 0.23 → ~0.5–0.7** ✅ optimizes |
| `val/csim` (held-out, σ=0.2 proxy) | **flat 0.482 → 0.481** ❌ |
| end-to-end CSIM (held-out **eval IDs**, ip_scale 1.0 / 1.5) | **0.0013 / 0.0038** ❌ (≤ baseline 0.0137) |

**Verdict: optimizes the training objective but does NOT generalize — it overfits.** The id_loss fires at
σ<0.3 where the decoded prediction *already contains the target* (latent leakage `x0_pred≈0.7·x0+…`); the
IP learns to nudge that already-present face — a per-sample correction **exploiting leakage**, not learning
to use the AdaFace condition. At inference (full denoise from noise, no leakage) it has nothing
generalizable and even degrades faces (CSIM≈0). 471.9M IP params on **1000 IDs / 2000 steps** is also
vastly under the IP-Adapter data regime (millions of images), so memorization dominates.

### Net across the 4-experiment arc (concat×{MSE,id_loss}, IP×{MSE,id_loss})
The cross-attn **mechanism is capable** (grad flows strongly; Run-B train objective provably optimizes —
the first config to do so), but **no config achieved generalizable ID injection** into the frozen
guidance-distilled FLUX.2-klein at this ablation scale. Concat can't inject at all (grad ~0 or
identity-neutral); IP can inject but, with small data + a low-σ id_loss that's gameable via target
leakage, it overfits. **Genuine paths forward (beyond ablation scope):** (a) the real IP-Adapter data
regime (≫1000 IDs); (b) an id_loss that can't exploit leakage — compute it at **high σ / on full
generations** (identity must come only from the condition); (c) drop/cropped reference so the condition
is non-redundant. `dup-id 3 + id-first` (CSIM 0.252, no training) remains the only working ID lever so far.

### Run C — IP + id_loss, QUALITY-preserving (gated IP + margin-clamp + TB image monitor)
Motivated by "id_csim up but image quality DOWN" — id_loss is a recognition-cosine objective that gets
*hacked* into high-freq artifacts. Three fixes: ① **gated IP** (learnable per-layer scalar gate, init 0.1
so the base model's quality is preserved at start and IP adds gently — exact-0 dead-locks the K/V grad);
③ **margin-clamped id_loss** `relu(0.6−cos)` (stop pushing once decoded CSIM hits 0.6 → no over-optimising
into artifacts); ⑤ **TensorBoard decoded-image logging** (`val/decoded`) to watch quality vs identity live.

| signal | result |
|---|---|
| `train/ip_gate_mean` | **0.10 → 0.09 (gate CLOSES)** — the model actively suppresses the IP |
| `val/csim` (held-out) | flat 0.482 |
| end-to-end CSIM (eval IDs, ip_scale 0.5 / 1.0) | **0.0131 / 0.0002** (≈ baseline 0.0137 / ~0) |

→ The quality fix works (quality preserved) **but the gate closing is the smoking gun:** given no
generalizable identity signal, the model's optimum is to *turn the IP off* → ID is never transferred.
Quality is preserved precisely by **not injecting**. (Visually confirmed by the user: the reference ID is
not applied to the outputs at all.)

### ROOT CAUSE (definitive, across all 5 configs) — the training OBJECTIVE never forces use of the condition
Every run trains by **reconstructing the target image whose identity is already in the noised latents**, so
the ID condition is **redundant** — the model can minimise the loss without it, and (Run C proves) actively
**learns to ignore/suppress it** (gate 0.10→0.09). This is **orthogonal to the injection mechanism** (concat
vs cross-attn) **and to quality tweaks** (gate/clamp): all five configs fail to transfer ID at inference
(held-out generation CSIM ≈ 0.001–0.016, never above the prompt-only floor). **The σ<0.3 `id_csim` "rising
to 0.5–0.7" was a misleading proxy** — it measured the *target leaking through* on training faces, not real
ID transfer; only the held-out end-to-end generation CSIM is trustworthy.

**To actually transfer ID, the objective must make the condition the ONLY identity source:**
(a) **full-generation id_loss** — periodically sample from pure noise with the IP condition and AdaFace-CSIM
the *final* image (backprop through sampling); the only way to win is to use the condition; (b) much more
data + the standard adapter regime (≫1000 IDs, GPU-days, à la InstantID/PuLID); (c) reference ≠ target view.
Until then, **training-free `dup-id` (keepid+dup3+idfirst, CSIM 0.252, ~2.7× baseline) is the ONLY lever that
visibly swaps identity** in this codebase (`idenh_qualitative.png`).

### Tooling this arc
`ip_adapter_flux2.py` (Flux2IPAttnProcessor / Flux2IPParallelProcessor with learnable `gate` +
`install_ip_adapter(gate_init)`/`ip_state_dict`/`load_ip`), `train_id_tokens.py --inject ip --ip-scale
--gate-init --id-margin --img-every` + TensorBoard (`runs/`: `train/id_csim,ip_gate_mean`, `val/csim`,
`val/decoded` images), `flux2_ip_gen.py`, runners `_run_idtok_ip.sh`, `_run_idtok_ip_idloss.sh`,
`_run_idtok_ip_quality.sh`. (Installed `tensorboard` into env `softrepa`.)

## 2026-06-10 — Full-generation id_loss (the leakage-free objective): correct in principle, still fails at ablation scale (NEGATIVE)

The fix the prior diagnosis demanded: instead of a leakage-prone one-step x0 id_loss, **generate from pure
noise through a DIFFERENTIABLE K-step Euler sampling with the IP condition, decode the FINAL image,
MTCNN-detect+align (in-process), AdaFace-CSIM vs target** (`fullgen_id_loss.py`). From noise there is no
target leakage, so the only way to lower the loss is to USE the condition. `--id-loss-mode fullgen` in the
trainer; the val + `train/id_csim` are now the **real generation CSIM** (val on truly held-out IDs excluded
from training), TB logs the actual generated images (`val/generated`). `mse-weight 0` (the MSE-reconstruct
anchor wants the IP OFF — it was what closed the gate — so it is dropped). Config: gen-steps 4, 512px, gate
0.1, λ 1.0, 1000 IDs, 2000 steps. **Standalone-verified the make-or-break:** face detected in the generated
image, **gradient flows strongly through the whole sampling (IP grad 2.66, proj 1.86)**, ~2 s/step, 23 GB.

| metric (2000 steps) | result |
|---|---|
| `val/csim` held-out **real generation** CSIM | bounces −0.08…+0.07, **mean ≈ 0, no trend** |
| `train/id_csim` (real gen) | noisy ~0.01–0.15, no rise; gate **0.10 → 0.096 (drifts down)** |
| Stage-C eval IDs s1.0 / s1.5 | CSIM 0.0038 / 0.0000, **align% 22% / 7%** (vs 90% baseline) |

**Verdict: NEGATIVE — and the align% collapse is the new evidence.** The objective is now *correct* (no
leakage; gradient proven to flow), yet on this budget the 471M IP K/V **cannot find a generalizable identity
direction** (held-out CSIM flat ≈ 0), and the strong-but-noisy per-sample full-gen gradient mostly
**destroys the face** — only 22% (s1.0) / 7% (s1.5) of generated images still have a *detectable* face (vs
90%). The gate even drifts *down*: the model "wants less IP" because the injection is net-harmful.

**Two compounding walls (the real conclusion of the whole investigation):**
1. **Data/compute.** Learning face-ID injection from scratch is data-hungry — InstantID/PuLID use *millions*
   of face images + GPU-days. 1000 IDs / 2000 steps / ~30 min is 3–4 orders of magnitude short; the per-step
   signal (one random ID/skeleton/seed) is too noisy for 471M params to converge.
2. **The distilled backbone is fragile.** FLUX.2-klein is a guidance-distilled *4-step* model with a narrow
   generation manifold; strong external IP injection knocks generation off-manifold → **faces stop forming**
   (the align% collapse). A non-distilled / many-step base would tolerate injection far better.

### FINAL conclusion (concat × {MSE, id_loss}; IP × {MSE, id_loss(onestep), id_loss(fullgen), gated+clamped})
**No learnable from-scratch approach transferred identity** into the frozen distilled FLUX.2-klein at
ablation scale — across both injection mechanisms and the full ladder of training signals up to the
theoretically-correct full-generation id_loss. The mechanism (cross-attn) and the objective (full-gen) are
each correct in principle; the binding constraints are **data/compute scale + the fragile distilled
manifold**, not a fixable detail. **The only lever that visibly swaps identity remains training-free
`dup-id` (keepid+dup3+idfirst, CSIM 0.252).** A real learnable adapter here would need: a much larger face
set (full LAION-10k+ / a face dataset), many more steps + LR schedule, likely a non-distilled base or a
pretrained face-adapter as init — a serious training run, not an ablation. New tooling: `fullgen_id_loss.py`,
`train_id_tokens.py --id-loss-mode fullgen --gen-steps --mse-weight`, `_run_idtok_fullgen.sh`. (Installed
`opencv-python-headless`==4.10 + pinned `numpy`==1.26.3 in env `softrepa` for in-process MTCNN.)

## 2026-06-10 (overnight) — Paper-inspired injections (EasyControl, XVerse) + ID↔quality Pareto

User directive: read `docs/paper/` for directions, run experiments to lift ID CSIM without hurting quality
or overfitting; and if ID is decent, validate pose + image quality. Studied **UniCTokens** (per-concept
test-time tuning — not zero-shot, N/A), **EasyControl** (ICCV25: Condition-Injection-LoRA — reuse frozen
W_k/W_v + rank-r LoRA, concat cond K/V into the *single softmax* → bounded/gentle, ~15M params), and
**XVerse** (NeurIPS25, the sharpest: *"injecting control through attention deviates the sampling
trajectory and reduces quality"* = our align-collapse; their fix = inject via **AdaLN text-stream
modulation**, with VAE features only as auxiliary detail). Every strong method uses **rich VAE/CLIP image
features, never a 512-d vector** — which is why our AdaFace-token track is weak and training-free dup-id (VAE
image ref) works.

### Three new injection mechanisms tried (all gentle/low-param; full-gen id_loss; 1000 IDs / 512px)
| inject | trainable | quality (gen face-detect) | held-out gen CSIM | verdict |
|---|---|---|---|---|
| **EasyControl-LoRA** + AdaFace token | 4.9M LoRA + 6.4M proj | preserved (fire ~0.8) | **flat ~0.0** | AdaFace token too weak |
| **EasyControl-LoRA** + VAE-image cond ("learnable dup-id") | 4.9M LoRA only | preserved (fire ~0.8) | **flat ~0.01** | LoRA can't amplify the gentle concat |
| **XVerse modulation** (temb += MLP(AdaFace), AdaLN) lr5e-4 | 1.8M MLP | **collapses (fire 0.20→0.04)** | flat | over-push drives AdaLN off-manifold |
| **XVerse modulation** lr1e-4 + margin0.4 | 1.8M MLP | preserved (fire ~0.84) | **flat ~0.0** | gentle → stable but no ID gain |

→ Confirms (now **8 learnable configs**, every injection × objective) that held-out CSIM **does not move**
at this data/compute scale. EasyControl/XVerse fix the *quality* side (gentle injection keeps faces
detectable — fire stays ~0.8, no align-collapse) but cannot conjure a generalizable identity mapping from
1000 IDs. New tooling: `ip_adapter_flux2.py` (`Flux2EasyCtrl*Processor`/`install_easycontrol`,
`IDModulationInjector`/`install_id_modulation`), `train_id_tokens.py --inject {easycontrol,modulation}
--lora-rank --cond-source {adaface,vaeimg} --cond-size`, `fullgen_id_loss.vae_cond_tokens`,
`_run_idtok_{lduid,modulation,modulation2}.sh`.

### dup-id is the working operating point — its ID↔pose↔QUALITY characterized (the conditional deliverable)
Since training-free **dup-id (keepid+dup3+idfirst) is the only thing that lifts ID** (full-2400 CSIM **0.252**),
validated it on pose **and** image quality (user asked):
- **Pose:** **YAW 82.2%** (left 100 / left-OTS 52 / right 98 / right-OTS 78), PITCH 40.2% (chin-down is the
  known model limit). i.e. pose is **good** and unharmed by dup-id.
- **Quality / distortion (`quality_assess.py`, no-ref Laplacian sharpness; 350 random imgs, SAME prompts):**
  dup-id **does soften** the image — **sharpness 47 vs 87** for the no-dup verified-OTS config (faces still
  form: align 87.8%, so no gross distortion, but visibly smoother). Montage `quality_dupid_vs_nodup.png`.
- **ID↔sharpness Pareto** (dup count, consistent 100-img subset): dup2 **0.199**/sharp 60 → dup3+idfirst
  **0.327**/52 → dup4+idfirst **0.352**/37 → dup5 **0.462**/12. **More dup = more ID, softer image.** So the
  user picks an operating point on this curve; dup3+idfirst is the knee (good ID, moderate softening).

(EasyControl-AdaFace, learnable-dup-id, modulation-lr5e-4 killed early once flat/collapsing; modulation2
final Stage-C: CSIM **0.0116** ≈ baseline 0.0137, align 50% — confirms flat.)

### The softening IS cheaply fixable — dup-id + unsharp (training-free, ID preserved) — **FULL 2400**
The only quality cost of dup-id is reduced sharpness; a post-hoc **unsharp mask (amt 1.0, radius 2)**
recovers it with negligible ID loss. Validated on the **full 2400** (`idpose_winner_unsharp/`,
`_run_unsharp_eval.sh`, `quality_assess.py`):
| (n=2400) | CSIM(f0) | align% | sharpness |
|---|---|---|---|
| dup-id (raw) | **0.252** | 87.8% | 47.8 |
| **dup-id + unsharp 1.0** | **0.249** | 87.3% | **166.8** |
→ sharpening **3.5×** (47.8→166.8, now *above* the no-dup ~87) for only **−0.003 CSIM** and no align change.
amt 1.0 slightly over-sharpens (no-dup level is ~amt 0.5); tune amt to taste. **So "high ID without quality
loss" is achievable *today*: dup-id (keepid+dup3+idfirst) for ID + a cheap unsharp post-process for
sharpness.** Pose is unaffected by sharpening (YAW stays 82.2%).

**Bottom line for the user:** no learnable adapter beat dup-id at this scale (8 mechanisms, incl. the two
most-relevant papers EasyControl + XVerse — they fix *quality* but can't learn ID from 1000 IDs).
**Recommended operating point: dup-id `keepid+dup3+idfirst` (CSIM 0.252, YAW 82.2%, pose unharmed) +
unsharp(1.0) to undo the softening (full-2400: CSIM 0.249, sharpness 47.8→166.8).** Tune ID↔sharpness via dup-count
(dup2→dup5: CSIM 0.20→0.46). A learnable adapter that lifts ID *natively* needs the real regime
(≫1000 IDs + non-distilled/longer-step base), not an ablation.

### Lever A — demographic-matched prompts: the one real ID gain (training-free) ✅
Diagnosis of the dup-id winner's CSIM: it's **bimodal** (median 0.29, p10 0.04 / p90 0.54 → 30% strong >0.4,
**21% fail <0.1**), pose-neutral (L/R/OTS all ~0.28), with **huge per-ID spread (0.03→0.50)**. The fails are
ID-specific; a big chunk is the **prompt's identity word conflicting with the ID's demographic** — each ID is
paired with all 40 pose prompts, ~half of which carry the wrong gender/race word ("A black man…" over an
asian-woman ID → CSIM≈0). The 40 prompts only cover {white,black,latino,middle-eastern} — **no asian/indian**,
yet ~15 IDs are asian/indian → always race-mismatched.

**Validated the lever before building it.** Labelled the 60 eval IDs' gender/race/age with **`google/gemma-4-E2B-it`**
(`Gemma/gemma_id_demographics.py` → `eval_id_demographics_gemma.csv`) — NOT DeepFace, whose race/age is too noisy
(Gemma vs DeepFace agree only 62% gender / 75% race; DeepFace over-called male). Splitting the winner's CSIM by
Gemma-match gives a clean ordering: **gender+race match 0.306 / gender-only 0.303 / race-only 0.217 / neither 0.201**
→ if every output were matched, expected mean ≈ **0.306** (vs 0.252); gender is the dominant factor.

**Lever A = rewrite each prompt's demographic to the ID's** (`flux2_id_pose_matched.py --demo-csv`: `rewrite_demo`
swaps the leading "A <race> <age> <man|woman>" + head pronoun to the ID's; pose/scene clauses untouched). **100-img
validation (same 5 IDs × 20 prompts, seed-matched):**
| | CSIM(f0) | align% |
|---|---|---|
| winner (keep-id) | 0.327 | 92% |
| **lever A (demo-match)** | **0.386** | 90% |
→ **+0.059 (+18%)**, *above* the estimate. Per-ID: the previously-mismatched demographics gain most (00901 man/white
0.42→0.56, 01264 woman 0.24→0.33, 08281 middle-eastern 0.29→0.34); already-strong IDs flat (08788 0.50).

**FULL 2400 (`leverA_demomatch_full/`, definitive):**
| (n=2400) | CSIM(f0) | CSIM(aln) | align% | YAW | PITCH |
|---|---|---|---|---|---|
| winner (keep-id) | 0.252 | 0.287 | 87.8% | 82.2% | 40.2% |
| **lever A (demo-match)** | **0.306** | **0.353** | 86.6% | 80.4% | 38.7% |
→ **+0.054 CSIM (+21%)** — exactly the estimate — for only **−1.8% YAW** (essentially free on pose) and negligible
align/pitch. (Unsharp stacks on top: preserves CSIM, restores sharpness, as in the dup-id+unsharp table above.)

**FINAL recommendation: dup-id `keepid+dup3+idfirst` + verified-OTS + Gemma demographic-matched prompts (lever A)
+ unsharp.** Lever A is the first thing to beat plain dup-id on ID (training-free, ~free on pose, no quality cost):
**CSIM 0.252→0.306 (+21%), YAW 82→80%.** Remaining drag = the ~10% reference/renderability fails (Fei-Fei/Stephen
type), which demographic-matching can't fix — a better/face-cropped reference per ID is the next lever.
Tooling: `Gemma/gemma_id_demographics.py` (gemma-4-E2B-it), `eval_id_demographics_gemma.csv`,
`flux2_id_pose_matched.py --demo-csv` (`rewrite_demo`), `_run_leverA{,_full}.sh`.

## ★ CURRENT STRONGEST VERSION (authoritative spec) — `leverA_demomatch_full/`

**Pipeline (all training-free, FLUX.2-klein frozen, 4 steps, 1024px, seed 42):**
1. **Pose control** = verified-OTS SAM-3D skeleton: left-OTS prompts→p24 skel, right-OTS→p15, all others→
   category-matched skel (`build_matching`, by measured dir+OTS). VAE-encoded → one reference token block.
2. **Identity** = the ID portrait, VAE-encoded → token block, **duplicated ×3** (`--dup-id 3`), placed
   **first** in the reference order (`--id-first`). So `image = [ID, ID, ID, skeleton]`.
3. **Prompt** = the pose prompt with its leading demographic descriptor + head pronoun **rewritten to the
   ID's gemma-4-E2B-it gender/race/age** (`--demo-csv eval_id_demographics_gemma.csv`, `rewrite_demo`);
   pose/scene clauses unchanged. Keep-identity (no strip).
4. `pipe(image=[ID×3, skel], prompt=rewritten)` → **unsharp mask (amt 0.5–1.0)** post-process for sharpness.

**Diff vs baselines** — `CSIM(aln)` = mean over **only the face-detected outputs**, with its count
`n_detected/N`; `CSIM(f0)` = all outputs (undetected face = 0); pose = **yaw AND pitch** acc:
| stage | what's added | CSIM(aln) ↑ | n_detected / N | CSIM(f0) ↑ | YAW ↑ | PITCH ↑ |
|---|---|---|---|---|---|---|
| no-ID (prompt+skel only) | — | 0.015 | 90/100 | 0.014 | 85.0% | 55.0% |
| exp1 (prompt-only `[ID]`) | ID ref ×1 | 0.153 | 2102/2400 | 0.134 | 73.7% | 46.8% |
| + SAM-3D control `[sam3d,ID]` | pose ctrl | 0.111 | 2016/2400 | 0.093 | **84.4%** | 50.8% |
| **winner** `[ID×3,sam3d]` id-first | dup-id ×3 + id-first | 0.287 | 2106/2400 | 0.252 | 82.2% | 40.2% |
| **★ STRONGEST** = winner + lever A | demographic-aligned prompt (+unsharp) | **0.353** | 2078/2400 | **0.306** | 80.4% | 38.7% |

(face-detect rate stays ~87% across the ID stages — the CSIM gains are real, not alignment survivorship.)
→ Strongest vs no-ID baseline: **CSIM 0.014→0.306 (~22×)**; vs prompt-only exp1: f0 0.134→0.306 (~2.3×);
vs the previous-best winner: **+0.054 f0 (+21%) at −1.8% YAW**, sharpness restored by unsharp (CSIM −0.003).
**Only added ingredient over the winner is a training-free textual prompt rewrite + a post-hoc unsharp.**
Remaining weak axes (unchanged, both model-level limits): chin-down PITCH (~39%), and ~10% per-ID
reference-renderability failures. Recorded in thesis `thesis_personalization/README.md` §3.3.4–3.3.5.
