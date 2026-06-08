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

# 2026-06-05 — T2I model comparison: pose accuracy across 5 generators (40 prompts)

5 text-to-image models, each generating the SAME 40 pose prompts (1 img/prompt). Measure pose accuracy = does the generated head pose match the prompt's intended pose. `t2i_pose_compare/` (`SUMMARY.csv`, `{model}_pose_acc.csv`, `{model}_angles.json`, `pose_compare.png`). Runner `_run_t2i_pose_compare.sh` (stage A) + `_run_t2i_poseB.sh` (stage B re-run with user thresholds).

**Pipeline (2 stage, 2 env):** A) `extract_angles_c.py <folder> --out _angles.json` (env smplestx) → SMPLest-X head_yaw/pitch/head_to_torso. B) `eval_smplestx_vs_prompt.py --folder --angles --out` (env paul) → T5Gemma classifies the prompt's intended pose, maps angles→labels, computes yaw/pitch match.

**Thresholds (user-specified, uniform across all 5):** yaw >15° = left/right, pitch >±15° = up/down, |head_to_torso_yaw| >25° = "over the shoulder". (Note: the older `pretrained/pose_acc_smplestx.csv` used the docstring rule pitch±25/OS35 — different; the `t2i_pose_compare/` numbers use 15/15/25.)

| model | yaw acc | pitch acc | mean \|yaw\| | mean \|pitch\| | SMPLest-X fail |
|---|---|---|---|---|---|
| **flux2** | **67.5%** | 40.0% | 50.4° | 14.0° | 0/40 |
| sana | 55.0% | 42.5% | 53.7° | 15.2° | 0/40 |
| softrepa | 42.5% | 37.5% | 59.6° | 15.8° | 0/40 |
| infinity | 35.0% | 37.5% | 30.2° | 13.8° | 0/40 |
| hart | 32.5% | 42.5% | 38.2° | 16.8° | 0/40 |

**Takeaways.** flux2 best yaw fidelity (67.5%), sana 2nd (55%); infinity/hart worst (~33%). mean|yaw| shows *willingness to turn*: softrepa turns most (59.6°) but imprecise (42.5%); **infinity under-rotates (30°) → lowest yaw acc**; flux2 turns a lot (50°) AND accurately. Pitch ~37–42% for all (chin is hard for every T2I, model-independent). All 5 had a person detected in 40/40. All prompts specify chin → pitch denominator = 40.

### Angle distributions (`angle_hist.png`, 5° bins): |yaw|, pitch split by prompt chin up/down
Balanced design: each model has exactly 20 chin-up + 20 chin-down prompts. Median measured angle:

| model | median \|yaw\| | chin-UP median pitch | chin-DOWN median pitch |
|---|---|---|---|
| flux2 | 50.8° | +20.2 | −4.5 |
| sana | 52.9° | +18.1 | −2.8 |
| infinity | 30.7° | +15.8 | −5.8 |
| softrepa | 59.5° | +18.5 | **+8.2** ⚠ |
| hart | 43.3° | +24.2 | **+8.7** ⚠ |

**The pitch failure is almost entirely chin-DOWN.** Chin-up is handled well by all (measured pitch peaks at +15–25°, clears the +15° threshold). Chin-down should be negative but the distributions straddle 0: flux2/sana/infinity only weakly negative (−3 to −6°), and **softrepa/hart actually go positive (median +8°) — they raise the chin when asked to lower it.** This is why pitch acc sits at ~40%; splitting up/down exposes it (the aggregate pitch_acc hid the asymmetry). |yaw| panel confirms infinity under-rotates (mass at 0–35°) while softrepa/sana/flux2 sit at 40–65°.


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
