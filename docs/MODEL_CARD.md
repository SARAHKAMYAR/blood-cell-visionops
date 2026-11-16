# Model Card: Blood Cell VisionOps detector

## Model details
- **Architecture:** Ultralytics YOLO11 (single-stage anchor-free detector), fine-tuned from COCO-pretrained weights.
- **Variants trained:** YOLO11n (2.6 M params) and YOLO11s (9.4 M params), 100 epochs max with early stopping (patience 30), 640 px input, cosine LR, seed 42.
- **Served model:** see `models/model_info.json` (source run, SHA-256 of `.pt` and `.onnx`, test metrics).
- **Classes:** `RBC` (red blood cell), `WBC` (white blood cell / leukocyte), `Platelets`.

## Intended use
- Research, education and engineering demonstration of a CV + QC + LLM-reporting pipeline.
- **Out of scope:** clinical diagnosis, patient management, or any decision affecting care. The model is not validated or cleared as a medical device.

## Training data
- **BCCD** (Blood Cell Count and Detection), 364 annotated 640×480 JPEG images of Giemsa-stained peripheral blood smears, Pascal-VOC boxes.
- Split (seeded, image-level): 254 train / 54 val / 56 test.
- Augmentation: HSV jitter, mosaic (off for the last 10 epochs), translate/scale, horizontal **and vertical** flips, ±10° rotation (microscopy has no canonical orientation).

## Evaluation
Metrics on the held-out **test** split (never used for training or model selection) are in `reports/metrics/*.json` and in the README:
- Detection: precision, recall, mAP@0.5, mAP@0.5:0.95 (overall and per class).
- Counting: per-image mean absolute count error and signed bias per class, at conf 0.25 and at per-class thresholds calibrated on the validation split.

### Headline test results (served YOLO11n)
| Metric | Value |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.946 / 0.644 |
| Per-class mAP@0.5:0.95 | RBC 0.644 · WBC 0.785 · Platelets 0.504 |
| Count MAE per image (calibrated thresholds) | RBC 3.11 · WBC 0.02 · Platelets 0.46 |
| Calibrated thresholds (val split) | RBC 0.55 · WBC 0.15 · Platelets 0.10 |
| Latency | 18.9 ms CPU (ONNX Runtime) · 10.9 ms Apple M4 (MPS) |

## Known limitations & biases
- **Small, single-source dataset:** one lab, one stain, one camera. Expect a drop in performance on other microscopes, stains or magnifications.
- **Label noise:** BCCD has many missing RBC annotations. At conf 0.25 the model counts +7.45 more RBCs per image than the labels (mean 11.3). Visual inspection shows most of these are real, unlabelled cells. RBC precision and count metrics are therefore pessimistic, and the val-calibrated RBC threshold (0.55) partly fits annotation density, not biology.
- **Overlapping / clumped RBCs** are hard to separate; QC flags `OVERLAPPING_BOXES` to surface them.
- **Platelets are small** (≈20–30 px), so they are the weakest class; tiled inference (SAHI) is a planned improvement.
- **One WBC per image:** BCCD images are centred on a leukocyte, so the QC rules `NO_WBC_DETECTED` / `UNUSUAL_WBC_COUNT` are dataset-specific heuristics.
- WBC sub-types (neutrophil, lymphocyte, ...) are **not** distinguished.

## LLM report component
- Model: Google Gemini (configurable via `GEMINI_MODEL`).
- Input: only aggregate measurements (counts, confidence stats, QC flags). No image and no patient data.
- Guardrails: system prompt forbids diagnosis and invented numbers; temperature 0.2; fixed section structure; deterministic template fallback on any error or missing key; mandatory disclaimer appended in code, not by the LLM.

## Ethical considerations
Automated counts can create false confidence. The system shows QC status and reviewer actions next to every result and labels all output as research-only.
