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
- Counting: per-image mean absolute count error and signed bias per class, at conf 0.25 and at per-class thresholds calibrated on the validation split, computed through the serving configuration (NMS IoU 0.5).
- Uncertainty: paired image-level bootstrap (B = 2000) 95% intervals for every metric and model comparison; see [EVALUATION_UNCERTAINTY.md](EVALUATION_UNCERTAINTY.md).

### Headline test results (served YOLO11n)
| Metric | Value |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 (95% CI) | 0.946 [0.924, 0.961] / 0.644 [0.618, 0.677] |
| Per-class mAP@0.5:0.95 | RBC 0.644 · WBC 0.784 · Platelets 0.504 |
| Count MAE per image (calibrated thresholds, 95% CI) | RBC 3.11 [2.46, 3.77] · WBC 0.02 [0.00, 0.05] · Platelets 0.46 [0.32, 0.62] |
| Calibrated thresholds (val split) | RBC 0.55 · WBC 0.15 · Platelets 0.10 |
| Latency | 18.9 ms CPU (ONNX Runtime) · 10.9 ms Apple M4 (MPS) |

## Known limitations & biases
- **Small, single-source dataset:** one lab, one stain, one camera. Expect a drop in performance on other microscopes, stains or magnifications.
- **Label noise:** BCCD has many missing RBC annotations. At conf 0.25 the model counts +7.45 more RBCs per image than the labels (mean 11.3). Visual inspection shows most of these are real, unlabelled cells. RBC precision and count metrics are therefore pessimistic, and the val-calibrated RBC threshold (0.55) partly fits annotation density, not biology.
- **Overlapping / clumped RBCs** are hard to separate; QC flags `OVERLAPPING_BOXES` to surface them.
- **Platelets are small** (≈20–30 px), so they are the weakest class; tiled inference (SAHI) is a planned improvement.
- **One WBC per image:** BCCD images are centred on a leukocyte, so the QC rules `NO_WBC_DETECTED` / `UNUSUAL_WBC_COUNT` are dataset-specific heuristics.
- WBC sub-types (neutrophil, lymphocyte, ...) are **not** distinguished.
- **Platelet threshold calibration does not transfer:** the val-calibrated platelet threshold (0.10) increases test count error versus conf 0.25 (+0.18 per image, 95% CI [+0.05, +0.30]). The validation set has too few platelets (~1 per image) to calibrate this class reliably. Open issue.
- **Counts depend on NMS settings.** Thresholds were calibrated with NMS IoU 0.5; serving uses the same value (`BCV_IOU`). Changing it invalidates the calibration.
- **PyTorch vs ONNX:** reported metrics use the PyTorch weights. The ONNX export served in Docker uses a fixed 640×640 input and gives slightly different detections (test count error RBC 2.62 · WBC 0.02 · Platelets 0.41, measured through the serving code).
- **Small test set:** 56 images. YOLO11n vs YOLO11s and 100-epoch vs 5-epoch training are not resolved overall at 95%; this is not evidence of equivalence.

## LLM report component
- Model: Google Gemini (configurable via `GEMINI_MODEL`).
- Input: only aggregate measurements (counts, confidence stats, QC flags). No image and no patient data.
- Output: structured JSON (`summary`, `quality_assessment`, `reviewer_actions`) validated with Pydantic. The LLM writes prose only; all tables and measurements are rendered by code.
- Claim check (deterministic, no second LLM): every number must trace to the payload at its written precision; class- and total-attached numbers must equal that specific count; the summary must state the total and QC status; listed clinical terms are rejected. In a one-off manual check, 0 of 112 faithful narratives from test-set outputs were falsely rejected.
- Known gaps in the claim check: it tests traceability, not meaning. It accepts a real number reused with the wrong meaning or unit, a wrong QC figure that equals another payload value, prose that contradicts the QC status while naming it, numbers written as words, and clinical wording outside the term list. The code-rendered tables are the source of truth.
- Guardrails: temperature 0.2; template fallback on missing key, API error, empty or safety-blocked response, invalid JSON, schema violation or failed verification; the reason is returned in `report_validation`; the disclaimer is appended in code.
- Not yet verified against the live Gemini API (no key in the development environment); all LLM paths are tested with recorded outputs.

## Ethical considerations
Automated counts can create false confidence. The system shows QC status and reviewer actions next to every result and labels all output as research-only.
