# Release checklist

Status of Blood Cell VisionOps before public release. ✅ = done and verified in
this repository; ⬜ = still to do (owner: repository author).

## Verified in this repository
- ✅ Test suite passes offline (`make test`): unit, API contract, report verification, statistics, real-model integration.
- ✅ Lint clean (`make lint`).
- ✅ Full stack previously ran in Docker Compose (Postgres + API + dashboard); analyses stored and reviewed in the dashboard.
- ✅ Reported mAP reproduced from committed per-image artefacts (`reports/predictions/`) to within 0.001 of `model.val()`, enforced by `tests/test_stats.py`.
- ✅ Counting metrics computed through the serving configuration (NMS IoU 0.5), also enforced by a test.
- ✅ All LLM report paths (valid, inconsistent, hallucinated, invalid JSON, API failure) tested with recorded outputs.
- ✅ No secrets tracked: `.env`, `**/secrets.toml`, virtualenvs and raw data are git-ignored.

## Blockers before going public
1. ⬜ **Choose a licence.** Ultralytics YOLO is AGPL-3.0; pick a compatible licence and add `LICENSE`.
2. ⬜ **Live Gemini check.** Put a key in `.env` (`GEMINI_API_KEY`), run
   `curl -F "file=@BCCD_Dataset/BCCD/JPEGImages/BloodImage_00003.jpg;type=image/jpeg" "localhost:8000/analyze?persist=false"`
   on 5–10 images and confirm `report_source` is `gemini` and `report_validation.status` is `verified`.
   If the SDK rejects the response schema, or narratives are rejected often, review the `issues` list before release.
   Paste one verified report into the README.
3. ⬜ **Dashboard screenshot.** Docker must be running.
   1. `docker compose up -d` (use `API_PORT=8010` if port 8000 is taken), then `make data` (if the dataset is not present) and `make ingest`.
   2. Open http://localhost:8501 → **Review** tab → pick an image with QC status **REVIEW**.
   3. Browser window about 1440×900; capture the QC badge, flags and side-by-side detections.
   4. Save as `docs/assets/dashboard.png` (PNG, < 500 kB), then un-comment the image line in the README's *Dashboard* section and delete the TODO comment.
4. ⬜ **Rebuild and restart the stack after this change set.** Serving NMS IoU changed from 0.7 to 0.5 to match the evaluated configuration; `docker compose up -d --build` picks it up.
5. ⬜ **Push and confirm CI is green** (the workflow has never run). Push the branch as `main` so the repo front page shows this work.

## Known open issues (document, don't hide)
- Platelet threshold calibration hurts on test (+0.18 count MAE, CI excludes 0). Options for a future version: keep conf 0.25 for platelets, or calibrate with cross-validation over train+val. Do **not** choose using the test set.
- Test set is 56 images; several model differences are not resolved (see `docs/EVALUATION_UNCERTAINTY.md`).
- The report checker tests traceability, not meaning: it accepts real numbers reused with the wrong meaning or unit, wrong QC figures that equal another value, and numbers written as words (full list in the README).

## Nice to have
- Hosted demo (e.g. Hugging Face Spaces) linked from the README.
- Short GIF of the Analyze tab.
- `CHANGELOG.md` and a `v1.0.0` tag.
