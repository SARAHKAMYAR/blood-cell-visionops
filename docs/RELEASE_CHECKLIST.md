# Release checklist

Status of Blood Cell VisionOps for public release. ✅ = done and checked in this
repository; ⬜ = still open.

## Verified
- ✅ 68 automated tests pass offline (`make test`); lint is clean (`make lint`).
- ✅ Docker Compose stack builds and runs (PostgreSQL + API + dashboard); `/health`, `/predict`,
  `/analyze` and upload validation exercised with real BCCD images against the running API.
- ✅ All four dashboard tabs render without errors; screenshot captured from the running stack
  (`docs/assets/dashboard.png`).
- ✅ Reported mAP reproduced from committed per-image artefacts to four decimals; bootstrap output
  is byte-identical on re-run (apart from the runtime line).
- ✅ Counting metrics reproduced through the real `Detector` class (3.11 / 0.02 / 0.46), and the
  served NMS IoU is pinned to the calibration value by tests.
- ✅ No credentials in tracked files or Git history (pattern scan); `.env`, Streamlit secrets,
  virtualenvs, datasets and training runs are git-ignored and were never committed.
- ✅ Released under AGPL-3.0 (`LICENSE`); third-party licences recorded in `THIRD_PARTY_NOTICES.md`.
- ✅ GitHub Actions passed on `main` (lint, tests, both Docker image builds).
- ✅ Commit history uses a GitHub no-reply address.

## Open
1. ⬜ **Live Gemini check.** No API key was available, so no real call has been made. Put
   `GEMINI_API_KEY=...` in `.env` (never commit it), then run `python scripts/live_gemini_check.py`.
   It makes one request, prints the claim-check result and saves `docs/examples/gemini_report_example.md`.
2. ⬜ **Dependency vulnerability scan** (`pip-audit` was not installed in the development environment).
3. ⬜ **Make the repository public** once the items above are accepted or done (it was published private first).

## Known open issues (documented, not hidden)
- Platelet threshold calibration slightly worsens platelet counting on test (+0.18, CI excludes 0).
  Do not tune this on the test set.
- 56 test images: several model differences are not resolved (`docs/EVALUATION_UNCERTAINTY.md`).
- The report checker tests traceability, not meaning (blind spots listed in the README).
- Evaluation used the PyTorch weights; the ONNX export served in Docker gives slightly different
  detections (test count error 2.62 / 0.02 / 0.41 vs 3.11 / 0.02 / 0.46).
- `models/best.pt` and `best.onnx` embed the local training paths in their metadata.
- The demo stack has no authentication and a default database password; it binds to localhost only.
