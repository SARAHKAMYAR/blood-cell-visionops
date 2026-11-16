"""Batch-analyse a folder of images and store the results in PostgreSQL.

Usage:
    python scripts/ingest.py data/blood_cells/images/test --limit 20
    python scripts/ingest.py data/blood_cells/images/test --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from bloodcell.config import PROJECT_ROOT, get_settings  # noqa: E402
from bloodcell.db import encode_jpeg, init_db, save_analysis  # noqa: E402
from bloodcell.inference import Detector  # noqa: E402
from bloodcell.pipeline import analyze_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Template reports only")
    args = parser.parse_args()

    folder = args.folder if args.folder.is_absolute() else PROJECT_ROOT / args.folder
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    images = images[: args.limit] if args.limit else images
    if not images:
        sys.exit(f"No images found in {folder}")

    settings = get_settings()
    init_db()
    detector = Detector(settings)
    for i, path in enumerate(images, 1):
        image = Image.open(path).convert("RGB")
        analysis = analyze_image(detector, image, path.name, settings, use_llm=not args.no_llm)
        row_id = save_analysis(analysis, image_jpeg=encode_jpeg(image))
        c = analysis.counts
        print(
            f"[{i}/{len(images)}] #{row_id} {path.name}: RBC={c['RBC']} WBC={c['WBC']} "
            f"Plt={c['Platelets']} QC={analysis.quality_control.status} "
            f"report={analysis.report_source}"
        )


if __name__ == "__main__":
    main()
