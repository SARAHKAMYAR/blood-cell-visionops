"""Generate README figures into docs/assets/.

  training_curves.png   validation mAP50-95 per epoch for each run
  predictions_grid.png  ground truth vs. prediction on held-out test images

Usage:
    python scripts/make_figures.py --runs runs/yolo11n_e100 runs/yolo11s_e100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from bloodcell.config import CLASS_NAMES, PROJECT_ROOT, get_settings  # noqa: E402

ASSETS = PROJECT_ROOT / "docs" / "assets"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical palette (CVD-safe)
CLASS_COLORS = dict(zip(CLASS_NAMES, SERIES))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def training_curves(runs: list[Path]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    for idx, (color, run) in enumerate(zip(SERIES, runs)):
        df = pd.read_csv(run / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        y = df["metrics/mAP50-95(B)"]
        ax.plot(df["epoch"], y, color=color, linewidth=2, label=run.name)
        best = y.idxmax()
        ax.scatter(df["epoch"][best], y[best], s=40, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(
            f"{run.name}  {y[best]:.3f}", (df["epoch"][best], y[best]),
            xytext=(-8, 10 if idx % 2 == 0 else -18), textcoords="offset points",
            fontsize=9, color=INK, ha="right",
        )
    ax.set_ylim(0.3, 0.72)
    ax.set_title("Validation mAP@0.5:0.95 by epoch", loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("mAP@0.5:0.95", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    out = ASSETS / "training_curves.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def _draw(image: Image.Image, boxes: list[tuple[str, list[float]]]) -> Image.Image:
    img = image.copy()
    d = ImageDraw.Draw(img)
    for name, (x1, y1, x2, y2) in boxes:
        d.rectangle([x1, y1, x2, y2], outline=CLASS_COLORS[name], width=3)
    return img


def predictions_grid(n: int = 3) -> Path:
    from bloodcell.inference import Detector

    settings = get_settings()
    root = settings.dataset_yaml.parent
    images = sorted((root / "images" / "test").glob("*.jpg"))[:: max(1, 56 // n)][:n]
    detector = Detector(settings)

    fig, axes = plt.subplots(n, 2, figsize=(9, 3.4 * n), dpi=130)
    for row, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        gt = []
        for line in (root / "labels" / "test" / f"{path.stem}.txt").read_text().splitlines():
            c, xc, yc, bw, bh = map(float, line.split())
            gt.append((CLASS_NAMES[int(c)], [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
        pred = detector.predict(image, path.name)
        pb = [(d.cell_class, d.bbox) for d in pred.detections]
        for col, (title, boxes) in enumerate([("Ground truth", gt), ("Prediction", pb)]):
            ax = axes[row, col]
            ax.imshow(_draw(image, boxes))
            counts = {k: sum(b[0] == k for b in boxes) for k in CLASS_NAMES}
            ax.set_title(
                f"{title}: RBC {counts['RBC']} · WBC {counts['WBC']} · Plt {counts['Platelets']}",
                fontsize=9, color=INK,
            )
            ax.axis("off")
    handles = [plt.Line2D([], [], color=c, linewidth=3, label=k) for k, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ASSETS / "predictions_grid.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = [r if r.is_absolute() else PROJECT_ROOT / r for r in args.runs]
    print(training_curves(runs))
    print(predictions_grid())


if __name__ == "__main__":
    main()
