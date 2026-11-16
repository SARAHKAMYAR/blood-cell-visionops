
"""Convert the BCCD Pascal-VOC annotations into a YOLO dataset.

Split: 70% train / 15% val / 15% test, seeded for reproducibility.
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / "BCCD_Dataset" / "BCCD"
OUTPUT = PROJECT / "data" / "blood_cells"

CLASSES = {"RBC": 0, "WBC": 1, "Platelets": 2}
random.seed(42)


def convert_annotation(xml_path, label_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {xml_path}")

    lines = []

    for obj in root.findall("object"):
        name = obj.findtext("name")

        if name not in CLASSES:
            raise ValueError(f"Unknown class: {name}")

        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Keep coordinates inside the image.
        xmin = max(0, min(xmin, img_w))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h))
        ymax = max(0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            print(f"WARNING: Skipping invalid {name} box in {xml_path.name}")
            continue

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(
            f"{CLASSES[name]} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    label_path.write_text("\n".join(lines) + "\n")


def main():
    images_dir = SOURCE / "JPEGImages"
    annotations_dir = SOURCE / "Annotations"

    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        images = sorted(images_dir.glob("*.png"))

    if not images:
        raise FileNotFoundError("No images found")

    random.shuffle(images)

    n = len(images)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    # Validate all image/annotation pairs first.
    for image in images:
        xml_path = annotations_dir / f"{image.stem}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)

    for split, split_images in splits.items():
        image_output = OUTPUT / "images" / split
        label_output = OUTPUT / "labels" / split

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, image_output / image.name)

            xml_path = annotations_dir / f"{image.stem}.xml"
            label_path = label_output / f"{image.stem}.txt"

            convert_annotation(xml_path, label_path)

        print(f"{split}: {len(split_images)} images")

    yaml_path = OUTPUT / "data.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: RBC\n"
        "  1: WBC\n"
        "  2: Platelets\n"
    )

    print(f"\nDataset ready: {OUTPUT}")
    print(f"YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()
