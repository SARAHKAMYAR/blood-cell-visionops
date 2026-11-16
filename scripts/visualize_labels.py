
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "blood_cells"

images = sorted((DATA / "images" / "train").glob("*.jpg"))

if not images:
    raise FileNotFoundError("No training images found")

image_path = images[0]
label_path = DATA / "labels" / "train" / f"{image_path.stem}.txt"

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Cannot open image: {image_path}")

height, width = image.shape[:2]

classes = ["RBC", "WBC", "Platelets"]
colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]

for line in label_path.read_text().splitlines():
    class_id, xc, yc, bw, bh = map(float, line.split())
    class_id = int(class_id)

    x1 = int((xc - bw / 2) * width)
    y1 = int((yc - bh / 2) * height)
    x2 = int((xc + bw / 2) * width)
    y2 = int((yc + bh / 2) * height)

    color = colors[class_id]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        classes[class_id],
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )

output = PROJECT / "reports" / "sample_annotations.jpg"
output.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output), image)

print("Image:", image_path.name)
print("Saved visualization:", output)
