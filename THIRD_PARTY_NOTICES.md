# Third-party notices

This file records the third-party software, model weights and data that Blood
Cell VisionOps uses or redistributes, with the licence each one declares.
Versions are those installed when the released model was trained and evaluated;
licence strings are copied from each package's own metadata. It is a factual
record, not legal advice.

The original code in this repository is licensed under AGPL-3.0 (see `LICENSE`).

## 1. Ultralytics YOLO (code and model weights): AGPL-3.0

- **Code:** the `ultralytics` Python package (version 8.4.212) is imported by
  `bloodcell/inference.py` and the scripts in `scripts/`. Its package metadata and bundled
  licence file declare **AGPL-3.0**. Ultralytics also sells a separate Enterprise licence.
- **Pretrained weights:** training started from `yolo11n.pt` / `yolo11s.pt`, downloaded from
  Ultralytics. Their embedded licence field reads "AGPL-3.0 License (https://ultralytics.com/license)".
  These files are **not** redistributed in this repository.
- **Fine-tuned weights redistributed here:** `models/best.pt` and `models/best.onnx` were produced by
  fine-tuning `yolo11n.pt` with the Ultralytics trainer. Both files carry an embedded licence field
  set by Ultralytics to AGPL-3.0, and Ultralytics' published position is that models trained with its
  software are covered by AGPL-3.0 unless an Enterprise licence is held. They are distributed here
  under AGPL-3.0 on that basis.
- **Source:** https://github.com/ultralytics/ultralytics · **Licence text:** https://www.gnu.org/licenses/agpl-3.0.txt

## 2. BCCD Dataset: MIT

- **What is used:** 364 annotated blood-smear images, used for training and evaluation.
- **What is redistributed here:** the dataset itself is **not** in this repository (`make data`
  clones it). Small derived items are: `docs/assets/predictions_grid.png` and
  `reports/sample_annotations.jpg` (four dataset images with boxes drawn on them), any dashboard
  screenshot showing a dataset image, and per-image evaluation statistics in `reports/predictions/`
  (numbers and file names only, no pixels).
- **Source:** https://github.com/Shenggan/BCCD_Dataset
- **Provenance note:** the BCCD README credits the original images and annotations to
  https://github.com/cosmicad/dataset and https://github.com/akshaylamba/all_CELL_data and states that
  the re-organised dataset is under the MIT licence. This project relies on that statement; the
  upstream sources' own terms and the patient-consent status of the images are not documented there
  and have not been independently verified.

BCCD licence notice (reproduced as the MIT licence requires):

```
MIT License

Copyright (c) 2017 shenggan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Python dependencies (installed, not redistributed in this repository)

| Package | Version | Declared licence |
|---|---|---|
| ultralytics | 8.4.212 | AGPL-3.0 |
| torch | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib |
| torchvision | 0.29.2 | BSD |
| fastapi | 0.163.0 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| python-multipart | 0.0.33 | Apache-2.0 |
| pydantic | 2.14.3 | MIT |
| pillow | 13.0.0 | MIT-CMU |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| psycopg2-binary | 2.9.13 | LGPL with exceptions |
| google-genai | 3.3.0 | Apache-2.0 |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| onnx | 1.23.2 | Apache-2.0 |
| onnxruntime | 1.32.0 | MIT License |
| onnxslim | 0.1.99 | MIT |
| streamlit | 1.67.0 | Apache-2.0 |
| pandas | 3.1.1 | BSD-3-Clause |
| requests | 2.34.2 | Apache-2.0 |
| altair | 6.3.0 | BSD-3-Clause (per bundled licence text) |
| opencv-python | 5.0.0.93 | Apache 2.0 |
| matplotlib | 3.11.2 | Matplotlib licence (PSF-based; figures script only) |

`psycopg2-binary` is LGPL with exceptions and is used unmodified as a library. The Docker images
built from `docker/` install these packages; anyone distributing those images takes on each
package's redistribution terms, including AGPL-3.0 for Ultralytics.

## 4. Google Gemini API

Optional, accessed over the network with the user's own API key under Google's API terms.
No Google code or model weights are redistributed.
