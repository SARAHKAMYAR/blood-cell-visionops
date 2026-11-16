FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BCV_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml README.md ./
COPY bloodcell ./bloodcell
COPY db ./db
RUN pip install ".[onnx]"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "bloodcell.api:app", "--host", "0.0.0.0", "--port", "8000"]
