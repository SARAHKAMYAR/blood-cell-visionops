PY ?= .venv/bin/python
RUNS ?= runs/yolo11n_e100 runs/yolo11s_e100
BEST ?= runs/yolo11n_e100

.PHONY: help install data train evaluate stats promote benchmark ingest api dashboard test lint up down

help:  ## Show available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

install:  ## Create venv and install everything
	python3.11 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[onnx,dashboard,dev]"

data:  ## Clone BCCD and convert to YOLO format
	test -d BCCD_Dataset || git clone --depth 1 https://github.com/Shenggan/BCCD_Dataset.git
	$(PY) scripts/prepare_dataset.py

train:  ## Train YOLO11n and YOLO11s (100 epochs)
	$(PY) scripts/train.py --model yolo11n.pt --epochs 100 --name yolo11n_e100
	$(PY) scripts/train.py --model yolo11s.pt --epochs 100 --name yolo11s_e100

evaluate:  ## Test-set detection + counting metrics
	$(PY) scripts/evaluate.py $(RUNS)

stats:  ## Per-image test artefacts + paired bootstrap confidence intervals
	$(PY) scripts/export_predictions.py $(RUNS) runs/blood_cells_baseline
	$(PY) scripts/bootstrap_eval.py --runs yolo11n_e100 yolo11s_e100 blood_cells_baseline

promote:  ## Copy best run to models/ and export ONNX
	$(PY) scripts/promote_model.py $(BEST)

benchmark:  ## Latency: PyTorch vs ONNX, CPU vs accelerator
	$(PY) scripts/benchmark.py models/best.pt models/best.onnx runs/yolo11s_e100/weights/best.pt

ingest:  ## Analyse the test split into PostgreSQL
	$(PY) scripts/ingest.py data/blood_cells/images/test

api:  ## Run the API locally (hot reload)
	$(PY) -m uvicorn bloodcell.api:app --reload

dashboard:  ## Run the dashboard locally
	$(PY) -m streamlit run dashboard/app.py

test:  ## Run tests
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	$(PY) -m ruff check bloodcell scripts tests dashboard

up:  ## Start the full stack (Postgres + API + dashboard)
	docker compose up --build -d

down:  ## Stop the stack
	docker compose down
