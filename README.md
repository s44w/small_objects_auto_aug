# Adaptive Augmentations for Small Object Detection (MVP)

This repository contains an MVP implementation of an interpretable, rule-based
augmentation selection pipeline for small-object detection, following
`instructions.md`.

## What is implemented

- YOLO dataset validation for VisDrone-like structure.
- Dataset analyzer with:
  - COCO-style area bins (`small`, `medium`, `large`),
  - `small_ratio`, `tiny_ratio`,
  - density, class imbalance, image sizes, illumination statistics.
- Rule-based adaptive policy engine with explainable `decision_report.json`.
- Custom augmentation layer:
  - bbox-aware crop,
  - bbox-only copy-paste with IoU/IoA constraints,
  - optional object bank.
- Training runner for:
  - `baseline`,
  - `manual`,
  - `adaptive`,
  - `adaptive_no_mosaic`,
  - `adaptive_no_custom_albu`.
- YOLO -> COCO conversion and COCOeval runner with `AP_small` and optional `AP_tiny`.
- Colab notebook for the main MVP pipeline.

## Repository layout

```text
configs/
  project_config.yaml
  baseline.yaml
  manual.yaml
src/
  data/
  analysis/
  policy/
  augmentation/
  training/
  evaluation/
  utils/
  pipeline_mvp.py
notebooks/
  mvp_pipeline_colab.ipynb
tests/
artifacts/
runs/
```

## Quick start (local)

```bash
pip install -r requirements.txt
python -m src.pipeline_mvp --project-config configs/project_config.yaml
```

With training and evaluation:

```bash
python -m src.pipeline_mvp \
  --project-config configs/project_config.yaml \
  --run-training \
  --run-eval \
  --pred-labels-dir path/to/predicted/labels
```

## Main Colab workflow

Use notebook:

- `notebooks/mvp_pipeline_colab.ipynb`

It runs:
1. dataset validation,
2. analysis and stats export,
3. adaptive policy generation,
4. optional training suite,
5. COCO conversion/evaluation.

## Local CPU component test (<= 30 min)

Use notebook:

- `notebooks/local_cpu_component_test.ipynb`

This notebook is optimized for local CPU smoke testing:
- builds a small subset (`train=120`, `val=40` by default),
- runs analyzer + rule engine,
- evaluates an **already trained YOLO** model on subset val,
- computes COCOeval metrics (`AP_small`, optional `AP_tiny`),
- logs stage timings to verify total runtime budget.

## Colab GPU in VS Code (Remote-SSH)

Automated setup files:
- `notebooks/colab_vscode_remote_gpu.ipynb`
- `scripts/setup_colab_vscode_remote.ps1`
- `docs/COLAB_VSCODE_REMOTE.md`
