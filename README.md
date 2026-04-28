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
- Object-bank based bbox copy-paste support for adaptive training.
- Experiment manifest with timings, artifact paths, and multi-run COCOeval reports.
- Optional multi-seed training suite (`training.seeds`) for mean/std replication.
- Budget-aware AutoAug-like random-search candidate generator for comparison studies.
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
  --run-eval
```

Notes:
- Dataset modes are configured in `configs/project_config.yaml`:
  - `dataset.mode=manual`: validate existing YOLO dataset.
  - `dataset.mode=auto`: convert from `dataset.raw_root` then validate.
- Evaluation can auto-generate predictions from the best available run
  and now evaluates all available MVP runs (`baseline`, `manual`, `adaptive`,
  `adaptive_no_mosaic`, `adaptive_no_custom_albu`) when `--pred-labels-dir` is omitted.
- Training profile can be selected via `--train-profile` (`fast`, `final`, `balanced`,
  `quality`, `hour`, `max_quality`).
- Set `training.seeds: [42, 43, 44]` for multi-seed replication.
- Set `policy.use_object_bank: true` to build `artifacts/policy/object_bank.json`
  and use it in adaptive bbox copy-paste.

## Documentation

- `docs/DATASET_ANALYTICS.md` describes dataset statistics, outputs, and recommended extensions.
- `docs/AUGMENTATION_POLICY.md` describes rule-based policy generation and custom augmentations.
- `docs/THRESHOLDS.md` documents threshold values, interpretation, and calibration.
- `docs/REFERENCES.md` lists the primary sources used by the MVP.

## Main Colab workflow

Use notebook:

- `notebooks/mvp_pipeline_colab.ipynb`
- `notebooks/coco_small_pipeline_colab.ipynb` (COCO-small pipeline)
- `notebooks/autoaug_vs_adaptive_comparison.ipynb` (AutoAug-like search vs this project)
- `notebooks/visdrone_tiny_fixture_smoke.ipynb` (self-contained VisDrone-like module smoke test, <1 min)
- `notebooks/visdrone_real_subset_smoke.ipynb` (fast smoke test on a real prepared VisDrone YOLO subset)

It runs:
1. dataset validation,
2. analysis and stats export,
3. adaptive policy generation,
4. optional training suite,
5. COCO conversion/evaluation.

## COCO-small workflow

COCO-small config:
- `configs/coco_small_config.yaml`

Key idea:
- convert COCO 2017 annotations to YOLO,
- keep only object instances with area `<= 32^2 px`,
- then run the same adaptive pipeline.

In Colab notebook for COCO-small, the first cells download archives directly
to Google Drive and unpack them there to avoid repeated local downloads.

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
