# Augmentation Policy

This document describes how dataset statistics are converted into augmentation
settings for small-object detection.

## Design Goal

The project uses an interpretable rule-based policy engine instead of an opaque
augmentation search. The rule engine reads `dataset_stats.json`, computes a small
set of dataset flags, and writes:

- `policy_adaptive.json`: full policy, including custom augmentation spec.
- `policy_adaptive.yaml`: Ultralytics-compatible scalar arguments.
- `decision_report.json`: fired rules and parameter changes.

The key contract is:

```text
dataset statistics -> flags -> fired rules -> policy parameters
```

## Policy Structure

`policy_adaptive.json` has this structure:

```json
{
  "policy_name": "adaptive_policy_mvp",
  "ultralytics_args": {},
  "albumentations_spec": [],
  "metadata": {
    "features": {},
    "flags": {},
    "tail_class_ids_small": []
  }
}
```

`policy_adaptive.yaml` contains only scalar Ultralytics training arguments. Custom
Albumentations transforms are Python objects and therefore cannot be represented
fully in YAML.

## Ultralytics Arguments

The project whitelists the following detection-compatible arguments:

| Parameter | Purpose |
|---|---|
| `mosaic` | Combines multiple images; useful for diversity and dense scenes. |
| `close_mosaic` | Disables mosaic during final epochs. |
| `hsv_h`, `hsv_s`, `hsv_v` | Color/illumination augmentation. |
| `degrees` | Rotation range. |
| `translate` | Translation range. |
| `scale` | Scale augmentation. |
| `perspective` | Perspective transform strength. |
| `fliplr`, `flipud` | Horizontal/vertical flip probabilities. |
| `mixup`, `cutmix` | Image mixing augmentations. |

`rect` and `multi_scale` are disabled in the default MVP configuration for
stability and reproducibility.

## Custom Augmentations

Custom augmentations are specified in `albumentations_spec` and instantiated only
through the Python API.

### BBoxAwareCrop

Purpose:

- create crops while preserving bbox consistency;
- avoid samples where all boxes disappear;
- focus training on small-object regions.

Important parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `p` | `0.60` when small-heavy, else `0.30` | Probability of applying crop. |
| `height`, `width` | `640`, `640` | Crop output size. |
| `min_visibility` | `0.30` | Minimum visible bbox fraction after crop. |
| `min_area` | `16.0` | Minimum bbox area in pixels after crop. |

If a crop removes all boxes, the transform falls back to the original sample.

### BBoxCopyPaste

Purpose:

- increase representation of small/tiny objects;
- oversample tail classes in small-object imbalance cases;
- add controlled object diversity without relying on segmentation masks.

Important parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `p` | `0.30` if small-imbalanced, else `0.10` | Probability of applying copy-paste. |
| `ioa_threshold` | `0.30` | Maximum allowed overlap by intersection-over-area. |
| `max_pastes` | `3` | Maximum inserted objects per image. |
| `prefer_small` | depends on small imbalance | Prefer small object donors. |
| `tail_class_ids` | computed | Small-object tail classes. |

When `policy.use_object_bank=true`, an object bank is built from train images and
used as a donor source for copy-paste.

## Rules

### R_mosaic

Condition:

```text
is_dense = objects_per_image_mean >= 15 OR objects_per_mpix_mean >= 30
```

Action:

```text
if is_dense: mosaic = 0.7
else:        mosaic = 0.3
```

Rationale:

Dense scenes benefit from moderate mosaic because it exposes the model to varied
layouts. The value is capped below 1.0 to avoid making already small objects too
small too often.

### R_geom_small_safe

Condition:

```text
is_small_heavy = small_ratio >= 0.5
```

Action:

```text
degrees <= 5
translate <= 0.05
scale <= 0.30
perspective = 0
```

Rationale:

Small objects are sensitive to aggressive geometric transformations. Strong
scaling, rotation, translation, and perspective can push them below the effective
resolution of the detector.

### R_flip

Condition:

```text
allow_flipud
```

Action:

```text
fliplr = 0.5
flipud = 0.1 if allow_flipud else 0.0
```

Rationale:

Horizontal flips are generally safe for VisDrone-like detection. Vertical flips
are disabled by default because aerial/drone imagery can still have semantic
orientation assumptions, and many real-world objects are not vertically symmetric
in context.

### R_photo

Condition:

```text
is_illum_var_high = illum_v_std_mean >= 35
```

Action:

```text
hsv_s += 0.10
hsv_v += 0.10
```

Rationale:

High illumination variability suggests that color and brightness robustness are
useful. The increase is intentionally modest to avoid unrealistic images.

### R_mixup_cutmix

Condition:

```text
enable_mixup_cutmix = true
is_low_variability = illum_v_std_mean <= 14
is_small_heavy = false
```

Action:

```text
mixup = 0.10
cutmix = 0.10
```

Rationale:

Mixing augmentations are disabled by default for small-heavy datasets because
they can obscure tiny objects and complicate labels. They can be enabled for less
small-object-heavy, low-variability datasets as an experimental option.

## Decision Report

`decision_report.json` records:

- extracted features;
- computed flags;
- fired rules;
- changed parameters;
- before/after values.

Example item:

```json
{
  "rule_name": "R_geom_small_safe",
  "conditions": {"is_small_heavy": true},
  "parameter": "scale",
  "before": 0.5,
  "after": 0.3
}
```

This file is the main explainability artifact and should be included in final
experiment reports.

## Baseline Fairness

Ultralytics may apply default Albumentations transforms when Albumentations is
installed. The project therefore includes:

```yaml
training:
  baseline_disable_albumentations: true
```

This prevents the baseline from receiving an uncontrolled augmentation advantage.

## Ablations

The default MVP suite includes:

| Run | Purpose |
|---|---|
| `baseline` | Ultralytics-like baseline. |
| `manual` | Static small-object policy. |
| `adaptive` | Rule-generated policy. |
| `adaptive_no_mosaic` | Measures mosaic contribution. |
| `adaptive_no_custom_albu` | Measures custom crop/copy-paste contribution. |

For the final diploma experiments, add group ablations:

- adaptive without density rules;
- adaptive without small-object geometry rules;
- adaptive without illumination rules;
- adaptive without imbalance/copy-paste rules.

