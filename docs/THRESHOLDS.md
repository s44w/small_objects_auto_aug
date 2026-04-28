# Thresholds and Calibration

This document lists the threshold values used by the MVP rule engine and explains
how they should be interpreted, reported, and calibrated.

## Configuration Location

Thresholds are defined in:

```text
configs/project_config.yaml
```

Main sections:

```yaml
analysis:
  small_max_area: 1024.0
  medium_max_area: 9216.0
  tiny_max_area: 256.0

policy:
  small_ratio_threshold: 0.5
  dense_objects_per_image_threshold: 15.0
  dense_objects_per_mpix_threshold: 30.0
  illum_v_std_threshold: 35.0
  imbalance_ratio_threshold: 10.0
  small_imbalance_ratio_threshold: 6.0
  low_variability_vstd_threshold: 14.0
  bbox_crop_min_visibility: 0.30
  bbox_crop_min_area: 16.0
  bbox_copy_paste_ioa_threshold: 0.30
  bbox_copy_paste_max_pastes: 3
```

## Area Thresholds

| Config key | Default | Meaning |
|---|---:|---|
| `analysis.tiny_max_area` | `256` | Project-specific tiny threshold, `16^2`. |
| `analysis.small_max_area` | `1024` | COCO small threshold, `32^2`. |
| `analysis.medium_max_area` | `9216` | COCO medium threshold, `96^2`. |

These thresholds are measured in pixels on the original image.

Important distinction:

- `AP_small` is a standard COCOeval metric.
- `AP_tiny` is a project-specific extension and must be described as non-standard.

## Policy Thresholds

### `small_ratio_threshold`

Default:

```yaml
small_ratio_threshold: 0.5
```

Flag:

```text
is_small_heavy = small_ratio >= 0.5
```

Effect:

- activates small-safe geometry;
- increases crop probability;
- keeps perspective disabled.

Interpretation:

If at least half of objects are small by COCO area, the dataset is treated as
small-object-heavy.

### Density Thresholds

Defaults:

```yaml
dense_objects_per_image_threshold: 15.0
dense_objects_per_mpix_threshold: 30.0
```

Flag:

```text
is_dense =
  objects_per_image_mean >= 15
  OR objects_per_mpix_mean >= 30
```

Effect:

- raises `mosaic` from `0.3` to `0.7`.

Interpretation:

The two density thresholds complement each other. `objects_per_image` captures
absolute crowding; `objects_per_mpix` normalizes for image resolution.

### Illumination Thresholds

Defaults:

```yaml
illum_v_std_threshold: 35.0
low_variability_vstd_threshold: 14.0
```

Flags:

```text
is_illum_var_high = illum_v_std_mean >= 35
is_low_variability = illum_v_std_mean <= 14
```

Effects:

- high illumination variability increases `hsv_s` and `hsv_v`;
- low variability can allow `mixup/cutmix` only when explicitly enabled and the
  dataset is not small-heavy.

Interpretation:

HSV V standard deviation is used as a simple brightness variability proxy.

### Class Imbalance Thresholds

Defaults:

```yaml
imbalance_ratio_threshold: 10.0
small_imbalance_ratio_threshold: 6.0
```

Flags:

```text
is_imbalanced = imbalance_ratio >= 10
is_small_imbalanced = imbalance_ratio_small >= 6
```

Effects:

- `is_small_imbalanced` increases bbox-copy-paste probability;
- small-object tail classes are selected as copy-paste donors.

Current imbalance formula:

```text
imbalance_ratio = max_count / max(1, min_nonzero_count)
```

Recommended extensions:

- entropy;
- Gini coefficient;
- effective number of samples.

### Crop Thresholds

Defaults:

```yaml
bbox_crop_min_visibility: 0.30
bbox_crop_min_area: 16.0
```

Meaning:

- `min_visibility`: bbox is kept only if at least 30% remains visible after crop.
- `min_area`: bbox is kept only if clipped area is at least 16 px².

Interpretation:

The values are conservative enough for tiny/small objects while filtering
degenerate boxes.

### Copy-Paste Thresholds

Defaults:

```yaml
bbox_copy_paste_ioa_threshold: 0.30
bbox_copy_paste_max_pastes: 3
```

Meaning:

- `ioa_threshold`: maximum allowed overlap during placement.
- `max_pastes`: maximum number of pasted objects per image.

Interpretation:

The overlap constraint prevents synthetic scenes from becoming unrealistic or
label-ambiguous.

## Training Thresholds and Switches

Important defaults:

```yaml
training:
  rect: false
  multi_scale: false
  baseline_disable_albumentations: true
  require_custom_augmentations: true
```

Rationale:

- `rect=false` and `multi_scale=false` improve experiment stability.
- `baseline_disable_albumentations=true` keeps baseline controlled.
- `require_custom_augmentations=true` prevents silent adaptive degradation if
  Ultralytics rejects custom Python transforms.

## Calibration Procedure

For MVP, thresholds are fixed. For final diploma work, report a calibration
procedure:

1. Run analyzer on VisDrone train.
2. Inspect `dataset_stats.json`.
3. Run baseline/manual/adaptive with fixed thresholds.
4. Run threshold sensitivity experiments:
   - `small_ratio_threshold`: `0.4`, `0.5`, `0.6`
   - `mosaic` dense value: `0.5`, `0.7`, `0.9`
   - `bbox_copy_paste_ioa_threshold`: `0.2`, `0.3`, `0.4`
   - `bbox_crop_min_visibility`: `0.2`, `0.3`, `0.5`
5. Select defaults that improve `AP_small` without degrading total mAP sharply.
6. Validate with at least 3 seeds.

## Reporting Format

Use this table in the final report:

| Threshold | Values tested | Selected | Reason |
|---|---|---:|---|
| `small_ratio_threshold` | `0.4/0.5/0.6` | `0.5` | Best AP_small/stability tradeoff. |
| `dense_objects_per_image_threshold` | `10/15/20` | `15` | Activates mosaic for crowded scenes. |
| `bbox_crop_min_visibility` | `0.2/0.3/0.5` | `0.3` | Avoids losing most of small objects. |
| `bbox_copy_paste_ioa_threshold` | `0.2/0.3/0.4` | `0.3` | Balances realism and placement success. |

## Guardrails

Recommended safety checks before final training:

- no negative or zero-area boxes after custom transforms;
- no labels outside image bounds;
- no empty batches caused by crop;
- object bank contains enough objects for selected tail classes;
- adaptive policy JSON and YAML are saved with every run;
- `decision_report.json` is included in final artifacts;
- train/val distribution shift is reported.

