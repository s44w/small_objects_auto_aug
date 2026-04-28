# Dataset Analytics

This document describes the dataset statistics used by the adaptive augmentation
pipeline. The goal is to make augmentation selection interpretable: every policy
change should be traceable to measurable dataset properties.

## Scope

The analyzer works with YOLO detection datasets:

```text
dataset_root/
  images/train/*.jpg
  labels/train/*.txt
  images/val/*.jpg
  labels/val/*.txt
```

Labels use normalized YOLO format:

```text
class_id x_center y_center width height
```

All object sizes used for analysis are converted back to **pixels of the original
image**, not resized training tensors.

## Outputs

The analyzer writes:

- `artifacts/stats/dataset_stats.json`: structured statistics used by the rule engine.
- `artifacts/stats/dataset_stats.csv`: flattened table for inspection.
- `artifacts/stats/plots/`: optional quick diagnostic plots.
- `artifacts/stats/validation_report.json`: image/label structure validation.

For VisDrone raw annotations, the pipeline can also write:

- `artifacts/stats/visdrone_scene_difficulty.json`: truncation and occlusion summary.

## Area Bins

The project follows COCO-style object area bins:

| Bin | Area in pixels |
|---|---:|
| `tiny` | `area <= 16^2`, project-specific extension |
| `small` | `area <= 32^2` |
| `medium` | `32^2 < area <= 96^2` |
| `large` | `area > 96^2` |

`AP_small` in evaluation is the standard COCOeval small-object metric. `AP_tiny`
is a non-standard project extension and must be reported as such.

## Current Statistics

### Dataset Size

Per split:

- `num_images`
- `num_label_files`
- `num_objects`
- `empty_labels_count`

These fields verify that the subset being analyzed is meaningful and that empty
labels do not dominate the run.

### Bounding Box Area

Field:

```text
splits.<split>.area_px2
```

Contains:

- `min`, `max`, `mean`, `median`
- `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`

This is the main scale descriptor for small-object detection. The rule engine
currently consumes `small_ratio` and `tiny_ratio`, while the percentile fields
are intended for reporting and future threshold calibration.

### Object Scale Ratios

Field:

```text
splits.<split>.ratios
```

Contains:

- `small_ratio`
- `medium_ratio`
- `large_ratio`
- `tiny_ratio`

Definitions:

- `small_ratio = num_small_objects / num_objects`
- `tiny_ratio = num_tiny_objects / num_objects`

`small_ratio` is the main signal for small-object-heavy datasets. In the default
configuration, `small_ratio >= 0.5` activates small-safe geometric rules.

### Density

Field:

```text
splits.<split>.density
```

Contains descriptive statistics for:

- `objects_per_image`
- `objects_per_mpix`

`objects_per_mpix` normalizes density by image area:

```text
objects_per_mpix = object_count / ((image_width * image_height) / 1e6)
```

The default dense-scene rule is activated when either:

```text
objects_per_image_mean >= 15
objects_per_mpix_mean >= 30
```

### Class Distribution

Field:

```text
splits.<split>.class_distribution
```

Contains:

- `counts`: object count per class.
- `small_counts`: small-object count per class.
- `imbalance_ratio`: `max_count / min_nonzero_count`.
- `imbalance_ratio_small`: same ratio using only small objects.

The policy engine uses small-class imbalance to select tail classes for
bbox-copy-paste.

### Image Size

Field:

```text
splits.<split>.image_size
```

Contains descriptive statistics for:

- `width`
- `height`
- `aspect_ratio`

These fields are useful for reporting and for deciding whether tiling/chips are
appropriate for large overhead datasets.

### Illumination

Field:

```text
splits.<split>.illumination
```

Contains descriptive statistics for:

- `v_mean`: mean HSV V channel.
- `v_std`: standard deviation of HSV V channel.
- `contrast`: grayscale standard deviation.

The current photo rule uses `illumination.v_std.mean`. By default:

```text
illum_v_std_mean >= 35 -> increase hsv_s and hsv_v
```

## Validation Report

The validation report checks:

- missing labels for images;
- orphan label files;
- empty label files;
- unreadable label files;
- image and label counts by split.

The dataset is marked invalid if images and labels are not paired correctly.

## Recommended Extensions

For a stronger diploma-level analysis, add these statistics:

| Statistic | Why it matters |
|---|---|
| `bbox_area_ratio = bbox_area / image_area` | Pixel area alone is not scale-normalized across resolutions. |
| `small_ratio_by_class`, `tiny_ratio_by_class` | Enables class-specific copy-paste and sampling. |
| bbox width/height percentiles | Captures object shape and model stride risk. |
| bbox aspect ratio by class | Helps avoid harmful geometry for elongated objects. |
| grid density, for example 4x4 or 8x8 | Detects local crowded regions better than image-level density. |
| bbox overlap/IoA statistics | Prevents too-aggressive copy-paste in already crowded scenes. |
| edge-object ratio | Controls crop/translate when many objects touch image borders. |
| blur/sharpness via Laplacian variance | Supports blur/noise/photo augmentation rules. |
| train/val shift statistics | Explains unstable validation metrics and dataset mismatch. |
| annotation quality report | Tracks invalid, duplicate, clipped, or extremely small boxes. |

## Train/Val Shift Recommendation

For final experiments, compare train and val distributions:

- class count divergence;
- small/tiny ratio difference;
- density difference;
- illumination difference;
- bbox area percentile difference.

This should be reported before interpreting AP changes. If train and val differ
strongly, policy effects can be confounded by dataset shift.

