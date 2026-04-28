# References Used as MVP Basis

The implementation follows `instructions.md` and the official documentation below:

Project documentation:

- Dataset analytics: `docs/DATASET_ANALYTICS.md`
- Augmentation policy: `docs/AUGMENTATION_POLICY.md`
- Thresholds and calibration: `docs/THRESHOLDS.md`

Primary sources:

- Ultralytics augmentations and train arguments:
  - https://docs.ultralytics.com/modes/train/
  - https://docs.ultralytics.com/guides/yolo-data-augmentation/
- Ultralytics dataset page for VisDrone:
  - https://docs.ultralytics.com/datasets/detect/visdrone/
- Albumentations bbox handling:
  - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
- pycocotools COCOeval:
  - https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
- COCO benchmark paper:
  - https://arxiv.org/abs/1405.0312
- VisDrone benchmark paper:
  - https://arxiv.org/abs/1804.07437
- AutoAugment paper:
  - https://arxiv.org/abs/1805.09501
- Scale-Aware AutoAugment for Object Detection:
  - https://arxiv.org/abs/2103.16119
- Select-Mosaic (dense small objects):
  - https://www.mdpi.com/1424-8220/23/18/7749
