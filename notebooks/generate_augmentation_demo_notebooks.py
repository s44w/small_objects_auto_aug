from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


DATASETS = [
    {
        "key": "visdrone",
        "title": "VisDrone Augmentation Demo",
        "dataset_dir": "visdrone_yolo",
        "config_rel": "configs/project_config.yaml",
        "notebook_name": "visdrone_augmentation_demo_colab.ipynb",
        "dataset_label": "VisDrone YOLO",
    },
    {
        "key": "dota",
        "title": "DOTA Augmentation Demo",
        "dataset_dir": "dota_yolo",
        "config_rel": "configs/dota_config.yaml",
        "notebook_name": "dota_augmentation_demo_colab.ipynb",
        "dataset_label": "DOTA YOLO",
    },
    {
        "key": "coco_small",
        "title": "COCO-small Augmentation Demo",
        "dataset_dir": "coco_small_yolo",
        "config_rel": "configs/coco_small_config.yaml",
        "notebook_name": "coco_small_augmentation_demo_colab.ipynb",
        "dataset_label": "COCO-small YOLO",
    },
]


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def build_cells(meta: dict) -> list[dict]:
    title = meta["title"]
    dataset_dir = meta["dataset_dir"]
    config_rel = meta["config_rel"]
    dataset_key = meta["key"]
    dataset_label = meta["dataset_label"]

    return [
        md_cell(
            f"""
            # {title}

            This notebook demonstrates how `baseline`, `manual`, and `adaptive`
            augmentation modes behave on the **{dataset_label}** dataset.

            Included sections:
            - dataset structure analysis and small-object statistics;
            - adaptive policy generation from dataset metrics;
            - active augmentation summary for all modes;
            - several `original / baseline / manual / adaptive` image examples;
            - one step-by-step adaptive example.

            The visualization is intentionally deterministic: we apply
            **representative values** of all active transforms so the effect
            of each mode is clearly visible in the rendered examples.
            """
        ),
        code_cell(
            """
            %pip -q install ultralytics albumentations opencv-python pyyaml pandas matplotlib tqdm
            """
        ),
        code_cell(
            """
            import os
            import subprocess
            import sys
            from pathlib import Path

            REPO_URL = 'https://github.com/s44w/small_objects_auto_aug.git'
            PROJECT_ROOT = Path('/content/small_objects_auto_aug')

            if not PROJECT_ROOT.exists():
                subprocess.run(['git', 'clone', REPO_URL, str(PROJECT_ROOT)], check=True)

            os.chdir(PROJECT_ROOT)
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            print('PROJECT_ROOT =', PROJECT_ROOT)
            """
        ),
        code_cell(
            f"""
            from pathlib import Path
            import shutil

            from google.colab import drive

            drive.mount('/content/drive', force_remount=False)

            DRIVE_ROOT = Path('/content/drive/MyDrive')
            SOURCE_DATASET_ROOT = DRIVE_ROOT / 'datasets' / '{dataset_dir}'
            LOCAL_DATASET_ROOT = Path('/content/datasets') / '{dataset_dir}'
            OUTPUT_ROOT = DRIVE_ROOT / 'experiments' / 'augmentation_demos' / '{dataset_key}'
            ARTIFACT_ROOT = OUTPUT_ROOT / 'artifacts'
            FIGURE_ROOT = OUTPUT_ROOT / 'figures'

            USE_LOCAL_RUNTIME_COPY = True
            FORCE_RECOPY = False
            IMAGE_EXTS = {{'.jpg', '.jpeg', '.png', '.bmp'}}

            ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
            FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

            def _count_images(images_dir: Path) -> int:
                if not images_dir.exists():
                    return 0
                return sum(1 for path in images_dir.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTS)

            def _is_ready_yolo(root: Path) -> bool:
                req = [
                    root / 'images' / 'train',
                    root / 'images' / 'val',
                    root / 'labels' / 'train',
                    root / 'labels' / 'val',
                ]
                if not all(path.exists() for path in req):
                    return False
                return _count_images(root / 'images' / 'train') > 0 and _count_images(root / 'images' / 'val') > 0

            def _copy_required_yolo_tree(src_root: Path, dst_root: Path) -> None:
                required_dirs = [
                    ('images', 'train'),
                    ('images', 'val'),
                    ('labels', 'train'),
                    ('labels', 'val'),
                ]

                if dst_root.exists():
                    shutil.rmtree(dst_root)
                dst_root.mkdir(parents=True, exist_ok=True)

                for group, split in required_dirs:
                    src_dir = src_root / group / split
                    dst_dir = dst_root / group / split
                    if not src_dir.exists():
                        raise FileNotFoundError(f'Missing required source directory: {{src_dir}}')
                    dst_dir.parent.mkdir(parents=True, exist_ok=True)
                    print(f'[INFO] Copy required subtree: {{src_dir}} -> {{dst_dir}}')
                    shutil.copytree(src_dir, dst_dir)

                for pattern in ('*.yaml', '*.yml', '*.json', '*.txt'):
                    for src_file in src_root.glob(pattern):
                        if src_file.is_file():
                            dst_file = dst_root / src_file.name
                            shutil.copy2(src_file, dst_file)

            if not _is_ready_yolo(SOURCE_DATASET_ROOT):
                raise FileNotFoundError(
                    f'Dataset not found or not ready: {{SOURCE_DATASET_ROOT}}. '
                    'Expected YOLO layout with images/{{train,val}} and labels/{{train,val}}.'
                )

            TARGET_DATASET_ROOT = SOURCE_DATASET_ROOT
            if USE_LOCAL_RUNTIME_COPY:
                if FORCE_RECOPY and LOCAL_DATASET_ROOT.exists():
                    shutil.rmtree(LOCAL_DATASET_ROOT)
                if not _is_ready_yolo(LOCAL_DATASET_ROOT):
                    print(f'[INFO] Copy dataset to local runtime: {{SOURCE_DATASET_ROOT}} -> {{LOCAL_DATASET_ROOT}}')
                    _copy_required_yolo_tree(SOURCE_DATASET_ROOT, LOCAL_DATASET_ROOT)
                TARGET_DATASET_ROOT = LOCAL_DATASET_ROOT

            print('SOURCE_DATASET_ROOT =', SOURCE_DATASET_ROOT)
            print('TARGET_DATASET_ROOT =', TARGET_DATASET_ROOT)
            print('train images =', _count_images(TARGET_DATASET_ROOT / 'images' / 'train'))
            print('val images =', _count_images(TARGET_DATASET_ROOT / 'images' / 'val'))
            print('OUTPUT_ROOT =', OUTPUT_ROOT)
            """
        ),
        code_cell(
            f"""
            import json
            import matplotlib.pyplot as plt
            import pandas as pd

            from src.analysis.dataset_analyzer import DatasetAnalyzerConfig, analyze_dataset
            from src.experiments.augmentation_demo import (
                apply_demo_mode,
                apply_demo_scalar_policy,
                build_object_bank_from_dataset,
                draw_sample,
                load_split_samples,
                load_yolo_sample,
                mode_summary_rows,
                pick_demo_image_paths,
                sample_object_summary,
            )
            from src.augmentation.policy_to_ultralytics import AugmentationPolicy
            from src.policy.rule_engine import RuleEngineConfig, generate_policy_from_stats, save_policy_artifacts
            from src.utils.io import load_yaml

            CFG_PATH = PROJECT_ROOT / '{config_rel}'
            cfg = load_yaml(CFG_PATH)
            cfg['dataset']['root'] = str(TARGET_DATASET_ROOT)
            cfg['analysis']['generate_plots'] = True

            class_names = cfg['dataset'].get('class_names') or cfg['dataset'].get('visdrone_classes') or []
            print('num classes =', len(class_names))
            print('first classes =', class_names[:10])
            """
        ),
        code_cell(
            """
            analysis_cfg = DatasetAnalyzerConfig(
                small_max_area=float(cfg['analysis']['small_max_area']),
                medium_max_area=float(cfg['analysis']['medium_max_area']),
                tiny_max_area=float(cfg['analysis']['tiny_max_area']),
                generate_plots=bool(cfg['analysis'].get('generate_plots', True)),
                show_progress=True,
            )

            stats = analyze_dataset(
                dataset_root=TARGET_DATASET_ROOT,
                output_dir=ARTIFACT_ROOT / 'stats',
                config=analysis_cfg,
            )

            split_rows = []
            for split_name, split_stats in stats['splits'].items():
                split_rows.append({
                    'split': split_name,
                    'num_images': split_stats['num_images'],
                    'num_objects': split_stats['num_objects'],
                    'small_ratio': round(split_stats['ratios']['small_ratio'], 4),
                    'tiny_ratio': round(split_stats['ratios']['tiny_ratio'], 4),
                    'objects_per_image_mean': round(split_stats['density']['objects_per_image']['mean'], 3),
                    'objects_per_mpix_mean': round(split_stats['density']['objects_per_mpix']['mean'], 3),
                    'illum_v_std_mean': round(split_stats['illumination']['v_std']['mean'], 3),
                    'imbalance_ratio': round(split_stats['class_distribution']['imbalance_ratio'], 3),
                    'imbalance_ratio_small': round(split_stats['class_distribution']['imbalance_ratio_small'], 3),
                })

            split_df = pd.DataFrame(split_rows)
            split_df
            """
        ),
        code_cell(
            """
            train_counts = pd.Series({
                int(key): int(value)
                for key, value in stats['splits']['train']['class_distribution']['counts'].items()
            }).sort_values(ascending=False)
            train_small_counts = pd.Series({
                int(key): int(value)
                for key, value in stats['splits']['train']['class_distribution']['small_counts'].items()
            }).sort_values(ascending=False)

            def _rename_index(series: pd.Series) -> pd.Series:
                renamed = []
                for idx in series.index:
                    if class_names and 0 <= int(idx) < len(class_names):
                        renamed.append(class_names[int(idx)])
                    else:
                        renamed.append(str(idx))
                out = series.copy()
                out.index = renamed
                return out

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            _rename_index(train_counts.head(10)).sort_values().plot.barh(ax=axes[0], color='#4E79A7')
            axes[0].set_title('Top-10 classes by object count (train)')
            axes[0].set_xlabel('objects')

            _rename_index(train_small_counts.head(10)).sort_values().plot.barh(ax=axes[1], color='#F28E2B')
            axes[1].set_title('Top-10 classes by small-object count (train)')
            axes[1].set_xlabel('small objects')

            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            rule_cfg = RuleEngineConfig.from_project_config(cfg)
            policy_payload, decision_report = generate_policy_from_stats(stats, cfg=rule_cfg)
            saved_paths = save_policy_artifacts(
                policy=policy_payload,
                decision_report=decision_report,
                output_dir=ARTIFACT_ROOT / 'policy',
            )
            adaptive_policy = AugmentationPolicy(payload=policy_payload)

            baseline_args = load_yaml(PROJECT_ROOT / 'configs' / 'baseline.yaml')
            manual_args = load_yaml(PROJECT_ROOT / 'configs' / 'manual.yaml')
            adaptive_args = adaptive_policy.get_ultralytics_train_args()

            print(json.dumps(saved_paths, indent=2, ensure_ascii=False))
            """
        ),
        code_cell(
            """
            feature_df = pd.DataFrame([decision_report['features']]).T.reset_index()
            feature_df.columns = ['feature', 'value']

            flag_df = pd.DataFrame([
                {'flag': key, 'value': value}
                for key, value in decision_report['flags'].items()
            ])

            rules_df = pd.DataFrame(decision_report['fired_rules'])
            if rules_df.empty:
                rules_df = pd.DataFrame(columns=['rule_name', 'parameter', 'before', 'after', 'conditions'])

            mode_df = pd.DataFrame(
                mode_summary_rows(
                    baseline_args=baseline_args,
                    manual_args=manual_args,
                    adaptive_args=adaptive_args,
                    adaptive_spec=adaptive_policy.get_albumentations_spec(),
                )
            )

            print('Adaptive policy features:')
            display(feature_df)
            print('Adaptive policy flags:')
            display(flag_df)
            print('Fired rules:')
            display(rules_df)
            print('Mode summary:')
            display(mode_df)
            """
        ),
        code_cell(
            """
            bank = build_object_bank_from_dataset(
                dataset_root=TARGET_DATASET_ROOT,
                split='train',
                small_max_area=float(cfg['analysis']['small_max_area']),
                tiny_max_area=float(cfg['analysis']['tiny_max_area']),
                max_items_per_class=int(cfg['policy']['object_bank'].get('max_items_per_class', 2000)),
                seed=int(cfg['project'].get('seed', 42)),
            )

            donor_pool = load_split_samples(
                dataset_root=TARGET_DATASET_ROOT,
                split='train',
                limit=48,
            )

            adaptive_custom_transforms = adaptive_policy.get_albumentations_transforms(
                object_bank=bank,
                seed=int(cfg['project'].get('seed', 42)),
            )

            demo_paths = pick_demo_image_paths(
                dataset_root=TARGET_DATASET_ROOT,
                split='train',
                limit=3,
            )

            print('object bank size =', bank.size)
            print('donor pool size =', len(donor_pool))
            print('demo images =', [path.name for path in demo_paths])
            """
        ),
        code_cell(
            """
            labels_dir = TARGET_DATASET_ROOT / 'labels' / 'train'
            mode_builders = [
                ('original', None, None),
                ('baseline', baseline_args, None),
                ('manual', manual_args, None),
                ('adaptive', adaptive_args, adaptive_custom_transforms),
            ]

            fig, axes = plt.subplots(len(demo_paths), len(mode_builders), figsize=(22, 5.8 * max(1, len(demo_paths))))
            if len(demo_paths) == 1:
                axes = [axes]

            for row_idx, image_path in enumerate(demo_paths):
                sample = load_yolo_sample(image_path=image_path, labels_dir=labels_dir)
                for col_idx, (mode_name, mode_args, custom_transforms) in enumerate(mode_builders):
                    if mode_name == 'original':
                        out = sample
                    else:
                        out = apply_demo_mode(
                            sample=sample,
                            scalar_args=mode_args,
                            donor_pool=donor_pool,
                            custom_transforms=custom_transforms,
                            seed=42 + row_idx * 17 + col_idx,
                        )
                    summary = sample_object_summary(
                        sample=out,
                        small_max_area=float(cfg['analysis']['small_max_area']),
                        tiny_max_area=float(cfg['analysis']['tiny_max_area']),
                    )
                    axes[row_idx][col_idx].imshow(draw_sample(out, class_names=class_names))
                    axes[row_idx][col_idx].set_title(
                        f"{mode_name}\\n"
                        f"boxes={{int(summary['num_objects'])}}, "
                        f"small={{int(summary['small_objects'])}}, "
                        f"tiny={{int(summary['tiny_objects'])}}"
                    )
                    axes[row_idx][col_idx].axis('off')

            plt.tight_layout()
            figure_path = FIGURE_ROOT / f'{TARGET_DATASET_ROOT.name}_mode_examples.png'
            plt.savefig(figure_path, dpi=180, bbox_inches='tight')
            plt.show()
            print('saved:', figure_path)
            """
        ),
        code_cell(
            """
            focus_path = demo_paths[0]
            focus_sample = load_yolo_sample(image_path=focus_path, labels_dir=labels_dir)

            adaptive_scalar_only = apply_demo_scalar_policy(
                sample=focus_sample,
                scalar_args=adaptive_args,
                donor_pool=donor_pool,
                seed=123,
            )
            adaptive_full = apply_demo_mode(
                sample=focus_sample,
                scalar_args=adaptive_args,
                donor_pool=donor_pool,
                custom_transforms=adaptive_custom_transforms,
                seed=123,
            )

            focus_modes = [
                ('original', focus_sample),
                ('adaptive scalar stage', adaptive_scalar_only),
                ('adaptive full stage', adaptive_full),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(19, 6))
            for idx, (name, sample_out) in enumerate(focus_modes):
                summary = sample_object_summary(
                    sample=sample_out,
                    small_max_area=float(cfg['analysis']['small_max_area']),
                    tiny_max_area=float(cfg['analysis']['tiny_max_area']),
                )
                axes[idx].imshow(draw_sample(sample_out, class_names=class_names))
                axes[idx].set_title(
                    f"{name}\\n"
                    f"boxes={{int(summary['num_objects'])}}, "
                    f"small={{int(summary['small_objects'])}}, "
                    f"tiny={{int(summary['tiny_objects'])}}"
                )
                axes[idx].axis('off')

            plt.tight_layout()
            figure_path = FIGURE_ROOT / f'{TARGET_DATASET_ROOT.name}_adaptive_step_by_step.png'
            plt.savefig(figure_path, dpi=180, bbox_inches='tight')
            plt.show()
            print('saved:', figure_path)
            """
        ),
        md_cell(
            """
            ## What to look at

            - `baseline` shows the reference scalar policy from `configs/baseline.yaml`;
            - `manual` shows the hand-tuned small-object-friendly scalar policy from `configs/manual.yaml`;
            - `adaptive` shows the rule-generated scalar policy together with custom bbox-aware transforms:
              `BBoxAwareCrop` and `BBoxCopyPaste`.

            Main artifacts saved by the notebook:
            - `artifacts/stats/dataset_stats.json`
            - `artifacts/policy/policy_adaptive.json`
            - `artifacts/policy/decision_report.json`
            - `figures/*mode_examples.png`
            - `figures/*adaptive_step_by_step.png`
            """
        ),
    ]


def build_notebook(meta: dict) -> dict:
    return {
        "cells": build_cells(meta),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "colab": {
                "name": meta["notebook_name"],
                "provenance": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for meta in DATASETS:
        notebook = build_notebook(meta)
        output_path = NOTEBOOKS_DIR / meta["notebook_name"]
        output_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(output_path.as_posix())

    print(json.dumps(generated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
