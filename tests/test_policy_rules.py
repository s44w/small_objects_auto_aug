from src.policy.rule_engine import RuleEngineConfig, generate_policy_from_stats


def _mock_stats(
    small_ratio: float,
    objects_per_image: float,
    objects_per_mpix: float,
    illum_v_std: float,
    imbalance_ratio: float,
    imbalance_ratio_small: float,
):
    return {
        "splits": {
            "train": {
                "ratios": {"small_ratio": small_ratio, "tiny_ratio": 0.2},
                "density": {
                    "objects_per_image": {"mean": objects_per_image},
                    "objects_per_mpix": {"mean": objects_per_mpix},
                },
                "illumination": {"v_std": {"mean": illum_v_std}},
                "class_distribution": {
                    "imbalance_ratio": imbalance_ratio,
                    "imbalance_ratio_small": imbalance_ratio_small,
                    "small_counts": {"0": 20, "1": 3, "2": 2, "3": 1},
                },
            }
        }
    }


def test_rule_engine_applies_dense_and_small_safe_rules():
    stats = _mock_stats(
        small_ratio=0.8,
        objects_per_image=20.0,
        objects_per_mpix=40.0,
        illum_v_std=50.0,
        imbalance_ratio=12.0,
        imbalance_ratio_small=8.0,
    )
    cfg = RuleEngineConfig()
    policy, report = generate_policy_from_stats(stats, cfg=cfg)

    ultra = policy["ultralytics_args"]
    flags = report["flags"]
    assert flags["is_dense"] is True
    assert flags["is_small_heavy"] is True
    assert ultra["mosaic"] == 0.7
    assert ultra["degrees"] <= 5.0
    assert ultra["translate"] <= 0.05
    assert ultra["scale"] <= 0.30
    copy_paste = next(item for item in policy["albumentations_spec"] if item["name"] == "BBoxCopyPaste")
    assert copy_paste["params"]["tail_class_ids"] == [3, 2]
    assert any(rule["rule_name"] == "R_mosaic" for rule in report["fired_rules"])
