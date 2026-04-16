from src.analysis.stats_schema import area_bucket, is_tiny


def test_area_bins_follow_coco_ranges():
    assert area_bucket(10.0) == "small"
    assert area_bucket(32.0**2) == "small"
    assert area_bucket((32.0**2) + 1) == "medium"
    assert area_bucket(96.0**2) == "medium"
    assert area_bucket((96.0**2) + 1) == "large"


def test_tiny_range_is_subset_of_small():
    assert is_tiny(1.0)
    assert is_tiny(16.0**2)
    assert not is_tiny((16.0**2) + 1.0)

