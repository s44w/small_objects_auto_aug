from src.data.yolo_label_reader import (
    parse_yolo_label_line,
    yolo_bbox_area_px,
    yolo_bbox_to_xywh_px,
)


def test_parse_yolo_line_and_area_conversion():
    bbox = parse_yolo_label_line("3 0.5 0.5 0.2 0.4")
    assert bbox.class_id == 3
    assert bbox.x_center == 0.5
    assert bbox.width == 0.2

    x, y, w, h = yolo_bbox_to_xywh_px(bbox, image_width=200, image_height=100)
    assert abs(w - 40.0) < 1e-9
    assert abs(h - 40.0) < 1e-9
    assert abs(x - 80.0) < 1e-9
    assert abs(y - 30.0) < 1e-9

    area = yolo_bbox_area_px(bbox, image_width=200, image_height=100)
    assert abs(area - 1600.0) < 1e-9

