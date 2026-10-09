"""CLI contracts: coordinates, Unicode discovery, negatives and input failures."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def infer():
    return importlib.import_module("infer") if importlib.util.find_spec("infer") else None


def test_json_and_yolo_share_absolute_clipped_boxes(infer):
    assert infer is not None, "Inference CLI is missing"
    import torch
    boxes = SimpleNamespace(xyxy=torch.tensor([[-2., 20., 52., 80.]]),
                            conf=torch.tensor([.9]), cls=torch.tensor([0.]))
    result = SimpleNamespace(boxes=boxes)
    record = infer.serialize_result(result, "полка.jpg", 100, 100, 12.3)
    assert record["image"] == "полка.jpg" and record["count"] == 1
    assert record["detections"][0]["bbox"] == [0., 20., 52., 80.]
    assert record["detections"][0]["label"] == "DENA"
    assert infer.yolo_text(record, 100, 100).split() == ["0", "0.260000", "0.500000", "0.520000", "0.600000"]


def test_negative_is_empty_txt_and_json_count_zero(infer):
    assert infer is not None, "Inference CLI is missing"
    record = infer.serialize_result(SimpleNamespace(boxes=None), "negative.jpg", 10, 10, 1.)
    assert record["count"] == 0 and record["detections"] == []
    assert infer.yolo_text(record, 10, 10) == ""


def test_directory_discovery_is_sorted_recursive_and_unicode_safe(infer, tmp_path):
    assert infer is not None, "Inference CLI is missing"
    (tmp_path / "магазин").mkdir()
    for name in ["z.JPG", "a.png", "магазин/video.mp4", "annotations.json"]:
        (tmp_path / name).write_bytes(b"placeholder")
    paths = infer.discover_sources(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == ["a.png", "z.JPG", "магазин/video.mp4"]


def test_duplicate_output_stems_are_rejected_before_writing(infer, tmp_path):
    assert infer is not None, "Inference CLI is missing"
    for suffix in ["jpg", "png"]:
        (tmp_path / f"photo.{suffix}").write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="collision"):
        infer.plan_outputs(infer.discover_sources(tmp_path), tmp_path, tmp_path / "out")


def test_corrupt_image_is_reported(infer, tmp_path):
    assert infer is not None, "Inference CLI is missing"
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        infer.read_bgr(path)


@pytest.mark.parametrize("flag,value", [("--conf", "nan"), ("--iou", "1.2"), ("--imgsz", "0")])
def test_invalid_arguments_fail_before_model_load(infer, flag, value):
    assert infer is not None, "Inference CLI is missing"
    with pytest.raises(SystemExit) as exc:
        infer.build_parser().parse_args(["--source", "photo.jpg", flag, value])
    assert exc.value.code == 2


def test_nonfinite_model_output_is_refused(infer):
    assert infer is not None, "Inference CLI is missing"
    import torch
    boxes = SimpleNamespace(xyxy=torch.tensor([[0., 0., float("nan"), 5.]]),
                            conf=torch.tensor([.9]), cls=torch.tensor([0.]))
    with pytest.raises(ValueError, match="finite"):
        infer.serialize_result(SimpleNamespace(boxes=boxes), "photo.jpg", 10, 10, 1.)
