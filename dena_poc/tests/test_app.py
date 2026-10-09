"""Exercise the UI inference boundary without repeatedly loading GPU weights."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def app():
    return importlib.import_module("app") if importlib.util.find_spec("app") else None


def make_detector(app, monkeypatch, tmp_path, names=None):
    assert app is not None, "Gradio application is missing"
    loads, calls = [], []
    model = SimpleNamespace(names=names if names is not None else {0: "DENA"})
    model.to = lambda device: model
    result = SimpleNamespace(boxes=[object(), object()])
    result.plot = lambda **kwargs: np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    model.predict = lambda image, **kwargs: calls.append((image, kwargs)) or [result]
    monkeypatch.setattr(app, "YOLO", lambda weights: loads.append(weights) or model)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: False)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"test checkpoint")
    return app.DenaDetector(weights), loads, calls


def test_reuses_model_and_returns_rgb_count_and_latency(app, monkeypatch, tmp_path):
    detector, loads, calls = make_detector(app, monkeypatch, tmp_path)
    for _ in range(2):
        image, summary = detector.infer(Image.new("L", (8, 8)), 0.25, 0.45)
        assert image[0, 0].tolist() == [30, 20, 10]
        assert summary["count"] == 2
        assert summary["inference_ms"] >= 0
    assert len(loads) == 1
    assert all(call[0].mode == "RGB" for call in calls)
    assert calls[0][1]["imgsz"] == 1024
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["classes"] == [0]


@pytest.mark.parametrize("conf,iou", [(float("nan"), .45), (.01, .45), (.25, 1.0)])
def test_invalid_thresholds_do_not_reach_model(app, monkeypatch, tmp_path, conf, iou):
    detector, _, calls = make_detector(app, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="threshold"):
        detector.infer(Image.new("RGB", (8, 8)), conf, iou)
    assert calls == []


def test_empty_upload_is_actionable(app, monkeypatch, tmp_path):
    detector, _, calls = make_detector(app, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="image"):
        detector.infer(None, .25, .45)
    assert calls == []


def test_missing_weights_do_not_trigger_download(app, tmp_path):
    assert app is not None, "Gradio application is missing"
    with pytest.raises(FileNotFoundError, match="best.pt"):
        app.DenaDetector(tmp_path / "best.pt")


def test_wrong_class_model_is_rejected(app, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="DENA"):
        make_detector(app, monkeypatch, tmp_path, {0: "person"})


def test_interface_exposes_upload_camera_and_threshold_controls(app, monkeypatch, tmp_path):
    detector, _, _ = make_detector(app, monkeypatch, tmp_path)
    demo = app.build_demo(detector)
    components = demo.get_config_file()["components"]
    images = [c["props"] for c in components if c["type"] == "image"]
    assert any(p["sources"] == ["upload", "webcam"] and p["type"] == "pil" for p in images)
    assert any(p.get("interactive") is False for p in images)
    sliders = [c["props"] for c in components if c["type"] == "slider"]
    assert {(p["minimum"], p["maximum"], p["step"], p["value"]) for p in sliders} == {
        (.05, .95, .05, .25), (.1, .9, .05, .45)}
    demo.close()
