"""Test the safety gates and accumulation behavior without downloading weights."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def training():
    return importlib.import_module("train_baseline") if importlib.util.find_spec("train_baseline") else None


def test_configuration_translates_accumulation_to_supported_ultralytics_args(training):
    assert training is not None, "Baseline training script is missing"
    args, accumulation = training.load_training_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml")
    assert accumulation == 2
    assert "accumulate" not in args
    assert args["nbs"] == args["batch"] * accumulation == 8
    assert args["batch"] == 4 and args["imgsz"] == 1024 and args["amp"] is True


@pytest.mark.parametrize("invalid", [{"batch": 8}, {"accumulate": 1}, {"amp": False}, {"imgsz": 640}])
def test_rejects_unsafe_configuration(training, tmp_path, invalid):
    assert training is not None, "Baseline training script is missing"
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/baseline.yaml").read_text())
    config.update(invalid)
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        training.load_training_config(path)


def test_warmup_cannot_change_accumulation_or_allow_silent_oom_reduction(training, monkeypatch):
    assert training is not None, "Baseline training script is missing"
    trainer = training.MemorySafeDetectionTrainer.__new__(training.MemorySafeDetectionTrainer)
    monkeypatch.setattr(training.DetectionTrainer, "preprocess_batch", lambda self, batch: batch)
    trainer.batch_size = 4
    trainer.args = SimpleNamespace(batch=4, nbs=8)
    trainer.accumulate = 1  # Ultralytics warmup rewrites this before preprocessing.
    trainer._oom_retries = 0
    assert trainer.preprocess_batch({"img": "example"}) == {"img": "example"}
    assert trainer.accumulate == 2
    assert trainer._oom_retries == 3


def test_validation_loader_does_not_double_batch_or_worker_count(training, monkeypatch):
    assert training is not None, "Baseline training script is missing"
    trainer = training.MemorySafeDetectionTrainer.__new__(training.MemorySafeDetectionTrainer)
    trainer.args = SimpleNamespace(batch=4, workers=4)
    calls = []
    def loader(self, dataset_path, batch_size, rank, mode):
        calls.append((batch_size, self.args.workers * 2))
        return "loader"
    monkeypatch.setattr(training.DetectionTrainer, "get_dataloader", loader)
    assert trainer.get_dataloader("val", batch_size=8, rank=-1, mode="val") == "loader"
    assert calls == [(4, 4)]
    assert trainer.args.workers == 4


def test_validation_loader_restores_workers_after_error(training, monkeypatch):
    assert training is not None, "Baseline training script is missing"
    trainer = training.MemorySafeDetectionTrainer.__new__(training.MemorySafeDetectionTrainer)
    trainer.args = SimpleNamespace(batch=4, workers=4)
    def fail(*args, **kwargs):
        raise RuntimeError("loader failed")
    monkeypatch.setattr(training.DetectionTrainer, "get_dataloader", fail)
    with pytest.raises(RuntimeError, match="loader failed"):
        trainer.get_dataloader("val", batch_size=8, rank=-1, mode="val")
    assert trainer.args.workers == 4


def test_test_metrics_reject_nan(training):
    assert training is not None, "Baseline training script is missing"
    metrics = SimpleNamespace(box=SimpleNamespace(map50=float("nan"), map=0.1, mp=0.2, mr=0.3))
    with pytest.raises(ValueError, match="finite"):
        training.extract_metrics(metrics)


def test_oom_handler_logs_usage_cleans_cache_and_records_failure(training, monkeypatch, tmp_path):
    assert training is not None, "Baseline training script is missing"
    import torch
    monkeypatch.setattr(training, "cuda_usage", lambda device: {"allocated_mib": 5500})
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    result = training.record_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"), tmp_path, 0, 1.5)
    assert emptied == [True]
    assert result["status"] == "cuda_out_of_memory"
    assert result["gpu_memory"]["allocated_mib"] == 5500
    assert (tmp_path / "training_summary.json").is_file()


def test_generated_label_caches_are_archived_outside_dataset(training, tmp_path):
    assert training is not None
    cache = tmp_path / "dataset/labels/train.cache"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"generated-label-cache")
    run_dir = tmp_path / "runs/detect/example"
    training.archive_label_caches(tmp_path, run_dir)
    assert not cache.exists()
    assert (run_dir / "label_cache/train.cache").read_bytes() == b"generated-label-cache"


def test_label_cache_archive_conflict_preserves_source(training, tmp_path):
    assert training is not None
    cache = tmp_path / "dataset/labels/train.cache"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"new-cache")
    run_dir = tmp_path / "runs/detect/example"
    archived = run_dir / "label_cache/train.cache"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"old-cache")
    with pytest.raises(ValueError, match="Conflicting"):
        training.archive_label_caches(tmp_path, run_dir)
    assert cache.read_bytes() == b"new-cache"
    assert archived.read_bytes() == b"old-cache"
