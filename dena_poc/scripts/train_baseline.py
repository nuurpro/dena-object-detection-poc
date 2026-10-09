#!/usr/bin/env python3
"""Train the authorized single-class DENA baseline and evaluate best.pt on test.

Ultralytics has no public `accumulate` argument: nbs=8 sets nominal accumulation,
and our trainer pins it to 2 after each warmup adjustment. Windows worker processes
are protected by the main guard. --dry-run checks readiness without loading weights.
"""
import argparse
import csv
import gc
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_settings"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version, settings
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer

from convert_anylabeling_to_yolo import atomic_write
from dataset_utils import contained_path, reject_link_ancestors
from setup_dataset import validate

LOG = logging.getLogger("dena_training")


class MemorySafeDetectionTrainer(DetectionTrainer):
    """Keep training and validation batches at four and accumulation at two."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "val":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        # The upstream trainer doubles both the validation batch and worker count.
        workers = self.args.workers
        self.args.workers = workers // 2
        try:
            return super().get_dataloader(dataset_path, min(batch_size, self.args.batch), rank, mode)
        finally:
            self.args.workers = workers

    def preprocess_batch(self, batch):
        if self.batch_size != 4:
            raise RuntimeError(f"Training batch changed unexpectedly: {self.batch_size}")
        self.accumulate = 2  # This executes after the upstream warmup interpolation.
        # The upstream first-epoch OOM recovery would silently halve the batch.
        # Exhaust its retry allowance so our outer CUDA OOM handler receives it.
        self._oom_retries = 3
        return super().preprocess_batch(batch)


def load_training_config(path):
    with path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("baseline.yaml must be a mapping")
    required = {"model": "yolov8s.pt", "epochs": 50, "imgsz": 1024, "batch": 4,
                "accumulate": 2, "amp": True, "workers": 4, "patience": 12, "seed": 42}
    for name, expected in required.items():
        if config.get(name) != expected or type(config.get(name)) is not type(expected):
            raise ValueError(f"Baseline requires {name}={expected!r}; found {config.get(name)!r}")
    accumulation = config.pop("accumulate")
    if "nbs" in config and config["nbs"] != config["batch"] * accumulation:
        raise ValueError("nbs conflicts with the effective batch size")
    config["nbs"] = config["batch"] * accumulation
    get_cfg(overrides=config)  # Reject unsupported options before downloading weights.
    return config, accumulation


def cuda_usage(device):
    free, total = torch.cuda.mem_get_info(device)
    return {"free_mib": round(free / 2**20, 2), "total_mib": round(total / 2**20, 2),
            "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 2),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 2),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 2)}


def write_summary(directory, summary):
    atomic_write(directory / "training_summary.json",
                 (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def archive_label_caches(root, run_dir):
    """Move generated scan caches out of the strictly validated dataset tree."""
    relative_run = run_dir.relative_to(root).as_posix()
    planned = []
    for split in ("train", "val", "test"):
        source = contained_path(root, f"dataset/labels/{split}.cache")
        if not source.exists():
            continue
        destination = contained_path(root, f"{relative_run}/label_cache/{split}.cache")
        content = source.read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Conflicting label cache archive: {destination}")
        planned.append((source, destination, content))
    for source, destination, content in planned:
        if not destination.exists():
            atomic_write(destination, content)
        if destination.read_bytes() != content or source.read_bytes() != content:
            raise ValueError(f"Label cache changed during archival: {source}")
        source.unlink()
    return [str(destination) for _, destination, _ in planned]


def record_oom(error, directory, device, seconds):
    usage = cuda_usage(device)
    LOG.error("CUDA out of memory after %.1fs. VRAM usage: %s", seconds, usage)
    gc.collect()
    torch.cuda.empty_cache()
    summary = {"status": "cuda_out_of_memory", "error": str(error),
               "duration_seconds": seconds, "gpu_memory": usage,
               "cuda_cache_cleanup": "completed"}
    write_summary(directory, summary)
    return summary


def extract_metrics(metrics):
    values = {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Test metrics must be finite")
    return values


def log_epoch(trainer):
    if not trainer.amp:
        raise RuntimeError("AMP was disabled by Ultralytics; aborting this memory-safe baseline")
    LOG.info("Epoch %d/%d complete; validation metrics=%s; peak VRAM=%s MiB",
             trainer.epoch + 1, trainer.epochs, trainer.metrics,
             round(torch.cuda.max_memory_reserved(trainer.device) / 2**20))


def prepare(args):
    root = args.root.absolute()
    reject_link_ancestors(root)
    root = root.resolve()
    config_path = args.config.resolve() if args.config else contained_path(root, "configs/baseline.yaml")
    config, accumulation = load_training_config(config_path)
    data_path = contained_path(root, "configs/data.yaml")
    with data_path.open(encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("names") != {0: "DENA"}:
        raise ValueError("data.yaml must define only class 0: DENA")
    gate = validate(argparse.Namespace(root=root, require_annotations=True))
    if gate != 0:
        raise ValueError(f"Dataset annotation gate failed with return code {gate}; training refused")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} is not available")
    if args.device < 0:
        raise ValueError("A nonnegative CUDA device index is required")
    torch.cuda.set_device(args.device)
    gpu = torch.cuda.get_device_name(args.device)
    LOG.info("CUDA device %d: %s; memory=%s", args.device, gpu, cuda_usage(args.device))
    project = contained_path(root, config.get("project", "runs/detect"))
    name = args.name or config.get("name", "dena_baseline")
    if not name or Path(name).name != name or name in {".", ".."} or any(c in name for c in "/\\:"):
        raise ValueError("Run name must be a plain directory name")
    run_dir = contained_path(root, (project / name).relative_to(root).as_posix())
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite an existing run: {run_dir}; choose --name for another run")
    config.update(data=str(data_path), device=args.device, project=str(project), name=name,
                  exist_ok=True, save=True, val=True, split="val", cache=False, compile=False)
    return root, config, accumulation, run_dir, gpu


def run(args):
    root, config, accumulation, run_dir, gpu = prepare(args)
    LOG.info("Training plan: batch=%d, accumulate=%d, nominal effective batch=%d, imgsz=%d, AMP=%s, workers=%d",
             config["batch"], accumulation, config["nbs"], config["imgsz"], config["amp"], config["workers"])
    if args.dry_run:
        LOG.info("Dry run passed; no weights downloaded and no training started")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    LOG.addHandler(handler)
    started = time.perf_counter()
    try:
        models = contained_path(root, "models")
        models.mkdir(parents=True, exist_ok=True)
        # Settings are local to YOLO_CONFIG_DIR, not the user's global config.
        settings.update({"weights_dir": str(models), "runs_dir": config["project"]})
        weights = models / config.pop("model")
        torch.cuda.reset_peak_memory_stats(args.device)
        LOG.info("Loading pretrained YOLOv8s from %s", weights)
        model = YOLO(str(weights))
        model.add_callback("on_fit_epoch_end", log_epoch)
        model.train(trainer=MemorySafeDetectionTrainer, **config)
        trained = time.perf_counter()
        actual_run_dir = Path(model.trainer.save_dir).resolve()
        artifacts = {"best_weights": actual_run_dir / "weights/best.pt",
                     "last_weights": actual_run_dir / "weights/last.pt",
                     "results_csv": actual_run_dir / "results.csv"}
        for label, path in artifacts.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing training artifact {label}: {path}")
        with artifacts["results_csv"].open(encoding="utf-8", newline="") as handle:
            completed_epochs = len(list(csv.DictReader(handle)))
        # Release optimizer, EMA and training model before loading best.pt for test.
        model.trainer.optimizer = None
        model.trainer.ema = None
        del model
        gc.collect()
        torch.cuda.empty_cache()
        LOG.info("Evaluating best.pt on the unseen test split")
        best = YOLO(str(artifacts["best_weights"]))
        evaluation = best.val(data=config["data"], split="test", imgsz=1024, batch=4,
                              device=args.device, workers=4, project=str(actual_run_dir),
                              name="test_evaluation", exist_ok=True, plots=True, verbose=True)
        metrics = extract_metrics(evaluation)
        finished = time.perf_counter()
        summary = {"status": "completed", "model": "yolov8s.pt", "gpu": gpu,
                   "torch_version": torch.__version__, "ultralytics_version": ultralytics_version,
                   "epochs_requested": config["epochs"], "epochs_completed": completed_epochs,
                   "training_args": config, "accumulate": accumulation, "effective_batch_size": config["nbs"],
                   "test_split": "test", "test_metrics": metrics,
                   "training_seconds": trained - started, "test_seconds": finished - trained,
                   "duration_seconds": finished - started, "gpu_memory": cuda_usage(args.device),
                   "artifacts": {k: str(v) for k, v in artifacts.items()},
                   "test_artifacts": str(evaluation.save_dir)}
        write_summary(actual_run_dir, summary)
        LOG.info("TEST METRICS: mAP50=%.6f, mAP50-95=%.6f, Precision=%.6f, Recall=%.6f",
                 metrics["mAP50"], metrics["mAP50_95"], metrics["precision"], metrics["recall"])
        LOG.info("Duration: %.1fs training, %.1fs test, %.1fs total; epochs=%d",
                 summary["training_seconds"], summary["test_seconds"], summary["duration_seconds"], completed_epochs)
        for label, path in artifacts.items():
            LOG.info("Confirmed %s: %s (%d bytes)", label, path, path.stat().st_size)
        LOG.info("Summary: %s", actual_run_dir / "training_summary.json")
        return 0
    except torch.cuda.OutOfMemoryError as error:
        record_oom(error, run_dir, args.device, time.perf_counter() - started)
        return 3
    except Exception as error:
        write_summary(run_dir, {"status": "failed", "error": str(error),
                               "duration_seconds": time.perf_counter() - started})
        raise
    finally:
        try:
            archive_label_caches(root, run_dir)
        finally:
            LOG.removeHandler(handler)
            handler.close()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, help="Baseline YAML; default: <root>/configs/baseline.yaml")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--name", help="Optional new run name; existing runs are never overwritten")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.warning("Training interrupted; saved checkpoints remain available")
        return 130
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        LOG.exception("Baseline pipeline failed: %s", error)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
