#!/usr/bin/env python3
"""Build and verify the portable DENA v1.0 inference release (stdlib only)."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("dena.release")
ARTIFACTS = (
    "runs/detect/dena_baseline/weights/best.pt",
    "AGENTS.md", "DEPLOYMENT_AGENT_INSTRUCTIONS.md", "README.md", "requirements.txt",
    "infer.py", "app.py", "configs/baseline.yaml", "configs/data.yaml", "manifest.json",
    "runs/detect/dena_baseline/training_summary.json", "runs/detect/dena_baseline/results.csv",
    "outputs/benchmark_eval/report.md", "outputs/benchmark_eval/evaluation.json",
    "outputs/benchmark_eval/metrics.csv", "outputs/benchmark_eval/visual_review.json",
    "outputs/benchmark_eval/5303_comparison.png", "outputs/benchmark_eval/4647_comparison.png",
    "outputs/benchmark_eval/5140_comparison.png", "outputs/benchmark_eval/9615_comparison.png",
    "outputs/benchmark_eval/7663_comparison.png",
    "scripts/setup_dataset.py", "scripts/dataset_utils.py", "scripts/convert_anylabeling_to_yolo.py",
    "scripts/train_baseline.py", "scripts/eval_benchmarks.py", "scripts/package_deliverables.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_readme(training, evaluation):
    if training.get("status") != "completed":
        raise ValueError("Release requires successfully completed training")
    metrics = training["test_metrics"]
    if any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1
           for k in ("mAP50", "mAP50_95", "precision", "recall")):
        raise ValueError("Training metrics must be finite probabilities")
    images = evaluation["images"]
    if not images:
        raise ValueError("Release requires benchmark evaluation records")
    tp = sum(row["TP"] for row in images)
    fp = sum(row["FP"] for row in images)
    fn = sum(row["FN"] for row in images)
    recall = tp/(tp+fn) if tp+fn else 0.
    precision = tp/(tp+fp) if tp+fp else 0.
    return f"""# DENA Object Detection PoC — Release v1.0

Single-class YOLOv8s retail detector: class `0: DENA`. Weights, local Gradio UI,
headless image/directory/video inference, deployment recipes and evaluation evidence
are included. Extract the ZIP to obtain the `dena_poc/` project directory.

| Measurement | Result | Evaluation method |
| --- | ---: | --- |
| mAP50 | {metrics['mAP50']:.1%} | Ultralytics test validator |
| mAP50–95 | {metrics['mAP50_95']:.1%} | Ultralytics test validator |
| Recall | {recall:.1%} | All test images, conf 0.25 / NMS IoU 0.45 / match IoU 0.50 |
| Precision | {precision:.1%} | Same fixed-threshold evaluation |
| Validator recall | {metrics['recall']:.1%} | Validator-selected operating point |
| Validator precision | {metrics['precision']:.1%} | Validator-selected operating point |

Fixed-threshold counts: {tp} TP, {fp} FP, {fn} FN. The recall of {recall:.1%}
and mAP50 of {metrics['mAP50']:.1%} come from different evaluation methods.
See `outputs/benchmark_eval/report.md` for per-case results and annotation concerns.
Angled carton localization remains a failure mode; audit partial boxes and probable
annotation gaps before changing thresholds or adding SAHI/CLAHE.

## Launch from the extracted project's parent directory

Use Python 3.11 for the pinned environment. Bootstrap and background/service
commands for Windows and Linux are in `AGENTS.md` (identical to
`DEPLOYMENT_AGENT_INSTRUCTIONS.md`).

```powershell
py -3.11 -m venv .\\dena_poc\\.venv
& .\\dena_poc\\.venv\\Scripts\\python.exe -m pip install -r .\\dena_poc\\requirements.txt
& .\\dena_poc\\.venv\\Scripts\\python.exe .\\dena_poc\\infer.py --source .\\photos --device 0 --save-json --save-txt
& .\\dena_poc\\.venv\\Scripts\\python.exe .\\dena_poc\\app.py
```

UI: http://127.0.0.1:7860, public sharing disabled. Use `--device cpu` for
headless inference without CUDA. Image JSON contains basename, count, DENA labels,
confidence, absolute pixel `[x1,y1,x2,y2]` boxes and `inference_ms`. TXT exports are
normalized predictions; they are not reviewed ground truth. Video has an annotated
MP4 and optional per-frame JSON/TXT; frame indices start at 0. Counts across frames
are repeated detections, not distinct physical packages.

## Included evidence and exclusions

Five locked benchmark comparison PNGs, dataset manifest, complete training summary,
results CSV, baseline/data configs and tooling are included.
Dataset images and labels are not included. The raw archive, virtualenv, uploads, pretrained initialization
weights and last.pt are also excluded. Inference and the UI need only best.pt;
retraining requires restoring the reviewed dataset and rebasing `configs/data.yaml`
as described in the agent instructions. Training is never triggered by packaging.

`release_manifest.json` lists every payload's archive path, byte size and SHA-256.
The manifest itself is excluded from its own file list. Archive CRCs and all payload
hashes are verified before publication. Re-run `scripts/package_deliverables.py`
to regenerate the bundle after authorized changes.
"""


def verify_archive(archive):
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt:
            raise ValueError(f"Archive CRC failed: {corrupt}")
        manifest = json.loads(bundle.read("dena_poc/release_manifest.json"))
        listed = {row["archive_path"] for row in manifest["files"]}
        names = bundle.namelist()
        if len(names) != len(set(names)) or listed != set(names) - {"dena_poc/release_manifest.json"}:
            raise ValueError("Archive contents do not match release manifest")
        for row in manifest["files"]:
            data = bundle.read(row["archive_path"])
            if len(data) != row["bytes"] or digest(data) != row["sha256"]:
                raise ValueError(f"Archive checksum failed: {row['archive_path']}")
    return manifest


def package_release(root=ROOT):
    root = Path(root).absolute()
    payloads = {}
    # Preflight and snapshot every required artifact before writing a release.
    for relative in ARTIFACTS:
        path = root / relative
        if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Release artifacts must not traverse links: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing required release artifact: {path}")
        data = path.read_bytes()
        if not data:
            raise ValueError(f"Empty release artifact: {path}")
        payloads[relative] = data
    if payloads["AGENTS.md"] != payloads["DEPLOYMENT_AGENT_INSTRUCTIONS.md"]:
        raise ValueError("Deployment instruction copies must be identical")
    training = json.loads(payloads["runs/detect/dena_baseline/training_summary.json"])
    evaluation = json.loads(payloads["outputs/benchmark_eval/evaluation.json"])
    for key, relative in (("weights_sha256", "runs/detect/dena_baseline/weights/best.pt"),
                          ("manifest_sha256", "manifest.json")):
        if key in evaluation and evaluation[key] != digest(payloads[relative]):
            raise ValueError(f"Benchmark evidence does not match {relative}")
    readme = build_readme(training, evaluation).encode("utf-8")
    payloads["README_DELIVERABLE.md"] = readme
    files = [{"path": relative, "archive_path": f"dena_poc/{relative}",
              "bytes": len(data), "sha256": digest(data)} for relative, data in sorted(payloads.items())]
    manifest = {"schema_version": 1, "release": "DENA_PoC", "version": "1.0",
                "class_names": {"0": "DENA"}, "dataset_included": False,
                "defaults": {"conf": .25, "iou": .45, "imgsz": 1024}, "files": files}
    manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    payloads["release_manifest.json"] = manifest_data
    archive = root / "outputs/DENA_PoC_Release_v1.0.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", dir=archive.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for relative, data in sorted(payloads.items()):
                # Stable timestamps and permissions make identical rebuilds byte-identical.
                info = zipfile.ZipInfo(f"dena_poc/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, data, compresslevel=6)
        verify_archive(temporary)
        atomic_write(root / "README_DELIVERABLE.md", readme)
        atomic_write(root / "release_manifest.json", manifest_data)
        os.replace(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    LOG.info("CRC and SHA-256 verified for every release payload")
    for row in files:
        LOG.info("%s (%d bytes)", row["archive_path"], row["bytes"])
    LOG.info("dena_poc/release_manifest.json (%d bytes)", len(manifest_data))
    LOG.info("Archive: %s; %d files; %d bytes (%.2f MiB); SHA-256 %s",
             archive, len(payloads), archive.stat().st_size, archive.stat().st_size/2**20,
             digest(archive.read_bytes()))
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        package_release(args.root)
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        LOG.error("Packaging failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
