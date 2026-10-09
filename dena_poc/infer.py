#!/usr/bin/env python3
"""Headless DENA inference for images, recursive directories and video files."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_settings"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

LOG = logging.getLogger("dena.infer")
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg"}


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Threshold must be finite and between 0 and 1")
    return number


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Image resolution must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image, video, or recursive media directory")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/detect/dena_baseline/weights/best.pt")
    parser.add_argument("--conf", type=probability, default=.25)
    parser.add_argument("--iou", type=probability, default=.45)
    parser.add_argument("--imgsz", type=positive_integer, default=1024)
    parser.add_argument("--device", help="Single CUDA index (e.g. 0), or cpu; default: CUDA when available")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/inference")
    parser.add_argument("--save-txt", action="store_true", help="Export predictions as normalized class-0 YOLO labels")
    parser.add_argument("--save-json", action="store_true", help="Export detections, count and latency per image/frame")
    return parser


def discover_sources(source):
    source = Path(source).absolute()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if source.is_file():
        paths = [source] if source.suffix.lower() in IMAGES | VIDEOS else []
    else:
        paths = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGES | VIDEOS),
                       key=lambda p: p.relative_to(source).as_posix())
    if not paths:
        raise ValueError(f"No supported image/video files found: {source}")
    return paths


def plan_outputs(paths, source, output):
    base_dir = source if source.is_dir() else source.parent
    planned, seen = {}, set()
    for path in paths:
        base = output / path.relative_to(base_dir).with_suffix("")
        key = str(base).casefold()
        if key in seen:
            raise ValueError(f"Output filename collision at {base}; rename media with the same stem")
        seen.add(key)
        planned[path] = base
    return planned


def read_bgr(path):
    # imread on Windows cannot reliably open Unicode paths; imdecode can.
    content = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(content, cv2.IMREAD_COLOR) if content.size else None
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def serialize_result(result, name, width, height, latency_ms):
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for box, score, class_id in zip(boxes, scores, classes, strict=True):
            if not all(math.isfinite(v) for v in [*box, score, class_id]):
                raise ValueError("Model outputs must contain finite coordinates and scores")
            if class_id != 0 or not 0 <= score <= 1:
                raise ValueError("Model output violates the class-0 DENA contract")
            x1, y1, x2, y2 = box
            clipped = [min(width, max(0., x1)), min(height, max(0., y1)),
                       min(width, max(0., x2)), min(height, max(0., y2))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError("Model produced a degenerate bounding box")
            detections.append({"label": "DENA", "confidence": float(score), "bbox": clipped})
    return {"image": name, "count": len(detections), "detections": detections,
            "inference_ms": round(latency_ms, 3)}


def yolo_text(record, width, height):
    lines = []
    for detection in record["detections"]:
        x1, y1, x2, y2 = detection["bbox"]
        values = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
        lines.append("0 " + " ".join(f"{v:.6f}" for v in values))
    return "\n".join(lines) + ("\n" if lines else "")


def export_record(base, record, width, height, args):
    if args.save_json:
        atomic_write(Path(str(base) + ".json"), (json.dumps(record, ensure_ascii=False, indent=2,
                                                            allow_nan=False) + "\n").encode("utf-8"))
    if args.save_txt:
        atomic_write(Path(str(base) + ".txt"), yolo_text(record, width, height).encode("utf-8"))


def resolve_device(requested):
    device = requested if requested is not None else ("0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return device
    if not device.isascii() or not device.isdigit():
        raise ValueError("--device must be a single CUDA index (0) or cpu")
    index = int(device)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is not available; use --device cpu")
    return str(index)


def run(args):
    source = args.source.absolute()
    output = args.output_dir.absolute()
    paths = discover_sources(source)
    if source.is_dir() and output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output directory must be outside the source directory to prevent re-ingesting predictions")
    planned = plan_outputs(paths, source, output)
    if not args.weights.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {args.weights}; automatic downloads are disabled")
    device = resolve_device(args.device)
    # Check all images before creating any output or loading model weights.
    for path in paths:
        if path.suffix.lower() in IMAGES:
            read_bgr(path)
    model = YOLO(str(args.weights.resolve())).to("cpu" if device == "cpu" else f"cuda:{device}")
    if model.names != {0: "DENA"}:
        raise ValueError(f"Expected single class 0: DENA, received {model.names}")
    LOG.info("Loaded %s once; device=%s, conf=%.2f, iou=%.2f, imgsz=%d",
             args.weights.resolve(), device, args.conf, args.iou, args.imgsz)
    frames, packages, total_ms, maximum_ms = 0, 0, 0., 0.
    started = perf_counter()

    def predict(image, name, base, metadata=None):
        nonlocal frames, packages, total_ms, maximum_ms
        if device != "cpu":
            torch.cuda.synchronize(int(device))
        tick = perf_counter()
        result = model.predict(image, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                               device=device, classes=[0], max_det=1000, verbose=False)[0]
        if device != "cpu":
            torch.cuda.synchronize(int(device))
        latency_ms = (perf_counter()-tick)*1000
        height, width = image.shape[:2]
        record = serialize_result(result, name, width, height, latency_ms)
        if metadata:
            record.update(metadata)
        export_record(base, record, width, height, args)
        frames += 1
        packages += record["count"]
        total_ms += latency_ms
        maximum_ms = max(maximum_ms, latency_ms)
        LOG.info("%s%s: %d DENA, %.1f ms", name,
                 f" frame {metadata['frame_index']}" if metadata else "", record["count"], latency_ms)
        return result.plot(labels=True, conf=True)

    for path, base in planned.items():
        if path.suffix.lower() in IMAGES:
            annotated = predict(read_bgr(path), path.name, base)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                raise OSError(f"Cannot encode annotated image: {path}")
            atomic_write(Path(str(base) + "_annotated.jpg"), encoded.tobytes())
        else:
            capture = cv2.VideoCapture(str(path))
            writer, temporary_video = None, None
            index = 0
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot decode video: {path}")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    LOG.warning("Unknown frame rate for %s; using 30 fps", path)
                    fps = 30.
                destination = Path(str(base) + "_annotated.mp4")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".mp4", dir=destination.parent, delete=False) as handle:
                    temporary_video = Path(handle.name)
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    height, width = image.shape[:2]
                    if writer is None:
                        writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise OSError("MP4 video encoder is unavailable")
                    frame_base = Path(str(base) + "_frames") / f"frame_{index:06d}"
                    annotated = predict(image, path.name, frame_base,
                                        {"frame_index": index, "timestamp_ms": round(index*1000/fps, 3)})
                    writer.write(annotated)
                    index += 1
                if index == 0:
                    raise ValueError(f"Video contains no decodable frames: {path}")
                writer.release()
                writer = None
                os.replace(temporary_video, destination)
            finally:
                capture.release()
                if writer is not None:
                    writer.release()
                if temporary_video is not None:
                    temporary_video.unlink(missing_ok=True)
    LOG.info("Complete: %d media files, %d images/frames, %d detections; mean %.1f ms, max %.1f ms; wall %.2fs",
             len(paths), frames, packages, total_ms/frames, maximum_ms, perf_counter()-started)
    LOG.info("Outputs: %s (video counts sum per-frame detections, not unique packages)", output)
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(args)
    except torch.cuda.OutOfMemoryError:
        LOG.exception("CUDA out of memory; retry with --device cpu or free GPU memory")
        torch.cuda.empty_cache()
        return 3
    except KeyboardInterrupt:
        LOG.warning("Inference interrupted; completed outputs remain available")
        return 130
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        LOG.error("Inference failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
