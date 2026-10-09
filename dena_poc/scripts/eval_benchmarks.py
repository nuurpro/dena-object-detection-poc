#!/usr/bin/env python3
"""Evaluate fixed-confidence DENA detections and produce benchmark diagnostics.

Matching is confidence-ordered, greedy, one-to-one at match_iou=0.50. NMS IoU
is independent (default 0.45). A diagnostic test threshold sweep never changes
the baseline results or annotations; deployment tuning must use validation data.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_settings"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

from convert_anylabeling_to_yolo import atomic_write
from dataset_utils import check_label, contained_path, reject_link_ancestors
from setup_dataset import load_manifest, validate

LOG = logging.getLogger("benchmark_eval")
BENCHMARKS = {"5303": "Small scale / distant shelf", "4647": "Refrigerator glare",
              "5140": "Angled display / occlusion", "9615": "Clutter / toys",
              "7663": "Mixed competitor shelf"}
COLORS = {"GT": (0, 180, 70), "TP": (30, 105, 255), "FP": (230, 35, 35), "FN": (255, 140, 0)}


def json_write(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def annotations_fingerprint(root, records):
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r["label_path"]):
        relative = record["label_path"]
        content = contained_path(root, relative).read_bytes()
        digest.update(relative.encode("utf-8") + b"\0" + len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def score_detections(ground_truth, predictions, match_iou=0.5):
    available = set(range(len(ground_truth)))
    matches = []
    for pred_index in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i)):
        if not available:
            break
        overlaps = [(box_iou(predictions[pred_index]["bbox"], ground_truth[g]), g) for g in sorted(available)]
        overlap, gt_index = max(overlaps, key=lambda pair: (pair[0], -pair[1]))
        if overlap >= match_iou:
            available.remove(gt_index)
            matches.append({"gt_index": gt_index, "pred_index": pred_index, "iou": overlap})
    matched_predictions = {pair["pred_index"] for pair in matches}
    tp, fp, fn = len(matches), len(predictions) - len(matches), len(available)
    return {"gt_count": len(ground_truth), "pred_count": len(predictions), "TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "matches": matches, "false_positive_indices": sorted(set(range(len(predictions))) - matched_predictions),
            "false_negative_indices": sorted(available)}


def aggregate_counts(records):
    totals = {key: sum(record[key] for record in records) for key in ("TP", "FP", "FN")}
    tp, fp, fn = totals["TP"], totals["FP"], totals["FN"]
    return {"GT": tp + fn, "Pred": tp + fp, **totals,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def load_ground_truth(path, width, height, status):
    state = check_label(path, status)
    if state == "missing":
        raise ValueError(f"Annotation missing: {path}")
    if state == "pending":
        raise ValueError(f"Annotation requires review: {path}")
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        boxes.append([max(0.0, (cx - w / 2) * width), max(0.0, (cy - h / 2) * height),
                      min(float(width), (cx + w / 2) * width), min(float(height), (cy + h / 2) * height)])
    return boxes


def infer(model, records, root, conf, iou, args):
    outputs = {}
    expected = {str((root / record["image_path"]).resolve()): record for record in records}
    results = model.predict(source=list(expected), stream=True, batch=1, conf=conf, iou=iou,
                            imgsz=args.imgsz, device=args.device, max_det=args.max_det,
                            save=False, verbose=False, augment=False)
    for result in results:
        key = str(Path(result.path).resolve())
        if key not in expected or key in outputs:
            raise ValueError(f"Unexpected or duplicate inference image: {result.path}")
        if tuple(result.orig_shape) != (expected[key]["height"], expected[key]["width"]):
            raise ValueError(f"Inference image dimensions disagree with manifest: {result.path}")
        predictions = []
        for bbox, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            if cls != 0 or not all(math.isfinite(v) for v in [*bbox, confidence]):
                raise ValueError(f"Invalid DENA prediction for {result.path}")
            predictions.append({"label": "DENA", "confidence": confidence, "bbox": bbox})
        outputs[key] = predictions
    if set(outputs) != set(expected):
        raise ValueError("Inference did not return every test image")
    return {record["image_path"]: outputs[str((root / record["image_path"]).resolve())] for record in records}


def font(size):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def draw_box(draw, box, color, text, offset_x=0, offset_y=0):
    coords = [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y]
    draw.rectangle(coords, outline=color, width=3)
    small = font(13)
    x, y = coords[0], max(offset_y, coords[1] - 16)
    extent = draw.textbbox((x, y), text, font=small)
    draw.rectangle(extent, fill=(20, 20, 20))
    draw.text((x, y), text, fill=color, font=small)


def visualize(root, record, case, destination):
    with Image.open(root / record["image_path"]) as source:
        original = source.convert("RGB")
    width, height = original.size
    header = 100
    canvas = Image.new("RGB", (width * 2 + 8, height + header), (245, 245, 245))
    canvas.paste(original, (0, header))
    canvas.paste(original, (width + 8, header))
    draw = ImageDraw.Draw(canvas)
    title = Path(record["image_path"]).name
    if len(title) > 40:
        title = title[:35] + "..."
    draw.text((10, 5), f"{title}  GT={case['gt_count']} Pred={case['pred_count']} TP={case['TP']} FP={case['FP']} FN={case['FN']}",
              font=font(19), fill="black")
    draw.text((10, 34), "GROUND TRUTH: green boxes; orange outline = FN", font=font(15), fill="black")
    draw.text((width + 18, 34), "PREDICTIONS: blue = TP; red = unmatched (FP)", font=font(15), fill="black")
    draw.text((10, 60), "G# and P# identify boxes in evaluation.json (1-based)", font=font(14), fill="black")
    draw.text((width + 18, 60), "Labels show prediction ID and confidence", font=font(14), fill="black")
    missed = set(case["false_negative_indices"])
    for index, box in enumerate(case["ground_truth"]):
        if index in missed:
            outer = [max(0, box[0]-4), max(0, box[1]-4), min(width-1, box[2]+4), min(height-1, box[3]+4)]
            draw.rectangle([outer[0], outer[1]+header, outer[2], outer[3]+header], outline=COLORS["FN"], width=3)
        draw_box(draw, box, COLORS["GT"], f"G{index+1}" + (" FN" if index in missed else ""), offset_y=header)
    false_alarms = set(case["false_positive_indices"])
    for index, prediction in enumerate(case["predictions"]):
        color = COLORS["FP"] if index in false_alarms else COLORS["TP"]
        draw_box(draw, prediction["bbox"], color, f"P{index+1} {prediction['confidence']:.2f}", width+8, header)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_write(destination, buffer.getvalue())


def metrics_table(images):
    lines = ["| Image | GT | Pred | TP | FP | FN | Precision | Recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for image in images:
        lines.append(f"| {image['image']} | {image['gt_count']} | {image['pred_count']} | {image['TP']} | {image['FP']} | {image['FN']} | {image['precision']:.2%} | {image['recall']:.2%} |")
    return "\n".join(lines)


def case_observation(identifier, case):
    counts = f"{case['TP']} TP, {case['FP']} unmatched predictions and {case['FN']} misses. "
    if identifier == "5303":
        return counts + "Check missed-box size and localization before attributing misses to scale or introducing slicing."
    if identifier == "4647":
        return counts + "Counts on a glare image do not establish glare causation; inspect each missed package and annotation coverage before proposing CLAHE."
    if identifier == "5140":
        return counts + "Inspect occlusion and IoU localization errors; NMS tuning does not repair inaccurate bounding boxes."
    return counts + "An unmatched prediction is an FP against the supplied labels, but only visual inspection can distinguish a wrong-brand/toy detection from an unlabeled real DENA package."


def render_report(summary, notes=None):
    benchmarks = [next(r for r in summary["images"] if r["benchmark_id"] == identifier) for identifier in BENCHMARKS]
    s = summary["settings"]
    lines = ["# DENA baseline benchmark evaluation", "",
             f"Model: `{summary['weights']}`. All {len(summary['images'])} test images evaluated.", "",
             f"Baseline inference: confidence **{s['conf']}**, NMS IoU **{s['iou']}**, image size **{s['imgsz']}**, maximum detections **{s['max_det']}**.", "",
             f"A TP requires IoU >= **{s['match_iou']}** with an unused GT box. Predictions are processed in descending confidence order and matched to their highest-IoU unused GT. Duplicates and unmatched predictions are FP; unused GT boxes are FN. All boxes are class 0: DENA. Zero-denominator precision/recall are reported as 0. These fixed-threshold counts differ from AP metrics and the training validator's automatically selected operating point.", "",
             "## Locked benchmark results", "", metrics_table(benchmarks), "",
             "## Overall test results", "", metrics_table(summary["images"]), ""]
    for label, counts in (("All test images", summary["test_totals"]), ("Five benchmarks", summary["benchmark_totals"])):
        lines.append(f"{label} (micro): GT={counts['GT']}, predictions={counts['Pred']}, TP={counts['TP']}, FP={counts['FP']}, FN={counts['FN']}; precision={counts['precision']:.2%}, recall={counts['recall']:.2%}, F1={counts['f1']:.2%}.")
        lines.append("")
    lines.extend(["## Qualitative failure analysis", ""])
    for case in benchmarks:
        identifier = case["benchmark_id"]
        reviewed = (notes or {}).get("observations", {}).get(identifier)
        lines.extend([f"### {identifier}: {BENCHMARKS[identifier]}", "", reviewed or case_observation(identifier, case), "",
                      f"![{identifier} comparison]({case['visualization']})", ""])
        for gt in case["gt_diagnostics"]:
            if not gt["matched"]:
                lines.append(f"- Miss G{gt['gt_index']+1}: shorter side at inference scale {gt['short_side_inference_px']:.1f}px; maximum baseline prediction IoU {gt['best_prediction_iou']:.3f}.")
        lines.append("")
    if summary["threshold_sweep"]:
        lines.extend(["## Diagnostic threshold sweep on test", "",
                      "Exploratory only: do not select deployment thresholds or claim fresh held-out performance from this sweep. Confirm any threshold choice on validation data. The default baseline table above remains unchanged.", "",
                      "| Confidence | NMS IoU | TP | FP | FN | Precision | Recall | F1 |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for entry in summary["threshold_sweep"]:
            c = entry["test_totals"]
            lines.append(f"| {entry['conf']:.2f} | {entry['iou']:.2f} | {c['TP']} | {c['FP']} | {c['FN']} | {c['precision']:.2%} | {c['recall']:.2%} | {c['f1']:.2%} |")
        lines.append("")
    lines.extend(["## Recommendation", "", (notes or {}).get("recommendation", "Visual review is required before choosing an intervention. The measurements establish errors against current labels, not their physical cause. Compare threshold recovery of misses and audit unmatched predictions before testing SAHI or CLAHE."), ""])
    for caveat in (notes or {}).get("caveats", []):
        lines.extend([caveat, ""])
    lines.extend(["## Reproducibility and artifacts", "",
                  f"- Weights SHA256: `{summary['weights_sha256']}`",
                  f"- Manifest SHA256: `{summary['manifest_sha256']}`",
                  f"- Test annotations SHA256: `{summary['annotations_sha256']}`",
                  f"- Evaluation duration: {summary['duration_seconds']:.2f} seconds.",
                  "- `evaluation.json`: exact GT/prediction boxes, matches, FN/FP indices, size diagnostics and per-image sweep results.",
                  "- `metrics.csv`: baseline counts and precision/recall for every test image.",
                  "- `predictions/<image-stem>.json`: DENA detections in absolute xyxy pixel coordinates.",
                  "- Comparison PNGs: green GT on the left, missed GT outlined orange; matched predictions blue and unmatched predictions red on the right.",
                  "- Source images, labels, manifests and model weights are never modified by evaluation."])
    if notes:
        lines.extend(["", f"Visual review provenance: {notes.get('reviewer', 'not specified')}."])
    return "\n".join(lines) + "\n"


def read_notes(path, summary):
    if not path.is_file():
        return None
    notes = json.loads(path.read_text(encoding="utf-8-sig"))
    for field in ("weights_sha256", "manifest_sha256", "annotations_sha256", "settings"):
        if notes.get(field) != summary[field]:
            raise ValueError(f"Visual review does not match this evaluation: {field}")
    if not isinstance(notes.get("observations"), dict) or not isinstance(notes.get("recommendation"), str):
        raise ValueError("Review notes require observations and a recommendation")
    return notes


def run(args):
    root = args.root.absolute()
    reject_link_ancestors(root)
    root = root.resolve()
    output = args.output.absolute() if args.output else root / "outputs/benchmark_eval"
    reject_link_ancestors(output)
    output = output.resolve()
    if output == root or (root / "dataset").resolve() in output.parents or output == root / "dataset":
        raise ValueError("Evaluation output must be outside the dataset and not the project root")
    notes_path = args.review_notes or output / "visual_review.json"
    if args.report_only:
        summary = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    else:
        gate = validate(argparse.Namespace(root=root, require_annotations=True))
        if gate != 0:
            raise ValueError(f"Dataset annotation gate failed: {gate}")
        manifest = load_manifest(root / "manifest.json")
        manifest_sha256 = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        records = sorted([r for r in manifest["images"] if r["split"] == "test"], key=lambda r: r["image_path"])
        source_root = Path(manifest["source_root"]).resolve()
        if output == source_root or source_root in output.parents:
            raise ValueError("Evaluation output cannot be inside the raw archive")
        if {r["benchmark_id"] for r in records if r["benchmark_id"]} != set(BENCHMARKS):
            raise ValueError("Test split must contain all five locked benchmarks")
        weights = args.weights.resolve() if args.weights else contained_path(root, "runs/detect/dena_baseline/weights/best.pt")
        if not weights.is_file():
            raise ValueError(f"Trained weights missing: {weights}")
        weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        model = YOLO(str(weights))
        if model.names != {0: "DENA"}:
            raise ValueError(f"Expected a single DENA class, got {model.names}")
        annotations_sha256 = annotations_fingerprint(root, records)
        ground_truth = {r["image_path"]: load_ground_truth(root / r["label_path"], r["width"], r["height"], r["annotation_status"]) for r in records}
        started = time.perf_counter()
        baseline = infer(model, records, root, args.conf, args.iou, args)
        output.mkdir(parents=True, exist_ok=True)
        images = []
        for record in records:
            key = record["image_path"]
            gt, predictions = ground_truth[key], baseline[key]
            case = {"image": Path(key).name, "image_path": key, "benchmark_id": record["benchmark_id"],
                    "ground_truth": gt, "predictions": predictions, **score_detections(gt, predictions, args.match_iou)}
            scale = args.imgsz / max(record["width"], record["height"])
            matched = {m["gt_index"] for m in case["matches"]}
            case["gt_diagnostics"] = [{"gt_index": i, "matched": i in matched,
                "short_side_inference_px": min(box[2]-box[0], box[3]-box[1]) * scale,
                "best_prediction_iou": max((box_iou(box, p["bbox"]) for p in predictions), default=0)} for i, box in enumerate(gt)]
            case["visualization"] = Path(key).stem + "_comparison.png"
            visualize(root, record, case, output / case["visualization"])
            json_write(output / "predictions" / (Path(key).stem + ".json"),
                       {"image": Path(key).name, "count": len(predictions), "detections": predictions})
            images.append(case)
            LOG.info("%s: GT=%d Pred=%d TP=%d FP=%d FN=%d", case["image"], case["gt_count"], case["pred_count"], case["TP"], case["FP"], case["FN"])
        sweep = []
        if not args.no_threshold_sweep:
            for conf in sorted({0.10, args.conf, 0.50}):
                for iou in sorted({0.30, args.iou, 0.60}):
                    predictions = baseline if (conf, iou) == (args.conf, args.iou) else infer(model, records, root, conf, iou, args)
                    scores = {Path(r["image_path"]).name: score_detections(ground_truth[r["image_path"]], predictions[r["image_path"]], args.match_iou) for r in records}
                    sweep.append({"conf": conf, "iou": iou, "test_totals": aggregate_counts(scores.values()),
                                  "images": scores})
                    LOG.info("Threshold diagnostic conf=%.2f iou=%.2f: %s", conf, iou, sweep[-1]["test_totals"])
        if (annotations_fingerprint(root, records) != annotations_sha256 or
                hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != manifest_sha256 or
                hashlib.sha256(weights.read_bytes()).hexdigest() != weights_sha256):
            raise ValueError("Model, manifest or annotations changed during evaluation; rerun with stable inputs")
        summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "weights": str(weights),
                   "weights_sha256": weights_sha256, "manifest_sha256": manifest_sha256,
                   "annotations_sha256": annotations_sha256,
                   "settings": {"conf": args.conf, "iou": args.iou, "match_iou": args.match_iou,
                                "imgsz": args.imgsz, "max_det": args.max_det, "device": args.device},
                   "images": images, "test_totals": aggregate_counts(images),
                   "benchmark_totals": aggregate_counts([r for r in images if r["benchmark_id"]]),
                   "threshold_sweep": sweep, "duration_seconds": time.perf_counter() - started}
        json_write(output / "evaluation.json", summary)
        fields = ["image", "benchmark_id", "gt_count", "pred_count", "TP", "FP", "FN", "precision", "recall", "f1"]
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(images)
        atomic_write(output / "metrics.csv", text.getvalue().encode("utf-8"))
    notes = read_notes(notes_path, summary)
    atomic_write(output / "report.md", render_report(summary, notes).encode("utf-8"))
    benchmarks = [next(r for r in summary["images"] if r["benchmark_id"] == identifier) for identifier in BENCHMARKS]
    print(metrics_table(benchmarks))
    print("All test images (micro): " + json.dumps(summary["test_totals"]))
    print("Conclusion: " + (notes or {}).get("recommendation", "Inspect the saved comparison images before drawing causal conclusions; no SAHI or CLAHE intervention has been applied."))
    print(f"Report: {output / 'report.md'}")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU required for a true positive")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--no-threshold-sweep", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report from saved evaluation and visual review")
    parser.add_argument("--review-notes", type=Path, help="Reviewed observations JSON; default: <output>/visual_review.json")
    args = parser.parse_args(argv)
    if not (0 < args.conf <= 1 and 0 < args.iou <= 1 and 0 < args.match_iou <= 1 and args.imgsz > 0 and args.max_det > 0):
        parser.error("Confidence/IoU thresholds must be in (0,1], image size and max detections must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(args)
    except torch.cuda.OutOfMemoryError:
        LOG.exception("CUDA OOM during benchmark inference")
        torch.cuda.empty_cache()
        return 3
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        LOG.exception("Benchmark evaluation failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
