#!/usr/bin/env python3
"""Convert reviewed AnyLabeling JSON exports to DENA YOLO detection labels.

JSON exports are archived outside the dataset. Images without an export remain
pending. The final dataset validation return code is propagated (0, 1 or 2).
"""
import argparse
from collections import Counter
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile

from dataset_utils import (
    DatasetError, SPLITS, check_dataset_tree, check_label, contained_path,
    inspect_image, reject_link_ancestors,
)
from setup_dataset import load_manifest, validate_manifest_contract

LOG = logging.getLogger("anylabeling")
SPURIOUS_LABELS = {"image", "background"}


def is_zero(value):
    """group_id is an instance grouping identifier, never a class identifier."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0) or value == "0"


def is_dena(shape):
    label = shape.get("label")
    text = label.strip().casefold() if isinstance(label, str) else ""
    if text in SPURIOUS_LABELS:
        return False
    return (text == "dena" or is_zero(label) or
            any(is_zero(shape.get(field)) for field in ("class_id", "classId", "class_index")))


def number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatasetError(f"{description} must be a finite number")
    return float(value)


def convert_shapes(document, record, source):
    """Return label bytes and the number of ignored shapes; reject bad DENA boxes."""
    if not isinstance(document, dict):
        raise DatasetError(f"Annotation must be a JSON object: {source}")
    width = number(document.get("imageWidth"), f"{source}: imageWidth")
    height = number(document.get("imageHeight"), f"{source}: imageHeight")
    if width <= 0 or height <= 0 or (width, height) != (record["width"], record["height"]):
        raise DatasetError(f"Annotation dimensions disagree with image/manifest: {source}")
    image_name = document.get("imagePath")
    if image_name is not None:
        if not isinstance(image_name, str) or PureWindowsPath(image_name).name != Path(record["image_path"]).name:
            raise DatasetError(f"Annotation imagePath disagrees with matching image: {source}")
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        raise DatasetError(f"Annotation requires a shapes list (empty for a reviewed negative): {source}")
    lines, ignored = [], 0
    for index, shape in enumerate(shapes, 1):
        context = f"{source}: shape {index}"
        if not isinstance(shape, dict):
            raise DatasetError(f"{context} must be an object")
        if not is_dena(shape):
            ignored += 1
            continue
        kind = shape.get("shape_type")
        points = shape.get("points")
        if kind not in {"rectangle", "polygon"}:
            raise DatasetError(f"Unsupported DENA shape_type {kind!r}: {context}")
        if not isinstance(points, list) or (kind == "rectangle" and len(points) != 2) or (kind == "polygon" and len(points) < 3):
            raise DatasetError(f"Invalid {kind} points: {context}")
        coords = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise DatasetError(f"Expected [x, y] point: {context}")
            coords.append((number(point[0], context), number(point[1], context)))
        x1, y1 = min(p[0] for p in coords), min(p[1] for p in coords)
        x2, y2 = max(p[0] for p in coords), max(p[1] for p in coords)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise DatasetError(f"DENA box is degenerate or outside image bounds: {context}")
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        lines.append("0 " + " ".join(f"{value:.12g}" for value in values))
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8"), len(lines), ignored


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".anylabeling-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def archive_path(archive, origin, split, path, content):
    digest = hashlib.sha256(content).hexdigest()[:16]
    return archive / origin / split / f"{path.stem}.{digest}{path.suffix.lower()}"


def prepare(root, overwrite):
    """Preflight every export, destination and manifest before any write."""
    manifest_path = root / "manifest.json"
    reject_link_ancestors(manifest_path)
    original_manifest = manifest_path.read_bytes()
    data = load_manifest(manifest_path)
    images, labels = validate_manifest_contract(data, root)
    records = {(r["split"], Path(r["image_path"]).stem): r for r in data["images"]}
    for record in data["images"]:
        metadata = inspect_image(contained_path(root, record["image_path"]))
        if any(metadata[field] != record[field] for field in ("width", "height", "sha256")):
            raise DatasetError(f"Image changed since dataset setup: {record['image_path']}")
    sources, documents = {}, {}
    for origin in ("images", "labels"):
        for split in SPLITS:
            folder = root / "dataset" / origin / split
            reject_link_ancestors(folder)
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() != ".json":
                    continue
                reject_link_ancestors(path)
                if not path.is_file():
                    raise DatasetError(f"JSON export is not a file: {path}")
                key = split, path.stem
                if key not in records:
                    raise DatasetError(f"JSON export has no matching manifest image: {path}")
                content = path.read_bytes()
                document = json.loads(content.decode("utf-8-sig"))
                if key in documents and document != documents[key]:
                    raise DatasetError(f"Conflicting image/label JSON exports for {split}/{path.stem}")
                documents[key] = document
                sources[path] = content
    allowed_images = images | {p.relative_to(root).as_posix() for p in sources if p.parent.parent.name == "images"}
    allowed_labels = labels | {p.relative_to(root).as_posix() for p in sources if p.parent.parent.name == "labels"}
    check_dataset_tree(root, allowed_images, allowed_labels)
    archive = root / "outputs/anylabeling_archive"
    reject_link_ancestors(archive)
    writes, original_labels, backups = {}, {}, {}
    counts = Counter()
    for key, document in sorted(documents.items()):
        record = records[key]
        content, boxes, ignored = convert_shapes(document, record, f"{key[0]}/{key[1]}.json")
        path = contained_path(root, record["label_path"])
        if path.exists() and not path.is_file():
            raise DatasetError(f"Label output is not a file: {path}")
        previous = path.read_bytes() if path.exists() else None
        if previous is not None and previous != content:
            if not overwrite:
                raise DatasetError(f"Existing label conflicts with JSON; use --overwrite to back it up and replace it: {path}")
            backups[archive_path(archive, "label_history", key[0], path, previous)] = previous
        original_labels[path] = previous
        writes[path] = content
        record["annotation_status"] = "reviewed_positive" if boxes else "reviewed_negative"
        counts["images"] += 1
        counts["boxes"] += boxes
        counts["ignored"] += ignored
        counts["positive" if boxes else "negative"] += 1
        counts[key[0]] += 1
    for path, content in sources.items():
        destination = archive_path(archive, path.parent.parent.name, path.parent.name, path, content)
        backups[destination] = content
    if sources:
        digest = hashlib.sha256(original_manifest).hexdigest()
        backups[archive / "manifest_before" / f"{digest}.json"] = original_manifest
    for path, content in backups.items():
        reject_link_ancestors(path)
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise DatasetError(f"Conflicting archive destination: {path}")
    updated = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return data, sources, writes, original_labels, backups, original_manifest, updated, counts


def publish(root, plan):
    data, sources, writes, previous_labels, backups, original_manifest, updated, _ = plan
    manifest = root / "manifest.json"
    if not sources:
        return
    # Backups precede mutation; a failed copy cannot discard the annotation source.
    for path, content in backups.items():
        if not path.exists():
            atomic_write(path, content)
    if manifest.read_bytes() != original_manifest or any(p.read_bytes() != c for p, c in sources.items()):
        raise DatasetError("Annotations or manifest changed during conversion; rerun on the new files")
    for path, previous in previous_labels.items():
        if (path.read_bytes() if path.exists() else None) != previous:
            raise DatasetError(f"Label changed during conversion: {path}")
    changed, removed = [], []
    try:
        for path, content in writes.items():
            atomic_write(path, content)
            changed.append(path)
        for record in data["images"]:
            label = root / record["label_path"]
            if label in writes:
                check_label(label, record["annotation_status"])
        atomic_write(manifest, updated)
        for path, original in sources.items():
            if path.read_bytes() != original:
                raise DatasetError(f"JSON changed before archival cleanup: {path}")
            destination = archive_path(root / "outputs/anylabeling_archive", path.parent.parent.name, path.parent.name, path, original)
            if destination.read_bytes() != original:
                raise DatasetError(f"Archive verification failed: {path}")
            path.unlink()
            removed.append(path)
    except Exception:
        # Recover source annotations and original labels/statuses after publish failure.
        for path in removed:
            atomic_write(path, sources[path])
        for path in changed:
            if previous_labels[path] is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, previous_labels[path])
        atomic_write(manifest, original_manifest)
        raise


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="PoC project root")
    parser.add_argument("--dry-run", action="store_true", help="Preflight and report without writing or validating dirty folders")
    parser.add_argument("--overwrite", action="store_true", help="Back up and replace conflicting existing YOLO labels")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        root = args.root.absolute()
        reject_link_ancestors(root)
        root = root.resolve()
        plan = prepare(root, args.overwrite)
        counts = plan[-1]
        print(f"Conversion: {counts['images']} annotated images ({counts['positive']} positive, {counts['negative']} negative), "
              f"{counts['boxes']} DENA boxes, {counts['ignored']} ignored shapes.")
        print("Converted by split: " + ", ".join(f"{split}={counts[split]}" for split in SPLITS))
        if args.dry_run:
            print("Dry run: no files or directories written; validation deferred until JSON exports are archived.")
            return 0
        publish(root, plan)
        LOG.info("Archived %d JSON files under %s; unannotated images keep their statuses.", len(plan[1]), root / "outputs/anylabeling_archive")
        print("Running setup_dataset.py validate --require-annotations", flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("setup_dataset.py")),
                                 "validate", "--root", str(root), "--require-annotations"], check=False)
        print(f"Validation return code: {result.returncode}", flush=True)
        return result.returncode
    except (OSError, ValueError, KeyError, TypeError, OverflowError, SyntaxError) as exc:
        LOG.error("Conversion stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
