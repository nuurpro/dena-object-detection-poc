"""Create negative labels for missing annotations after explicit human review.

Use only when the user has confirmed every remaining unlabeled image is negative.
Existing labels and benchmark positive annotations are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_anylabeling_to_yolo import atomic_write
from dataset_utils import check_label, contained_path, reject_link_ancestors
from setup_dataset import load_manifest, validate_manifest_contract


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--confirm-human-review", action="store_true", required=True,
                        help="Confirm all images missing labels have been reviewed as negative")
    args = parser.parse_args()
    root = args.root.absolute()
    reject_link_ancestors(root)
    root = root.resolve()
    manifest_path = contained_path(root, "manifest.json")
    original = manifest_path.read_bytes()
    manifest = load_manifest(manifest_path)
    validate_manifest_contract(manifest, root)
    images_before, labels_before, missing = {}, {}, []
    for record in manifest["images"]:
        image = contained_path(root, record["image_path"])
        image_hash = sha256(image)
        if image_hash != record["sha256"]:
            raise ValueError(f"Image changed: {image}")
        images_before[image] = image_hash
        label = contained_path(root, record["label_path"])
        if record["benchmark_id"]:
            if record["annotation_status"] != "reviewed_positive" or check_label(label, record["annotation_status"]) != "reviewed_positive":
                raise ValueError(f"Benchmark requires positive annotations: {record['benchmark_id']}")
        if label.exists():
            check_label(label, record["annotation_status"])
            labels_before[label] = sha256(label)
        else:
            if record["annotation_status"] == "reviewed_positive":
                raise ValueError(f"Refusing to make a reviewed positive negative: {image}")
            missing.append((record, label))

    out = contained_path(root, "outputs/final_dataset_gate")
    out.mkdir(parents=True, exist_ok=True)
    backup = out / f"manifest_before_{hashlib.sha256(original).hexdigest()}.json"
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError("Conflicting manifest backup")
    if not backup.exists():
        atomic_write(backup, original)
    created = []
    changed_manifest = False
    try:
        for record, label in missing:
            with label.open("xb"):
                pass
            created.append(label)
            record["annotation_status"] = "reviewed_negative"
        if manifest_path.read_bytes() != original:
            raise ValueError("Manifest changed during finalization")
        if created:
            atomic_write(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            changed_manifest = True
        for path, old_hash in {**images_before, **labels_before}.items():
            if sha256(path) != old_hash:
                raise ValueError(f"Previously existing file changed: {path}")
        for record in manifest["images"]:
            check_label(contained_path(root, record["label_path"]), record["annotation_status"])
    except Exception:
        if changed_manifest:
            atomic_write(manifest_path, original)
        for label in created:
            label.unlink()
        raise

    by_split = {split: dict(Counter(r["annotation_status"] for r in manifest["images"] if r["split"] == split))
                for split in ("train", "val", "test")}
    benchmarks = {r["benchmark_id"]: len((root / r["label_path"]).read_text(encoding="utf-8").splitlines())
                  for r in manifest["images"] if r["benchmark_id"]}
    summary = {
        "negative_review_authority": "User confirmed all remaining unlabeled images have no identifiable DENA",
        "new_empty_labels": len(created),
        "new_negative_images": [r["image_path"] for r, _ in missing],
        "annotation_statuses": dict(Counter(r["annotation_status"] for r in manifest["images"])),
        "by_split": by_split,
        "benchmark_boxes": benchmarks,
        "existing_labels_preserved": len(labels_before),
        "images_preserved": len(images_before),
    }
    atomic_write(out / "summary.json", (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"Created {len(created)} empty labels for user-reviewed negative images.")
    print("Statuses: " + json.dumps(summary["annotation_statuses"]))
    print("By split: " + json.dumps(by_split))
    print("Preserved benchmark annotations: " + json.dumps(benchmarks))
    print(f"Preserved {len(labels_before)} existing labels and {len(images_before)} images.")


if __name__ == "__main__":
    main()
