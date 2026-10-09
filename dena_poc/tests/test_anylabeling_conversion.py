"""Conversion tests use synthetic images and annotations, never real data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from dataset_utils import BENCHMARKS, RATIOS, expected_image_path, expected_label_path

SCRIPT = Path(__file__).parents[1] / "scripts/convert_anylabeling_to_yolo.py"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "PoC полка"
    records = []
    sources = [(path, benchmark, "test") for path, benchmark in BENCHMARKS.items()]
    sources += [("Rep/shelf-img-ab_90001-1.png", "", "train"),
                ("Rep/shelf-img-ab_90002-2.png", "", "val")]
    for n, (source, benchmark, split) in enumerate(sources, 1):
        # Compute the filename using the same contract as the production manifest.
        temp = tmp_path / (f"photo{n}" + Path(source).suffix)
        Image.new("RGB", (100, 80), (n * 23, n * 13, n * 7)).save(temp)
        import re
        record = dict(source_path=source, outlet_id=re.search(r"img-[a-z0-9]+_(\d+)-", source).group(1),
                      benchmark_id=benchmark, split=split, annotation_status="pending",
                      sha256=hashlib.sha256(temp.read_bytes()).hexdigest(), width=100, height=80,
                      exif_orientation=1, orientation_concern=False)
        record["image_path"] = expected_image_path(record)
        record["label_path"] = expected_label_path(record)
        target = root / record["image_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(temp.read_bytes())
        records.append(record)
    for category in ("images", "labels"):
        for split in RATIOS:
            (root / "dataset" / category / split).mkdir(parents=True, exist_ok=True)
    data = dict(version=1, seed=42, source_root=str((tmp_path / "originals").resolve()),
                ratios=RATIOS, counts={s: sum(r["split"] == s for r in records) for s in RATIOS}, images=records)
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs/data.yaml").write_text(yaml.safe_dump(dict(path=str((root / "dataset").resolve()),
                train="images/train", val="images/val", test="images/test", names={0: "DENA"})), encoding="utf-8")
    return root, records


def shape(label="DENA", kind="rectangle", points=None, **fields):
    return dict(label=label, shape_type=kind, points=points or [[10, 20], [50, 60]], **fields)


def annotation(root, record, shapes, location="images", **fields):
    path = root / "dataset" / location / record["split"] / (Path(record["image_path"]).stem + ".json")
    doc = dict(imageWidth=100, imageHeight=80, imagePath=Path(record["image_path"]).name, shapes=shapes, **fields)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def annotate_all(root, records):
    return [annotation(root, record, [shape()]) for record in records]


def cli(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *args],
                          text=True, capture_output=True, encoding="utf-8")


def read_manifest(root):
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_rectangles_polygons_mapped_classes_and_spurious_labels(project):
    root, records = project
    paths = annotate_all(root, records)
    annotation(root, records[0], [shape(" dEnA ", points=[[50, 60], [10, 20]]),
        shape("DENA", "polygon", [[10, 10], [70, 20], [60, 70]]), shape("0"),
        shape(0), shape("juice", class_id=0), shape("image", group_id=0),
        shape("background", class_id=0), shape("Bliss", group_id=0)])
    result = cli(root)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (root / records[0]["label_path"]).read_text().splitlines()
    assert len(lines) == 5
    assert list(map(float, lines[0].split())) == pytest.approx([0, .3, .5, .4, .5])
    assert list(map(float, lines[1].split())) == pytest.approx([0, .4, .5, .6, .75])
    assert all(r["annotation_status"] == "reviewed_positive" for r in read_manifest(root)["images"])
    assert all(not p.exists() for p in paths)
    assert "Validation return code: 0" in result.stdout


def test_empty_reviewed_image_is_negative_missing_images_stay_pending(project):
    root, records = project
    annotation(root, records[0], [])
    annotation(root, records[1], [shape()])
    result = cli(root)
    assert result.returncode == 2, result.stdout + result.stderr
    data = read_manifest(root)
    assert data["images"][0]["annotation_status"] == "reviewed_negative"
    assert (root / records[0]["label_path"]).read_bytes() == b""
    assert data["images"][1]["annotation_status"] == "reviewed_positive"
    assert all(r["annotation_status"] == "pending" for r in data["images"][2:])
    assert all(not (root / r["label_path"]).exists() for r in records[2:])
    assert "5 missing labels" in result.stdout and "Validation return code: 2" in result.stdout


def test_mirror_jsons_archived_byte_for_byte_and_rerun_is_idempotent(project):
    root, records = project
    sources = annotate_all(root, records)
    mirror = annotation(root, records[0], [shape()], location="labels")
    sources.append(mirror)
    original = {p: p.read_bytes() for p in sources}
    before = (root / "manifest.json").read_bytes()
    assert cli(root).returncode == 0
    archive = root / "outputs/anylabeling_archive"
    copies = [p.read_bytes() for p in archive.rglob("*.json")]
    assert before in copies
    assert all(value in copies for value in original.values())
    assert all(not p.exists() for p in sources)
    snapshot = (root / "manifest.json").read_bytes()
    labels = {r["label_path"]: (root / r["label_path"]).read_bytes() for r in records}
    result = cli(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "manifest.json").read_bytes() == snapshot
    assert labels == {r["label_path"]: (root / r["label_path"]).read_bytes() for r in records}


@pytest.mark.parametrize("damage", ["malformed", "dimension", "zero_size", "out_of_bounds", "nan", "unsupported", "missing_shapes", "missing_dimension", "different_image", "orphan", "mirror_conflict"])
def test_bad_inputs_fail_preflight_without_mutating_labels_or_jsons(project, damage):
    root, records = project
    paths = annotate_all(root, records)
    path = paths[0]
    data = json.loads(path.read_text())
    if damage == "malformed": path.write_text("not JSON")
    elif damage == "dimension": data["imageWidth"] = 101
    elif damage == "zero_size": data["shapes"][0]["points"] = [[10, 20], [10, 60]]
    elif damage == "out_of_bounds": data["shapes"][0]["points"] = [[-5, 20], [50, 60]]
    elif damage == "nan": data["shapes"][0]["points"][0][0] = float("nan")
    elif damage == "unsupported": data["shapes"][0]["shape_type"] = "circle"
    elif damage == "missing_shapes": del data["shapes"]
    elif damage == "missing_dimension": del data["imageHeight"]
    elif damage == "different_image": data["imagePath"] = "different.jpg"
    elif damage == "orphan": path.rename(path.with_name("orphan.json"))
    elif damage == "mirror_conflict": annotation(root, records[0], [], location="labels")
    if damage not in {"malformed", "orphan"}: path.write_text(json.dumps(data))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = cli(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_dry_run_has_no_writes(project):
    root, records = project
    annotate_all(root, records)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = cli(root, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_existing_label_conflict_requires_explicit_overwrite_and_backup(project):
    root, records = project
    annotate_all(root, records)
    label = root / records[0]["label_path"]
    original = b"0 0.5 0.5 0.1 0.1\n"
    label.write_bytes(original)
    assert cli(root).returncode == 1
    assert label.read_bytes() == original
    result = cli(root, "--overwrite")
    assert result.returncode == 0, result.stdout + result.stderr
    assert original in [p.read_bytes() for p in (root / "outputs/anylabeling_archive").rglob("*.txt")]


def test_only_spurious_labels_produce_reviewed_negative(project):
    root, records = project
    annotate_all(root, records)
    annotation(root, records[0], [shape("image", "circle"), shape("background"), shape("Bliss", group_id=0)])
    assert cli(root).returncode == 0
    assert (root / records[0]["label_path"]).read_bytes() == b""
    assert read_manifest(root)["images"][0]["annotation_status"] == "reviewed_negative"


def test_label_directory_only_export_supported(project):
    root, records = project
    for record in records:
        annotation(root, record, [shape()], location="labels")
    result = cli(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list((root / "dataset").rglob("*.json"))


def test_cleanup_failure_rolls_back_labels_statuses_and_removed_jsons(project, monkeypatch):
    import convert_anylabeling_to_yolo as converter
    root, records = project
    sources = annotate_all(root, records)
    original_jsons = {path: path.read_bytes() for path in sources}
    original_manifest = (root / "manifest.json").read_bytes()
    plan = converter.prepare(root, overwrite=False)
    original_unlink = Path.unlink
    calls = 0

    def locked_second_json(path, *args, **kwargs):
        nonlocal calls
        if path in original_jsons:
            calls += 1
            if calls == 2:
                raise OSError("Simulated annotation file lock")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_second_json)
    with pytest.raises(OSError, match="Simulated annotation file lock"):
        converter.publish(root, plan)
    assert (root / "manifest.json").read_bytes() == original_manifest
    assert all(path.read_bytes() == content for path, content in original_jsons.items())
    assert all(not (root / record["label_path"]).exists() for record in records)
