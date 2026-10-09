"""Release completeness, checksum verification and preflight guarantees."""
import importlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def packaging():
    return importlib.import_module("package_deliverables") if importlib.util.find_spec("package_deliverables") else None


def fake_project(packaging, root):
    assert packaging is not None, "Release packaging script is missing"
    for relative in packaging.ARTIFACTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test artifact")
    (root / "runs/detect/dena_baseline/training_summary.json").write_text(json.dumps({
        "status": "completed", "test_metrics": {"mAP50": .7412915, "mAP50_95": .4907,
        "precision": .7162, "recall": .8310}, "epochs_completed": 33}))
    (root / "outputs/benchmark_eval/evaluation.json").write_text(json.dumps({
        "images": [{"TP": 63, "FP": 39, "FN": 8}]}))


def test_missing_required_artifact_creates_no_release(packaging, tmp_path):
    assert packaging is not None, "Release packaging script is missing"
    fake_project(packaging, tmp_path)
    (tmp_path / "infer.py").unlink()
    with pytest.raises(FileNotFoundError, match="infer.py"):
        packaging.package_release(tmp_path)
    assert not (tmp_path / "outputs/DENA_PoC_Release_v1.0.zip").exists()
    assert not (tmp_path / "README_DELIVERABLE.md").exists()


def test_release_manifest_covers_every_payload_with_checksums(packaging, tmp_path):
    assert packaging is not None, "Release packaging script is missing"
    fake_project(packaging, tmp_path)
    archive = packaging.package_release(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        manifest = json.loads(bundle.read("dena_poc/release_manifest.json"))
        payloads = {item["archive_path"] for item in manifest["files"]}
        assert payloads == set(bundle.namelist()) - {"dena_poc/release_manifest.json"}
        for item in manifest["files"]:
            data = bundle.read(item["archive_path"])
            assert packaging.digest(data) == item["sha256"]
            assert len(data) == item["bytes"]
        readme = bundle.read("dena_poc/README_DELIVERABLE.md").decode()
        assert "74.1%" in readme and "88.7%" in readme and "83.1%" in readme
        assert "dataset images and labels are not included" in readme.lower()
    first_hash = packaging.digest(archive.read_bytes())
    assert packaging.digest(packaging.package_release(tmp_path).read_bytes()) == first_hash


def test_tampered_payload_fails_checksum_even_with_valid_zip_crc(packaging, tmp_path):
    assert packaging is not None, "Release packaging script is missing"
    fake_project(packaging, tmp_path)
    archive = packaging.package_release(tmp_path)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(tampered, "w") as changed:
        for name in original.namelist():
            changed.writestr(name, b"tampered" if name == "dena_poc/infer.py" else original.read(name))
    with pytest.raises(ValueError, match="checksum"):
        packaging.verify_archive(tampered)


def test_symlink_artifact_is_not_followed(packaging, tmp_path):
    assert packaging is not None, "Release packaging script is missing"
    fake_project(packaging, tmp_path)
    path = tmp_path / "infer.py"
    path.unlink()
    target = tmp_path / "unlisted.py"
    target.write_text("unlisted")
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(ValueError, match="link"):
        packaging.package_release(tmp_path)
