"""Verify benchmark matching and count conservation without model inference."""
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def evaluation():
    return importlib.import_module("eval_benchmarks") if importlib.util.find_spec("eval_benchmarks") else None


def test_duplicate_predictions_match_only_once_and_highest_confidence_wins(evaluation):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    gt = [[0, 0, 10, 10]]
    predictions = [{"bbox": [1, 1, 9, 9], "confidence": 0.9},
                   {"bbox": [0, 0, 10, 10], "confidence": 0.5}]
    scored = evaluation.score_detections(gt, predictions, 0.5)
    assert (scored["TP"], scored["FP"], scored["FN"]) == (1, 1, 0)
    assert scored["matches"][0]["pred_index"] == 0
    assert scored["false_positive_indices"] == [1]


def test_crowded_boxes_match_distinct_ground_truth(evaluation):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    gt = [[0, 0, 10, 10], [5, 0, 15, 10]]
    predictions = [{"bbox": [3, 0, 13, 10], "confidence": 0.9},
                   {"bbox": [0, 0, 10, 10], "confidence": 0.8}]
    scored = evaluation.score_detections(gt, predictions, 0.5)
    assert scored["TP"] == 2
    assert {m["gt_index"] for m in scored["matches"]} == {0, 1}


def test_matching_threshold_is_distinct_from_nms_threshold(evaluation):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    scored = evaluation.score_detections([[0, 0, 10, 10]],
             [{"bbox": [3.5, 0, 13.5, 10], "confidence": 0.9}], 0.5)
    assert (scored["TP"], scored["FP"], scored["FN"]) == (0, 1, 1)


@pytest.mark.parametrize("gt,pred,expected", [([], [], (0, 0, 0)),
    ([[0, 0, 10, 10]], [], (0, 0, 1)),
    ([], [{"bbox": [0, 0, 10, 10], "confidence": 0.5}], (0, 1, 0))])
def test_empty_positive_and_negative_images(evaluation, gt, pred, expected):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    result = evaluation.score_detections(gt, pred, 0.5)
    assert (result["TP"], result["FP"], result["FN"]) == expected
    assert result["TP"] + result["FN"] == result["gt_count"]
    assert result["TP"] + result["FP"] == result["pred_count"]


def test_gt_coordinates_use_actual_image_dimensions(evaluation, tmp_path):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    path = tmp_path / "label.txt"
    path.write_text("0 0.5 0.5 0.2 0.4\n")
    assert evaluation.load_ground_truth(path, 100, 200, "reviewed_positive") == [[40, 60, 60, 140]]


def test_missing_ground_truth_is_not_a_negative(evaluation, tmp_path):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    with pytest.raises(ValueError, match="missing|requires"):
        evaluation.load_ground_truth(tmp_path / "missing.txt", 100, 200, "reviewed_negative")


def test_summary_uses_micro_counts_not_average_per_image_precision(evaluation):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    summary = evaluation.aggregate_counts([{"TP": 9, "FP": 1, "FN": 0}, {"TP": 1, "FP": 9, "FN": 2}])
    assert summary["precision"] == 0.5
    assert summary["recall"] == pytest.approx(10/12)
    assert summary["GT"] == 12 and summary["Pred"] == 20


def test_report_recommendations_do_not_infer_glare_causation_from_counts(evaluation):
    assert evaluation is not None, "Benchmark evaluation script is missing"
    case = {"FN": 1, "FP": 2, "TP": 2, "gt_count": 3, "pred_count": 4,
            "precision": 0.5, "recall": 2/3}
    observations = evaluation.case_observation("4647", case)
    assert "causation" in observations.lower() or "causal" in observations.lower()


def test_annotation_fingerprint_changes_when_boxes_change(evaluation, tmp_path):
    assert evaluation is not None
    label = tmp_path / "label.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    records = [{"label_path": "label.txt"}]
    before = evaluation.annotations_fingerprint(tmp_path, records)
    label.write_text("0 0.5 0.5 0.3 0.3\n")
    assert evaluation.annotations_fingerprint(tmp_path, records) != before


def test_visual_review_rejects_changed_ground_truth(evaluation, tmp_path):
    assert evaluation is not None
    import json
    meta = {"weights_sha256": "weights", "manifest_sha256": "manifest", "settings": {}, "annotations_sha256": "new-labels"}
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({**meta, "annotations_sha256": "old-labels",
                               "observations": {}, "recommendation": "review"}))
    with pytest.raises(ValueError, match="annotations_sha256"):
        evaluation.read_notes(path, meta)
