# DENA detection PoC — dataset preparation

Single detection class: `0: DENA`. This phase prepares data and validates
annotations. It does not train a model, fetch weights, classify SKUs, or build a UI.

## Environment

Run these PowerShell commands from `E:\Coding\Shelf_recognition`:

```powershell
.\dena_poc\.venv\Scripts\Activate.ps1
```

Activation is optional. Every command below invokes the project interpreter
directly, which also avoids PowerShell activation policy restrictions:

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\tools\check_environment.py --output .\dena_poc\outputs\environment.json
& .\dena_poc\.venv\Scripts\python.exe -m pip check
```

The environment uses Python 3.11.9. The Microsoft Store base interpreter needs
execution outside Codex's sandbox on this machine; this does not require another
Python installation. `requirements.txt` records the resolved environment, including
CUDA wheel versions, and includes the official CUDA wheel index for reproduction:

```powershell
py -3.11 -m venv .\dena_poc\.venv
# Put large CUDA download/unpack files on E:, rather than the small system drive.
New-Item -ItemType Directory -Force .\dena_poc\outputs\install_tmp | Out-Null
$env:TEMP = (Resolve-Path .\dena_poc\outputs\install_tmp).Path
$env:TMP = $env:TEMP
& .\dena_poc\.venv\Scripts\python.exe -m pip install -r .\dena_poc\requirements.txt --cache-dir .\dena_poc\outputs\pip_cache --timeout 120
```

## Inventory and selection

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py inventory --source .\Archive_1790087449 --output .\dena_poc\configs\inventory.csv
```

Inventory records every JPG/JPEG/PNG with its relative path, dimensions, SHA-256,
inferred outlet ID, benchmark candidate, decoding errors, and EXIF orientation.
Corrupt files remain visible as error rows and return a nonzero exit code.

`configs/selection.csv` contains 100 images selected through visual contact-sheet
review. The review artifacts live in `outputs/selection_review/`. The raw archive
originally contained 476 images; 295 filenames were relevant candidates for this
focused review. During implementation the source archive changed to 175 images.
At the user's request the selection was rebuilt from the remaining photos; all five
benchmarks remain available. The current inventory covers those 175 images. The
earlier selection is retained in the review artifacts for provenance.
Storefront and phone-order photos were excluded from this initial candidate pool.
The remaining selection covers shelves, refrigerators, mixed competitor scenes,
transport cartons, and clutter. Several useful poster/competitor negatives were
removed with the other source files, reducing negative coverage in this initial set.
Filename category and selecton review are **not** ground-truth annotations.

Each CSV row selects one image. Required columns are `source_path` (POSIX path
relative to the archive) and `outlet_id`. `benchmark_id` is optional and inferred
from the five exact paths when omitted. Additional review columns are informational.
The parsed outlet ID follows `img-<token>_<outlet>-<timestamp>`; notably benchmark
5303 belongs to outlet `1236`, and benchmark 4647 belongs to outlet `5013`.
Outlet identifiers are treated as globally unique in this archive. Verify that
assumption with the data owner before expanding to another export.

The five visually verified benchmark photos are pinned by their full relative
paths, not by arbitrary numeric substrings. Every selected photo from their outlets
goes to test. Selection rejects missing images, duplicate bytes/paths, invalid outlet
IDs, benchmark mismatches, traversal, and linked source/destination paths.

## Split and inspect

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py split --source .\Archive_1790087449 --selection .\dena_poc\configs\selection.csv --output .\dena_poc --seed 42 --dry-run
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py split --source .\Archive_1790087449 --selection .\dena_poc\configs\selection.csv --output .\dena_poc --seed 42
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py validate --root .\dena_poc
```

`--output` and `--root` identify the **project root**, not the dataset directory.
The split writes `dataset/images/{train,val,test}`, creates matching label
directories, writes `configs/data.yaml`, and records provenance in `manifest.json`.
Images are copied byte-for-byte; the archive remains untouched. Benchmarks are named
`5303.jpg`, `4647.jpg`, `5140.jpg`, `9615.jpg`, and `7663.jpg`. Other names use their
content hash. Keep the manifest when exporting images for annotation.

Groups are allocated deterministically using seed 42, with 70%/15%/15% image targets
and whole-outlet moves that strictly improve count deviation. Counts may differ
from exact targets because outlets cannot be split. The CLI reports those deviations.
The prepared selection contains 100 images across 88 outlets and achieved exactly
70 train / 15 validation / 15 test images. Annotation review status is tracked in
`manifest.json` and checked by the readiness validator.
Dry run writes nothing. An identical rerun preserves labels and their review status;
changed selections, stale dataset files, and conflicting images/configuration are
rejected before publishing. Use a separate output root for a different selection.

## Annotation contract and readiness

Annotate individual, identifiable physical DENA juice packages in the copied images.
Use tight boxes around the visible package extent, including identifiable partial
packages; do not infer fully hidden products. Competing brands, promotional artwork,
and package pictures printed on outer transport cartons are background. Do not box
an entire carton stack as one juice package. Brand-ambiguous fragments need human
review before the image is marked complete. SKU, flavor, and volume are not classes.

Write matching UTF-8 YOLO TXT files under `dataset/labels/<split>/`:

```text
0 0.5 0.5 0.2 0.3
```

Format: class ID, normalized center x/y, normalized width/height. Width and height
must be positive, values finite, and the entire box within image bounds. Annotation
tools and image loading must use the same pixel orientation. Inventory flags EXIF
rotation concerns; resolve them consistently before labeling rather than rotating
already-labeled images. Originals and prepared copies are never silently reoriented.

Every record starts with `annotation_status: "pending"` in `manifest.json`.
After the full image has been annotated and reviewed, set its status to
`"reviewed_positive"`. For a reviewed image containing zero DENA packages, create
an empty TXT file and set `"reviewed_negative"`. Missing TXT files stay pending;
the setup script never manufactures empty negatives. Nonempty unreviewed labels
remain pending until the manifest review status is updated.

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py validate --root .\dena_poc --require-annotations
```

Exit codes: `0` structurally valid (and ready when requested), `1` invalid input or
structural/label errors, `2` structurally valid but annotation readiness incomplete.
Normal validation reports missing annotations without treating them as structural
errors. Training must pass `--require-annotations` first.

## Import AnyLabeling annotations

Save reviewed AnyLabeling JSON files beside their images, then run:

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\convert_anylabeling_to_yolo.py --root .\dena_poc --dry-run
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\convert_anylabeling_to_yolo.py --root .\dena_poc
```

The converter accepts case-insensitive `DENA`, numeric/string label `0`, and explicit
shape `class_id`, `classId`, or `class_index` equal to `0`. It ignores `image`,
`background`, and other unmapped labels. `group_id` is an instance grouping field,
not a class mapping. Rectangles accept either corner ordering; polygons become
axis-aligned boxes. Coordinates are normalized by JSON `imageWidth`/`imageHeight`,
which must agree with the actual image and manifest. Invalid DENA shapes, unsupported
shape types, dimension mismatches, and conflicting JSON copies stop before writes.

Exports found in label directories are also supported, including identical copies
of image-directory exports. JSON files are preserved byte-for-byte under
`outputs/anylabeling_archive/{images,labels}/<split>/` with a content-hash suffix.
The previous manifest is backed up there before labels or review statuses change.
The dataset directories contain only their images and YOLO TXT files afterward.

An export with DENA boxes becomes `reviewed_positive`; a reviewed export with zero
DENA boxes creates an empty TXT and becomes `reviewed_negative`. Images without
JSON exports retain their existing statuses and labels. Do not use missing exports
as evidence of negative images. The converter automatically runs the requested
readiness validation, prints its summary and return code, and returns that same
code. Code `2` means conversion completed but some images still need annotations.

Identical reruns are safe. To intentionally replace conflicting existing TXT labels,
use `--overwrite`; previous labels are archived first. After editing an annotation,
copy its updated JSON export back beside the image and run the converter again.

Initial import converted 54 train images: 53 positives, 1 negative, and 341 DENA
boxes. It archived 108 JSON files including duplicate copies in label directories.
After visual review under the ambiguity rule, two additional non-benchmark images
received empty labels: one severely blurred shelf photo and one competitor shelf.
That intermediate manifest contained 53 reviewed positives, 3 reviewed negatives,
and 44 pending images (15 train, 14 validation, 15 test). Review decisions, numbered
contact sheets, and the previous manifest are preserved in
`outputs/ambiguity_review/`; this review was performed by Codex, not a human auditor.

The final import added 24 positive annotations (11 validation, 13 test), including
all five benchmark photos, with 184 additional DENA boxes. The user confirmed that
all 20 remaining unlabeled images were reviewed negatives; these now have empty
TXT labels and `reviewed_negative` manifest statuses. Final counts:

| Split | Reviewed positive | Reviewed negative | Total |
| --- | ---: | ---: | ---: |
| Train | 53 | 17 | 70 |
| Validation | 11 | 4 | 15 |
| Test | 13 | 2 | 15 |
| Total | 77 | 23 | 100 |

Finalization preserves existing labels, all five positive benchmark annotations,
and every dataset image. The prior manifest, user-review authority, affected image
paths, and gate results are recorded in `outputs/final_dataset_gate/`. To finalize
a future batch after explicit human review of all remaining unlabeled images:

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\tools\finalize_reviewed_negatives.py --confirm-human-review
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py validate --root .\dena_poc --require-annotations
```

## Verification and next phase

```powershell
& .\dena_poc\.venv\Scripts\python.exe -m pytest .\dena_poc\tests -q
```

Tests cover Unicode paths, determinism, outlet isolation, benchmark routing,
corruption, duplicate selections, unsafe paths, dry runs, conflicts, reruns, and
YOLO validation/readiness. Symlink tests may skip on Windows if link privileges
are unavailable. `outputs/environment.json` records the installed versions and
small CUDA tensor/NMS checks; these allocate no model and download no weights.
Verified locally: 62 tests passed and 2 symlink tests skipped, including AnyLabeling
conversion, archival, and rollback tests. CUDA 12.6 PyTorch
and CUDA NMS passed on the NVIDIA GeForce RTX 3060 Laptop GPU (6 GB).

## Baseline training

`scripts/train_baseline.py` loads the baseline and dataset YAML files, checks the
strict annotation gate, and trains pretrained YOLOv8s on CUDA. Defaults are 50
epochs, image size 1024, batch 4, two accumulation steps (nominal effective batch
8), AMP, four workers, patience 12, seed 42, and the existing mild augmentations.
Ultralytics does not accept an `accumulate` keyword: the wrapper sets `nbs=8` and
uses a small trainer subclass to keep accumulation at two during warmup as well.
Validation loaders also use batch 4 and four workers. Partial final batches can
contain fewer images. The wrapper aborts on CUDA OOM instead of silently halving
the batch, logs VRAM usage, cleans the CUDA cache, and records a failure summary.

```powershell
# Readiness and GPU checks only; no weights downloaded or training started.
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\train_baseline.py --dry-run
# Run only after explicit user authorization of heavy training.
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\train_baseline.py
```

The pretrained model downloads into `models/`; Ultralytics may also download its
small AMP-check model. Training artifacts are stored in
`runs/detect/dena_baseline/`: `weights/best.pt`, `weights/last.pt`, `results.csv`,
`training.log`, and `training_summary.json`. The summary records duration, GPU
memory, resolved training settings, completed epochs, artifact paths, and test
mAP50, mAP50-95, precision and recall. These are Ultralytics aggregate test metrics;
precision and recall use its selected operating point, not a fixed deployment
confidence threshold. Training uses only train/validation splits; post-training
test evaluation reloads `best.pt` and writes plots under `test_evaluation/`.

Existing runs are protected; choose `--name <new_run>` for a separate experiment.
Generated label scan caches are archived under the run's `label_cache/` directory
so the dataset stays compatible with strict validation. No training run begins
implicitly during setup, annotation conversion, or a dry run. Inference scripting
and baseline failure analysis follow separately; evidence must establish a need
before adding SAHI or CLAHE.

The initial authorized run completed successfully: early stopping at epoch 33
(best validation checkpoint at epoch 21), 182.3 seconds total, and 2350 MiB peak
reserved CUDA memory. Evaluation of `best.pt` on the 15-image test split yielded
mAP50 0.741292, mAP50-95 0.490697, precision 0.716232, and recall 0.830986.
Both checkpoints and `results.csv` were verified. The post-training strict dataset
gate also returned 0. The test suite passed 74 tests with 2 Windows symlink skips.

## Benchmark evaluation

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\eval_benchmarks.py
```

The script evaluates all 15 test images with `best.pt`, confidence 0.25, NMS IoU
0.45 and image size 1024. TP/FP/FN matching independently uses IoU >= 0.50,
confidence-first greedy one-to-one assignments. Batch 1 limits inference memory;
`max_det=1000` avoids the usual 300-detection limit on dense shelves. Empty label
files remain reviewed negatives, and missing annotations stop evaluation.

`outputs/benchmark_eval/` contains `report.md`, `evaluation.json`, `metrics.csv`,
per-image prediction JSON files and 15 comparison PNGs. GT boxes are green;
missed GT receive an orange outline. Matched predictions are blue with confidence,
and unmatched predictions are red. Box IDs refer to the JSON indices plus one.
The five benchmark images receive their own metric rows and visual observations.

A nine-setting confidence/NMS sweep provides exploratory diagnostics; it never
changes the default baseline counts. Select deployment thresholds using validation
data rather than this already-inspected test set. Use `--no-threshold-sweep` to
skip the extra inference passes. Other options include `--weights`, `--output`,
`--device cpu`, `--conf`, `--iou`, `--match-iou` and `--imgsz`.

`visual_review.json` records the completed Codex inspection and intervention
recommendation. `--report-only` regenerates the report from measured results and
those notes without inference. Notes must match weights, manifest, actual test
annotation hashes and inference settings; updated labels require new review.
`tools/write_benchmark_review.py` preserves this initial run's observations and
should only be used for that inspected baseline, not to certify another model.

Initial fixed-threshold results: all test images have 63 TP, 39 FP and 8 FN;
the five benchmarks have 33 TP, 15 FP and 2 FN. The distant shelf and glare
benchmarks recover every annotated package. Both benchmark misses are oblique
package localization failures in 5140. Visual inspection also identifies probable
annotation gaps in 4647, 7663 and other test photos. Current evidence favors
annotation auditing and angle/localization data improvements before SAHI or CLAHE;
threshold tuning alone does not recover the residual misses. Source annotations
and the locked benchmark were preserved.

## Local detection UI

From the workspace root, launch the Gradio app with:

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\app.py
```

Open http://127.0.0.1:7860, upload/drop a shelf image or capture a camera photo,
adjust the confidence and NMS IoU sliders, and click **Detect DENA**. The output
shows DENA boxes with confidence scores and a JSON summary containing `count`
and `inference_ms`. Camera capture requires browser permission.

The app loads the existing `runs/detect/dena_baseline/weights/best.pt` once,
uses CUDA when available (otherwise CPU), and predicts at image size 1024.
Requests run one at a time. Latency measures the prediction call with CUDA
synchronization; the first request includes predictor initialization/warmup.
Uploaded/generated images are cached under `outputs/gradio_cache`, with hourly
cleanup of files older than one hour. The dataset remains untouched. The server
binds only to 127.0.0.1:7860 with public sharing disabled. Press Ctrl+C to stop.
Gradio and its resolved dependencies are pinned in `requirements.txt`.

## Headless inference and release bundle

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\infer.py --source .\photos --device 0 --save-json --save-txt
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\package_deliverables.py
```

`infer.py` supports individual images, recursive directories and videos. Outputs
default to `outputs/inference`; annotated images/videos are always saved. Optional
JSON contains absolute pixel boxes, confidence, count and prediction latency;
optional TXT contains normalized class-0 predictions. Videos get per-frame JSON/TXT
with zero-based indices and timestamps. Counts across frames are not unique objects.
Use `--device cpu` without CUDA. Keep output outside the source directory.

`AGENTS.md` and its identical `DEPLOYMENT_AGENT_INSTRUCTIONS.md` copy provide
machine-readable settings plus PowerShell/Bash bootstrap, inference, background UI
and authorized retraining commands. The resolved environment uses Python 3.11.

The packaging script preflights artifacts and produces `README_DELIVERABLE.md`,
`release_manifest.json` and `outputs/DENA_PoC_Release_v1.0.zip`. The ZIP contains a
`dena_poc/` root with best.pt, deployment tooling, pinned requirements, configs,
training metadata and five benchmark comparisons. CRCs and every payload SHA-256
are verified. Dataset photos/labels and the virtualenv are excluded. For retraining
after extraction, restore the reviewed dataset and rebase `configs/data.yaml` using
the agent recipes. The released summary labels mAP50 74.1%, fixed-threshold test
recall 88.7%, and validator recall 83.1% separately.
