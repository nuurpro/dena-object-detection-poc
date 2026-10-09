# DENA PoC execution contract

Audience: autonomous coding agents (Codex, Cursor, Claude Code) and MLOps agents.
Parse the following YAML contract, then execute the recipes from the extracted
project's **parent directory**. Relative paths below use `dena_poc/` as the root.

```yaml
contract_version: 1
project: DENA_PoC
release: '1.0'
task: single_class_retail_package_detection
architecture: YOLOv8s
classes: {0: DENA}
runtime:
  architecture_python_minimum: '3.10'
  pinned_release_python: '3.11'
  verified_python: '3.11.9'
  framework: PyTorch
  cuda_wheel_index: https://download.pytorch.org/whl/cu126
  gpu_reference: NVIDIA_GeForce_RTX_3060_Laptop_6GB
  inference_cpu_fallback: true
weights: runs/detect/dena_baseline/weights/best.pt
inference:
  ndarray_color_order: BGR
  ndarray_shape: HWC
  ndarray_dtype: uint8
  imgsz: 1024
  conf: 0.25
  nms_iou: 0.45
  class_filter: [0]
  max_det: 1000
  batch: 1
  json_bbox: absolute_pixel_xyxy
  video_frame_index_origin: 0
  json_keys: [image, count, detections, inference_ms]
ui: {host: 127.0.0.1, port: 7860, share: false}
training:
  model: yolov8s.pt
  epochs: 50
  patience: 12
  seed: 42
  imgsz: 1024
  batch: 4
  accumulate: 2
  nbs: 8
  amp: true
  workers: 4
  authorization: explicit_user_authorization_or_preapproved_MLOps_training_job
  annotation_gate: require_annotations
dataset:
  included_in_release: false
  splits: {train: 70, val: 15, test: 15}
  grouping: outlet_id
  verified_positive_images: 77
  verified_negative_images: 23
  locked_test_benchmarks: ['5303', '4647', '5140', '9615', '7663']
  negative_status: reviewed_negative
  positive_status: reviewed_positive
  pending_status: pending
scope_exclusions: [SKU_classification, share_of_shelf, planogram, embeddings]
exit_codes:
  inference: {success: 0, input_or_runtime_error: 1, argument_error: 2, cuda_oom: 3, interrupted: 130}
  packaging: {success: 0, error: 1}
```

## Bootstrap without interactive prompts

The architecture supports Python 3.10+, but the **resolved release pins require
Python 3.11+** (including pandas 3). Use Python 3.11 for reproducibility. Install
a compatible NVIDIA driver before deploying the CUDA wheels; the CUDA toolkit is
not required for prebuilt PyTorch wheels. CPU inference remains supported.

PowerShell, with Python 3.11 installed:

```powershell
py -3.11 -m venv .\dena_poc\.venv
& .\dena_poc\.venv\Scripts\python.exe -m pip install --no-input -r .\dena_poc\requirements.txt
& .\dena_poc\.venv\Scripts\python.exe -m pip check
& .\dena_poc\.venv\Scripts\python.exe -c "import torch, ultralytics, gradio; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Bash, with Python 3.11 and its venv module installed:

```bash
python3.11 -m venv dena_poc/.venv
dena_poc/.venv/bin/python -m pip install --no-input -r dena_poc/requirements.txt
dena_poc/.venv/bin/python -m pip check
dena_poc/.venv/bin/python -c "import torch, ultralytics, gradio; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`requirements.txt` pins ultralytics, gradio, torch, torchvision and transitive
dependencies. The original environment is Windows/Python 3.11/CUDA 12.6; verify
wheel availability and imports on a different OS. No pretrained model downloads
or training are part of bootstrap. Missing best.pt is an error.

## Headless inference

Execute these commands with an existing `photos/` directory (or substitute an
image/video path). Choose `--device cpu` when CUDA is unavailable. Paths with
spaces must be quoted. CLI input arrays are BGR; the UI accepts PIL RGB and lets
Ultralytics perform its internal conversion. Do not manually swap a BGR array.

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\infer.py --source .\photos --weights .\dena_poc\runs\detect\dena_baseline\weights\best.pt --conf 0.25 --iou 0.45 --imgsz 1024 --device 0 --output-dir .\dena_poc\outputs\inference --save-json --save-txt
```

```bash
dena_poc/.venv/bin/python dena_poc/infer.py --source ./photos --weights dena_poc/runs/detect/dena_baseline/weights/best.pt --conf 0.25 --iou 0.45 --imgsz 1024 --device 0 --output-dir dena_poc/outputs/inference --save-json --save-txt
```

For `photos/subdir/shelf.jpg`, outputs are `subdir/shelf_annotated.jpg`,
`subdir/shelf.json` and `subdir/shelf.txt`. Optional JSON/TXT flags are independent;
annotated media are always saved. Empty predictions create empty TXT when requested.
JSON detections contain `label: DENA`, `confidence`, absolute pixel
`bbox: [x1,y1,x2,y2]`, count and prediction latency (milliseconds). The first
prediction includes model predictor warmup; timings exclude file I/O and drawing.
Model weights are loaded once per process. CPU timings are expected to be slower.

Videos produce `<stem>_annotated.mp4` and optional
`<stem>_frames/frame_000000.json`/`.txt` for every decoded frame. JSON adds
`frame_index` and `timestamp_ms`. Video counts are per-frame detections, not unique
package counts. Inputs are files, not network streams; camera capture uses the UI.
Directory discovery is recursive and deterministic; unsupported files are ignored.
Same-directory media with the same stem are rejected to avoid output collisions.
Keep output outside the input directory. Re-running replaces generated outputs;
use a new output directory to retain previous predictions. Never import predicted
TXT files into ground truth without human review. Exit codes are in the YAML above.

## UI background launch

PowerShell: spawn a hidden process; log stdout/stderr and save its PID. A missing
port listener or startup error in the logs means deployment failed.

```powershell
$denaRoot = (Resolve-Path .\dena_poc).Path
$denaPython = Join-Path $denaRoot '.venv\Scripts\python.exe'
$denaAppArgument = '"' + (Join-Path $denaRoot 'app.py') + '"'
New-Item -ItemType Directory -Force -Path (Join-Path $denaRoot 'outputs') | Out-Null
$denaUi = Start-Process -FilePath $denaPython -ArgumentList @('-u', $denaAppArgument) -WorkingDirectory $denaRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $denaRoot 'outputs\ui.stdout.log') -RedirectStandardError (Join-Path $denaRoot 'outputs\ui.stderr.log') -PassThru
$denaUi.Id | Set-Content (Join-Path $denaRoot 'outputs\ui.pid')
```

Bash:

```bash
cd dena_poc
mkdir -p outputs
nohup .venv/bin/python -u app.py > outputs/ui.log 2>&1 < /dev/null &
echo $! > outputs/ui.pid
cd ..
```

For a Linux systemd service, use absolute deployment paths in the following unit,
then start it via the host's service manager under its deployment account:

```ini
[Unit]
Description=DENA local detection UI
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/dena_poc
ExecStart=/opt/dena_poc/.venv/bin/python -u /opt/dena_poc/app.py
Environment=GRADIO_ANALYTICS_ENABLED=False
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Health check: HTTP GET `http://127.0.0.1:7860/` must return 200. The process loads
best.pt once and serializes inference to limit VRAM use. Keep `share=False` and
loopback binding unless a separate deployment task authorizes network exposure.
Use the saved PID only after confirming it belongs to this app when stopping it.

## Dataset extension and retraining loop

The ZIP carries metadata, not the photos/labels. Restore the reviewed dataset to
`dena_poc/dataset/{images,labels}/{train,val,test}` before validation/retraining.
`configs/data.yaml` retains the original machine's path for provenance. Rebase it
automatically after extraction; these commands resolve the restored dataset:

```powershell
& .\dena_poc\.venv\Scripts\python.exe -c "from pathlib import Path; import yaml; p=Path('dena_poc/configs/data.yaml'); d=yaml.safe_load(p.read_text(encoding='utf-8')); d['path']=str(Path('dena_poc/dataset').resolve()); p.write_text(yaml.safe_dump(d, sort_keys=False), encoding='utf-8')"
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\setup_dataset.py validate --root .\dena_poc --require-annotations
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\train_baseline.py --root .\dena_poc --config .\dena_poc\configs\baseline.yaml --device 0 --name dena_retrain_v1 --dry-run
```

```bash
dena_poc/.venv/bin/python -c "from pathlib import Path; import yaml; p=Path('dena_poc/configs/data.yaml'); d=yaml.safe_load(p.read_text(encoding='utf-8')); d['path']=str(Path('dena_poc/dataset').resolve()); p.write_text(yaml.safe_dump(d, sort_keys=False), encoding='utf-8')"
dena_poc/.venv/bin/python dena_poc/scripts/setup_dataset.py validate --root dena_poc --require-annotations
dena_poc/.venv/bin/python dena_poc/scripts/train_baseline.py --root dena_poc --config dena_poc/configs/baseline.yaml --device 0 --name dena_retrain_v1 --dry-run
```

After explicit authorization or inside an already approved MLOps training job:

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\train_baseline.py --root .\dena_poc --config .\dena_poc\configs\baseline.yaml --device 0 --name dena_retrain_v1
```

```bash
dena_poc/.venv/bin/python dena_poc/scripts/train_baseline.py --root dena_poc --config dena_poc/configs/baseline.yaml --device 0 --name dena_retrain_v1
```

Choose a fresh run name for every attempt. The YAML enforces batch=4,
accumulate=2, imgsz=1024, AMP=true, workers=4 and nominal effective batch 8.
`accumulate` is not a public Ultralytics CLI argument: the provided custom trainer
translates it to nbs=8 and enforces accumulation=2 after warmup. Validation batches
are capped at 4. Do not substitute a plain `yolo train accumulate=2` command.
CUDA OOM exits with diagnostics and cache cleanup; this is a memory-conscious
configuration, not a guarantee against other processes consuming VRAM.
Packaging or inference must never start a training job automatically.

Preserve outlet isolation and all five locked benchmark outlet groups in test.
Only class 0 is valid. Keep human-verified negatives with empty TXT and manifest
status `reviewed_negative`; missing labels remain pending, never inferred negative
from filenames or failed detections. Human-reviewed discernible DENA packages get
tight full visible-package boxes, including caps. Inspect angled/partially occluded
cartons carefully: 5140 has IoU localization failures and probable missed annotations.
Audit competing-brand false alarms and preserve useful hard negatives. Do not edit
locked benchmark labels silently; record adjudication and regenerate evaluation
evidence after approved annotation changes. SAHI/CLAHE require measured benefit.

## Release refresh

```powershell
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\scripts\package_deliverables.py
```

```bash
dena_poc/.venv/bin/python dena_poc/scripts/package_deliverables.py
```

Keep `AGENTS.md` and `DEPLOYMENT_AGENT_INSTRUCTIONS.md` byte-identical.
The packager preflights explicit artifacts, writes `release_manifest.json` and
`README_DELIVERABLE.md`, verifies every ZIP CRC and payload SHA-256, and replaces
`outputs/DENA_PoC_Release_v1.0.zip` atomically. Evidence must match packaged weights
and dataset manifest. The release includes helper scripts needed by training.
