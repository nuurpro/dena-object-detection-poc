# DENA Object Detection PoC — Release v1.0

Single-class YOLOv8s retail detector: class `0: DENA`. Weights, local Gradio UI,
headless image/directory/video inference, deployment recipes and evaluation evidence
are included. Extract the ZIP to obtain the `dena_poc/` project directory.

| Measurement | Result | Evaluation method |
| --- | ---: | --- |
| mAP50 | 74.1% | Ultralytics test validator |
| mAP50–95 | 49.1% | Ultralytics test validator |
| Recall | 88.7% | All test images, conf 0.25 / NMS IoU 0.45 / match IoU 0.50 |
| Precision | 61.8% | Same fixed-threshold evaluation |
| Validator recall | 83.1% | Validator-selected operating point |
| Validator precision | 71.6% | Validator-selected operating point |

Fixed-threshold counts: 63 TP, 39 FP, 8 FN. The recall of 88.7%
and mAP50 of 74.1% come from different evaluation methods.
See `outputs/benchmark_eval/report.md` for per-case results and annotation concerns.
Angled carton localization remains a failure mode; audit partial boxes and probable
annotation gaps before changing thresholds or adding SAHI/CLAHE.

## Launch from the extracted project's parent directory

Use Python 3.11 for the pinned environment. Bootstrap and background/service
commands for Windows and Linux are in `AGENTS.md` (identical to
`DEPLOYMENT_AGENT_INSTRUCTIONS.md`).

```powershell
py -3.11 -m venv .\dena_poc\.venv
& .\dena_poc\.venv\Scripts\python.exe -m pip install -r .\dena_poc\requirements.txt
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\infer.py --source .\photos --device 0 --save-json --save-txt
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\app.py
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
