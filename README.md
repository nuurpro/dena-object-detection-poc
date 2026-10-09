# DENA retail package detection PoC

Single-class **YOLOv8s** detector for DENA juice packaging (class `0`). Includes
headless inference, a local Gradio interface, dataset preparation, annotation
conversion, memory-conscious training, benchmark evaluation and release packaging.

The trained checkpoint is included at
`dena_poc/runs/detect/dena_baseline/weights/best.pt`.

## Quick start

Run from this repository's root using Python 3.11:

```powershell
py -3.11 -m venv .\dena_poc\.venv
& .\dena_poc\.venv\Scripts\python.exe -m pip install -r .\dena_poc\requirements.txt
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\infer.py --source .\photos --device 0 --save-json --save-txt
& .\dena_poc\.venv\Scripts\python.exe .\dena_poc\app.py
```

Open http://127.0.0.1:7860 for the UI. Use `--device cpu` for headless inference
without CUDA. Defaults: image size 1024, confidence 0.25, NMS IoU 0.45.

The [v1.0 release](https://github.com/nuurpro/dena-object-detection-poc/releases/tag/v1.0)
contains the standalone ZIP bundle. Dataset photographs, annotation TXT files,
virtualenvs and generated inference outputs are excluded from Git and the bundle.
Dataset metadata and the five benchmark comparison visualizations are included.

## Evaluation

- Test mAP50: **74.1%**, mAP50–95: **49.1%** (Ultralytics validator).
- Fixed-threshold test recall: **88.7%**, precision: **61.8%**.
- Validator-selected recall: **83.1%**, precision: **71.6%**.

Metrics use different operating points. Angled carton localization and annotation
coverage remain known limitations; see the
[benchmark report](dena_poc/outputs/benchmark_eval/report.md).

See [project documentation](dena_poc/README.md),
[deployment agent instructions](dena_poc/AGENTS.md), and
[release manifest](dena_poc/release_manifest.json). The dataset must be restored
and `configs/data.yaml` rebased before validation or explicitly authorized retraining.
