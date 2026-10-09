"""Check installed libraries and CUDA operations without downloading weights."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save verification results as JSON")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "outputs/ultralytics_settings"
    # Ultralytics checks the parent for writability before creating its directory.
    config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = root / "outputs/matplotlib_cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import PIL
    import yaml
    import torch
    import torchvision
    import ultralytics

    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("Pillow", "PyYAML", "torch", "torchvision", "ultralytics", "pytest")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "weights_downloaded": False,
        "training_started": False,
    }
    if not report["cuda_available"]:
        print(json.dumps(report, indent=2))
        raise RuntimeError("CUDA unavailable; GPU environment verification failed")
    properties = torch.cuda.get_device_properties(0)
    report["gpu"] = properties.name
    report["gpu_memory_mib"] = properties.total_memory // (1024 * 1024)
    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    assert torch.equal(sample @ torch.eye(2, device="cuda"), sample)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    assert torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist() == [0]
    torch.cuda.synchronize()
    report["cuda_tensor_and_nms_checks"] = "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
