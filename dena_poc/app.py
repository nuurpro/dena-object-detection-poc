"""Local DENA detection UI. Run with the project virtualenv's Python interpreter."""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs/detect/dena_baseline/weights/best.pt"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "outputs/gradio_cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_settings"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
from ultralytics import YOLO

LOGGER = logging.getLogger("dena.app")


class DenaDetector:
    """Keep one model resident and serialize access to its predictor."""

    def __init__(self, weights: Path = WEIGHTS) -> None:
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Trained best.pt checkpoint is missing: {weights}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights)).to(self.device)
        if self.model.names != {0: "DENA"}:
            raise ValueError(f"Expected single class 0: DENA; got {self.model.names}")
        self.lock = threading.Lock()
        LOGGER.info("Loaded %s once on %s", weights, self.device)
        if self.device != "cpu":
            LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    def infer(self, image: Image.Image, conf: float, iou: float) -> tuple[np.ndarray, dict]:
        if not isinstance(image, Image.Image):
            raise ValueError("Upload or capture an image before detecting DENA.")
        for name, value, lower, upper in (
            ("Confidence", conf, .05, .95), ("NMS IoU", iou, .1, .9)
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} threshold must be between {lower} and {upper}.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        with self.lock:
            if self.device != "cpu":
                torch.cuda.synchronize()
            start = perf_counter()
            result = self.model.predict(
                image, conf=float(conf), iou=float(iou), imgsz=1024,
                device=self.device, classes=[0], max_det=1000, verbose=False,
            )[0]
            if self.device != "cpu":
                torch.cuda.synchronize()
            latency_ms = (perf_counter() - start) * 1000
            count = len(result.boxes) if result.boxes is not None else 0
            # Ultralytics plots are BGR; Gradio displays RGB arrays.
            annotated = result.plot(labels=True, conf=True)[..., ::-1].copy()
        LOGGER.info("Detected %d DENA packages in %.1f ms (conf=%.2f, iou=%.2f)",
                    count, latency_ms, conf, iou)
        return annotated, {"count": count, "inference_ms": round(latency_ms, 2)}


def build_demo(detector: DenaDetector) -> gr.Blocks:
    def detect(image, conf, iou):
        try:
            return detector.infer(image, conf, iou)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            LOGGER.exception("CUDA ran out of memory during inference")
            torch.cuda.empty_cache()
            raise gr.Error("GPU memory is full. Close other GPU apps and try again.") from exc
        except Exception as exc:
            LOGGER.exception("DENA inference failed")
            raise gr.Error("Detection failed. Check the server console for details.") from exc

    with gr.Blocks(title="DENA Detection PoC", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# DENA shelf detector\nUpload a shelf photo or use your camera, then click **Detect DENA**.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Shelf image", type="pil", image_mode="RGB",
                                 sources=["upload", "webcam"])
                conf = gr.Slider(.05, .95, value=.25, step=.05, label="Confidence threshold")
                iou = gr.Slider(.1, .9, value=.45, step=.05, label="NMS IoU threshold")
                button = gr.Button("Detect DENA", variant="primary")
            with gr.Column():
                annotated = gr.Image(label="Detected DENA packages", type="numpy",
                                     interactive=False, format="png")
                summary = gr.JSON(label="Detection summary (count and latency in ms)")
        button.click(detect, inputs=[image, conf, iou], outputs=[annotated, summary],
                     api_name="detect", concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        demo = build_demo(DenaDetector())
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                    inbrowser=False, show_error=False)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Unable to start DENA app: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
