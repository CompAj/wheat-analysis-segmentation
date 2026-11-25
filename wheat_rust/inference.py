import hashlib
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np


MODEL_DIR = Path(__file__).resolve().parent / "models"
LEAF_SIZE = 640
RUST_SIZE = 1024
CONFIDENCE = 0.20
METHOD = "original"


class Models:
    def __init__(self):
        from ultralytics import YOLO

        self.hashes = {}
        for line in (MODEL_DIR / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split()
            with (MODEL_DIR / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
            self.hashes[name] = digest
        self.leaf = YOLO(MODEL_DIR / "leaf.pt")
        self.rust = YOLO(MODEL_DIR / "rust.pt")

    def metadata(self):
        return {
            "checkpoints_sha256": self.hashes, "rust_input": METHOD, "device": "cpu",
            "leaf_imgsz": LEAF_SIZE, "rust_imgsz": RUST_SIZE, "confidence": CONFIDENCE,
            "retina_masks": True, "mask_grid": "inference size, nearest-neighbor to source",
            "packages": {
                name: version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python")
            },
        }

    @staticmethod
    def masks(model, image, size):
        # Bound per-instance mask memory on multi-megapixel field photos.
        height, width = image.shape[:2]
        factor = min(1.0, size / max(height, width))
        source = image
        if factor < 1:
            source = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))),
                                interpolation=cv2.INTER_LINEAR)
        result = model.predict(source, imgsz=size, conf=CONFIDENCE, device="cpu",
                               retina_masks=True, verbose=False, save=False)[0]
        if result.masks is not None:
            for mask in result.masks.data:
                binary = mask.cpu().numpy().astype(np.uint8)
                if binary.shape != image.shape[:2]:
                    binary = cv2.resize(binary, (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                yield binary != 0

    def leaf_mask(self, image):
        leaf = np.zeros(image.shape[:2], dtype=bool)
        for mask in self.masks(self.leaf, image, LEAF_SIZE):
            leaf |= mask
        return leaf

    def rust_masks(self, image):
        return self.masks(self.rust, image, RUST_SIZE)


def measure(image, leaf, models, method=METHOD):
    """Count instances intersecting the leaf and measure their union once."""
    leaf_pixels = int(np.count_nonzero(leaf))
    infected = np.zeros(leaf.shape, dtype=bool)
    count = 0
    if leaf_pixels:
        source = image if method == "original" else cv2.bitwise_and(image, image, mask=leaf.astype("uint8"))
        for mask in models.rust_masks(source):
            on_leaf = mask & leaf
            if on_leaf.any():
                count += 1
                infected |= on_leaf
    rust_pixels = int(np.count_nonzero(infected))
    return infected, {
        "status": "ok" if leaf_pixels else "no_leaf", "leaf_pixels": leaf_pixels,
        "rust_pixels": rust_pixels, "on_leaf_detections": count,
        "coverage_pct": round(100 * rust_pixels / leaf_pixels, 6) if leaf_pixels else None,
    }
