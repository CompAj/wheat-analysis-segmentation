import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .inference import measure


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FIELDS = ("filename", "status", "mask_source", "leaf_pixels", "rust_pixels",
          "coverage_pct", "on_leaf_detections", "artifact_dir")


def input_images(path):
    if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                name = Path(entry.filename)
                if entry.is_dir() or "__MACOSX" in name.parts or name.name.startswith("._"):
                    continue
                if name.suffix.lower() in EXTENSIONS:
                    yield entry.filename, archive.read(entry)
    else:
        yield path.name, path.read_bytes()


def decode(data):
    if data:
        try:
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            pass
    return None


def write_images(directory, image, leaf, rust):
    directory.mkdir(parents=True)
    leaf_png = leaf.astype("uint8") * 255
    overlay = image.copy()
    if rust.any():
        overlay[rust] = image[rust] * 0.5 + np.array([0, 0, 255]) * 0.5
    contours, _ = cv2.findContours(leaf_png, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), max(1, min(leaf.shape) // 600))
    for name, pixels in (("leaf.png", leaf_png), ("rust.png", rust.astype("uint8") * 255),
                         ("overlay.png", overlay)):
        success, encoded = cv2.imencode(".png", pixels)
        if not success:
            raise OSError(f"Could not encode {directory / name}")
        (directory / name).write_bytes(encoded.tobytes())


def run(path, output, models):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    rows = []
    for index, (name, data) in enumerate(input_images(path), 1):
        image = decode(data)
        row = {"filename": name, "status": "unreadable"}
        rows.append(row)
        if image is None:
            continue
        supplied = name.lower().endswith("_segmented.png")
        leaf = np.any(image != 0, axis=2) if supplied else models.leaf_mask(image)
        rust, metrics = measure(image, leaf, models)
        folder = f"images/{index:05d}"
        write_images(output / folder, image, leaf, rust)
        row.update(metrics, mask_source="nonblack_selection" if supplied else "leaf_model",
                   artifact_dir=folder)
    if not any(row["status"] != "unreadable" for row in rows):
        raise ValueError("Input contains no decodable images")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"schema_version": 1, "input_count": len(rows), **models.metadata()}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
