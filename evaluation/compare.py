"""Compare both inference paths on the existing FarmSpectra validation split."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from wheat_rust.inference import Models, measure
from wheat_rust.job import EXTENSIONS


def compare(validation, raw, output):
    models = Models()
    rows = []
    for path in sorted(validation.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        matches = [p for p in raw.glob(f"*/images/{path.name}")
                   if p.read_bytes() == path.read_bytes()]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one matching raw image for {path.name}")
        truth_path = matches[0].parents[1] / "masks" / (path.stem + ".png")
        truth = cv2.imdecode(np.frombuffer(truth_path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if truth is None or image is None or truth.shape != image.shape[:2]:
            raise ValueError(f"Invalid image or reference mask for {path.name}")
        truth = truth > 127
        leaf = models.leaf_mask(image)
        row = {"filename": path.name, "leaf_pixels": int(leaf.sum()), "reference_pixels": int(truth.sum())}
        for method in ("original", "masked"):
            prediction, _ = measure(image, leaf, models, method)
            total = int(prediction.sum()) + row["reference_pixels"]
            row[method] = 2 * int((prediction & truth).sum()) / total if total else 1.0
        rows.append(row)
    if not rows:
        raise ValueError("Validation set is empty")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    scores = {method: mean(row[method] for row in rows) for method in ("original", "masked")}
    print(json.dumps({"images": len(rows), "no_leaf": sum(r["leaf_pixels"] == 0 for r in rows),
                      "mean_dice": scores, "selected": max(scores, key=scores.get)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True, help="rust-trainer/data/val/images")
    parser.add_argument("--raw", type=Path, required=True, help="data/raw/NWRD_Raw")
    parser.add_argument("--output", type=Path, required=True, help="Per-image results CSV")
    args = parser.parse_args()
    compare(args.validation, args.raw, args.output)
