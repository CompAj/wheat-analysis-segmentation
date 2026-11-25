import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from wheat_rust.inference import measure
from wheat_rust.job import run


class StubModels:
    def __init__(self, masks=()):
        self.masks = masks
        self.leaf_calls = 0

    def rust_masks(self, image):
        return iter(self.masks)

    def leaf_mask(self, image):
        self.leaf_calls += 1
        return np.ones(image.shape[:2], dtype=bool)

    def metadata(self):
        return {"test": True}


class RustJobTests(unittest.TestCase):
    def test_union_count_and_denominator(self):
        leaf = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
        first = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
        second = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
        rust, metrics = measure(np.zeros((2, 3, 3), np.uint8), leaf, StubModels([first, second, ~leaf]))
        self.assertEqual(metrics["coverage_pct"], 75)
        self.assertEqual(metrics["on_leaf_detections"], 2)
        self.assertEqual(int(rust.sum()), 3)

    def test_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "images.zip"
            image = np.full((3, 4, 3), 90, dtype=np.uint8)
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in ("a.jpg", "a.png", "manual_segmented.png"):
                    archive.writestr(name, cv2.imencode(Path(name).suffix, image)[1].tobytes())
                archive.writestr("empty_segmented.png", cv2.imencode(".png", image * 0)[1].tobytes())
                archive.writestr("bad.jpg", b"broken")
            models = StubModels()
            out = root / "out"
            run(archive_path, out, models)
            with (out / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(models.leaf_calls, 2)
            self.assertEqual([r["status"] for r in rows], ["ok", "ok", "ok", "no_leaf", "unreadable"])
            self.assertEqual(rows[2]["mask_source"], "nonblack_selection")
            self.assertEqual(float(rows[0]["coverage_pct"]), 0)
            self.assertEqual(rows[3]["coverage_pct"], "")
            self.assertNotEqual(rows[0]["artifact_dir"], rows[1]["artifact_dir"])
            self.assertTrue((out / rows[0]["artifact_dir"] / "overlay.png").is_file())
            self.assertEqual(json.loads((out / "run.json").read_text())["input_count"], 5)

    def test_failed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "broken.zip"
            path.write_bytes(b"bad")
            with self.assertRaises(zipfile.BadZipFile):
                run(path, root / "out", StubModels())
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("broken.jpg", b"")
            with self.assertRaisesRegex(ValueError, "no decodable images"):
                run(path, root / "out", StubModels())
