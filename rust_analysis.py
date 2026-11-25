import argparse
from pathlib import Path
from zipfile import BadZipFile

from wheat_rust.inference import Models
from wheat_rust.job import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure predicted rust coverage on wheat leaves.")
    parser.add_argument("--input", type=Path, required=True, help="RGB image or GreenSkEye ZIP")
    parser.add_argument("--output", type=Path, required=True, help="Empty result directory")
    args = parser.parse_args()
    try:
        run(args.input, args.output, Models())
    except (OSError, BadZipFile, ValueError) as exc:
        parser.exit(1, f"Rust analysis failed: {exc}\n")
