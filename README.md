# wheat-analysis-segmentation
Segments wheat and the different organs using different segementation models.

## GreenSkEye wheat rust job

This research job estimates the percentage of selected or detected leaf area covered by predicted rust. Inspect the cyan leaf boundary and red rust overlay before using a measurement. In the existing NWRD validation split, automatic leaf segmentation returned an empty mask on 16 of 18 photos. GreenSkEye users can select the leaf in the dashboard before submitting a job.

### Run

On a Linux CPU worker, from the repository root:

```sh
conda env create --prefix ./.conda-env --file environment.yml
conda run --prefix ./.conda-env python3 rust_analysis.py --input images.zip --output results
conda run --prefix ./.conda-env python3 -m unittest discover -s tests -v
```

Use an empty or new output directory. A single photo also works. A dashboard ZIP may contain original photos and black-background `*_segmented.png` files. For the latter the job treats every nonblack pixel as selected, so genuinely black pixels within a selection are excluded. Without a supplied selection the job uses the union of the leaf model's predicted segments. Images are processed sequentially.

### Results

- `summary.csv` contains one row per image, including its archive filename and artifact directory. Coverage is `100 × rust_pixels / leaf_pixels`. Overlapping detections count once toward area. The detection count includes each predicted instance that intersects the leaf.
- Each decoded image has source-resolution `leaf.png`, `rust.png`, and `overlay.png` files. White mask pixels contribute to the corresponding pixel count. Numbered directories distinguish repeated filenames.
- A valid leaf with zero predicted rust has 0% coverage. An empty leaf mask has status `no_leaf` and blank coverage. Undecodable images have status `unreadable`. Invalid archives, missing or altered checkpoints, and batches without a decodable image fail the job.
- `run.json` records package versions, inference parameters, checkpoint hashes, and the coverage method. The worker uploads the output directory as one ZIP.

### Model provenance and validation

Both FarmSpectra checkpoints are included and verified before inference. `wheat_rust/models/leaf.pt` comes from `models/data-trained/yolo11s/best.pt`. `wheat_rust/models/rust.pt` comes from `rust-trainer/output/best.pt`. The job uses CPU inference at 640 and 1024 pixels respectively, with confidence 0.20. It restores masks to the source grid before intersecting and counting pixels.

Across all 18 validation images, original-image inference had mean Dice **0.00844**, versus **0.00808** for inference on the masked photo. Both predictions were clipped to the same detected leaf region and compared with the full rust annotation. The two measurable images were `62.jpg` and `80.jpg`; their original-image Dice scores were 0.151 and 0.00095. The job uses original-image inference on ordinary photos. This split was available during model development. It establishes neither field accuracy nor accuracy on manually selected leaves.

The [per-image scores](evaluation/validation_dice.csv) and [archived FarmSpectra results](evaluation/farmspectra_baseline.csv) support comparisons. The latter are historical measurements taken from the original saved JSON. Reproduce the method comparison from this repository with the original data:

```sh
python -m evaluation.compare --validation ../rust-analysis/FarmSpectra/rust-trainer/data/val/images --raw ../rust-analysis/FarmSpectra/data/raw/NWRD_Raw --output scores.csv
```

### Field-image verification

A clean Linux CPU worker submitted through the scheduler produced a downloadable result for three original photos and one black-background selection. All four exported mask pairs reproduced their CSV coverage. Running the CLI on the same batch with the environment installed took 8.39 seconds and peaked at 1.05 GiB resident memory on the verification machine. Larger images and batches may differ.

Visual review found under-detection on `Rust1.png`: numerous orange spots are visible, but the model marked only 0.51% of the estimated leaf area. The supplied `lolr(121)_segmented.png` selection gave a measurable leaf and zero predicted rust despite visible discoloration. These images have no verified lesion masks. A small follow-up labeling effort would be most informative on missed leaf regions and on the small spots the detector overlooks.
