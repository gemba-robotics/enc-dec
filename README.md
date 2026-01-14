# enc-dec (grayscale autoencoder anomaly detection)

## Dataset layout

Expected folders under a dataset root:

- `train/` (good only)
- `val/` (good only, held out for threshold)
- `test_good/` (optional)
- `test_bad/` (optional)

Corrupted JPGs are skipped and logged during file listing/verification.

## Native setup

- Install Python deps: `python3 -m pip install -r requirements.txt`
- Install a platform-appropriate `torch` + `torchvision` separately.

## Train

`python3 train.py --config config.yaml --data_root /path/to/dataset`

Artifacts are written to `runs/<timestamp>/`:

- `runs/<timestamp>/checkpoints/best.pt`, `last.pt`
- `runs/<timestamp>/threshold.json` (computed from `val/` scores)
- `runs/<timestamp>/val_scores.csv`
- `runs/<timestamp>/logs/metrics.csv`
- `runs/<timestamp>/recons/epoch_*.png` (every N epochs)

## Score

Score the test folders (`test_good/` and `test_bad/`) and write a CSV:

`python3 score.py --checkpoint runs/<timestamp>/checkpoints/best.pt --data_root /path/to/dataset --output_csv runs/<timestamp>/test_scores.csv`

Optionally add predictions with a threshold:

`python3 score.py --checkpoint runs/<timestamp>/checkpoints/best.pt --data_root /path/to/dataset --output_csv runs/<timestamp>/test_scores.csv --threshold_file runs/<timestamp>/threshold.json`

Optionally save qualitative reconstructions + per-pixel error heatmaps:

`python3 score.py --checkpoint runs/<timestamp>/checkpoints/best.pt --data_root /path/to/dataset --output_csv runs/<timestamp>/test_scores.csv --save_qualitative`

## Eval

`python3 eval.py --csv runs/<timestamp>/test_scores.csv --threshold_file runs/<timestamp>/threshold.json`

Outputs to `runs/<timestamp>/eval/`.

## Smoke test (Jetson-friendly)

`python3 smoke_test.py --config config.yaml --data_root /path/to/dataset`

## Docker (Jetson Orin)

The Docker image is parameterized so you can match your JetPack/L4T version:

- Build: `docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:<tag> -t encdec:jetson .`
- Run (mount dataset + runs): `docker run --rm -it --runtime nvidia -v /path/to/dataset:/data:ro -v $PWD/runs:/workspace/runs encdec:jetson python3 train.py --config config.yaml --data_root /data`

JetPack 6.2.1 corresponds to L4T R36.4.x; pick a `l4t-pytorch` tag that matches your device’s `/etc/nv_tegra_release` (example: `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3`).

Optional `docker-compose.yml` usage:

- `export BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:<tag>`
- `export DATASET_DIR=/path/to/dataset`
- `docker compose run --rm encdec python3 train.py --config config.yaml --data_root /data`
