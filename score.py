from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from encdec.data import (
    AugmentConfig,
    GrayscaleImageDataset,
    build_transform,
    filter_corrupted,
    list_images,
    split_test_paths,
    try_decode_grayscale,
)
from encdec.model import ConvAutoencoder, ModelConfig
from encdec.runtime import make_worker_init_fn, pick_device
from encdec.scoring import anomaly_score_mse
from encdec.vis import save_triplet_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score images with a trained autoencoder.")
    p.add_argument("--checkpoint", type=str, required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_root", type=str, help="Dataset root containing test_good/ and optionally test_bad/.")
    group.add_argument("--input_dir", type=str, help="Folder of images to score (labels omitted).")
    group.add_argument("--manifest_csv", type=str, help="CSV with a 'path' column and optional 'label' column.")
    p.add_argument("--manifest_base_dir", type=str, default=None, help="Prepended to manifest paths if provided.")

    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--threshold", type=float, default=None, help="If set, adds predicted_bad column.")
    p.add_argument("--threshold_file", type=str, default=None, help="JSON file with a 'threshold' field.")

    p.add_argument("--save_qualitative", action="store_true")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--qualitative_dir", type=str, default=None)
    return p.parse_args()


def _safe_stem(path_str: str) -> str:
    p = Path(path_str)
    h = hashlib.sha1(path_str.encode("utf-8")).hexdigest()[:10]
    return f"{p.stem}_{h}"


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    device_str = args.device if args.device is not None else cfg["runtime"]["device"]
    device = pick_device(device_str)

    img_size = int(cfg["preprocess"]["img_size"])
    if cfg["preprocess"].get("resize_policy", "direct") != "direct":
        raise ValueError("This implementation is configured for resize_policy=direct only.")

    model_cfg = ModelConfig(
        img_size=img_size,
        base_channels=int(cfg["model"]["base_channels"]),
        num_downsamples=int(cfg["model"]["num_downsamples"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
    )
    model = ConvAutoencoder(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    extensions = cfg["data"]["extensions"]
    verify_images = bool(cfg["data"].get("verify_images", True))

    if args.data_root:
        paths, labels = split_test_paths(
            data_root=args.data_root,
            test_good_dir=cfg["data"]["test_good_dir"],
            test_bad_dir=cfg["data"]["test_bad_dir"],
            extensions=extensions,
            verify_images=verify_images,
            corrupt_log_path=Path(args.output_csv).with_suffix(".corrupt.txt"),
        )
    elif args.input_dir:
        paths = list_images(args.input_dir, extensions)
        if verify_images:
            paths = filter_corrupted(paths, Path(args.output_csv).with_suffix(".corrupt.txt"))
        labels = None
    else:
        manifest = pd.read_csv(args.manifest_csv)
        if "path" not in manifest.columns:
            raise ValueError("manifest_csv must include a 'path' column.")
        base = Path(args.manifest_base_dir) if args.manifest_base_dir else None
        raw_paths = [Path(p) for p in manifest["path"].astype(str).tolist()]
        paths = [base / p if base is not None and not p.is_absolute() else p for p in raw_paths]
        labels = manifest["label"].tolist() if "label" in manifest.columns else None
        if verify_images:
            before = list(paths)
            paths = filter_corrupted(paths, Path(args.output_csv).with_suffix(".corrupt.txt"))
            if labels is not None:
                kept = {str(p) for p in paths}
                labels = [y for p, y in zip(before, labels) if str(p) in kept]

    if len(paths) == 0:
        raise RuntimeError("No images found to score.")

    tf = build_transform(img_size=img_size, train=False, augment=AugmentConfig(enabled=False))
    ds = GrayscaleImageDataset(paths, transform=tf, labels=labels)

    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg["training"]["batch_size"])
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg["training"]["num_workers"])

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=make_worker_init_fn(int(cfg["training"]["seed"]) + 20_000),
        drop_last=False,
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring"):
            if labels is None:
                x, paths_b = batch
                labels_b = None
            else:
                x, paths_b, labels_b = batch

            x = x.to(device, non_blocking=True)
            recon = model(x)
            scores = anomaly_score_mse(x, recon).detach().cpu().numpy()

            if labels_b is None:
                for p, s in zip(paths_b, scores):
                    rows.append({"path": p, "label": None, "score": float(s)})
            else:
                for p, y, s in zip(paths_b, labels_b, scores):
                    rows.append({"path": p, "label": int(y), "score": float(s)})

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    threshold = None
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif args.threshold_file is not None:
        with open(args.threshold_file, "r", encoding="utf-8") as f:
            threshold = float(json.load(f)["threshold"])
    else:
        run_dir = ckpt_path.parent.parent  # runs/<timestamp> when checkpoint is in runs/<timestamp>/checkpoints/
        cand = run_dir / "threshold.json"
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                threshold = float(json.load(f)["threshold"])

    if threshold is not None:
        df["predicted_bad"] = (df["score"] > threshold).astype(int)
        df.to_csv(args.output_csv, index=False)

    if args.save_qualitative and labels is not None:
        out_dir = Path(args.qualitative_dir) if args.qualitative_dir else Path(args.output_csv).parent / "qualitative"
        out_dir.mkdir(parents=True, exist_ok=True)

        good = df[df["label"] == 0].sort_values("score", ascending=False).head(int(args.topk))
        bad = df[df["label"] == 1].sort_values("score", ascending=True).head(int(args.topk))

        selected = pd.concat([good.assign(_group="good_high"), bad.assign(_group="bad_low")], axis=0)

        for _, row in tqdm(selected.iterrows(), total=len(selected), desc="saving qualitative"):
            path_str = row["path"]
            group = row["_group"]
            stem = _safe_stem(path_str)

            img = try_decode_grayscale(Path(path_str))
            if img is None:
                continue
            x = tf(img)
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                recon = model(x)
            save_triplet_images(out_dir / group, stem, x[0].cpu(), recon[0].cpu())


if __name__ == "__main__":
    main()
