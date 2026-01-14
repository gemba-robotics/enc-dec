from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from encdec.config_utils import deep_update, load_yaml
from encdec.data import AugmentConfig, GrayscaleImageDataset, filter_corrupted, list_images, build_transform
from encdec.model import ConvAutoencoder, ModelConfig
from encdec.runtime import make_worker_init_fn, pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jetson-friendly smoke test: one batch + forward pass.")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--checkpoint", type=str, default=None, help="Optional: load model weights.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, {"data": {"root": args.data_root}})
    if args.device is not None:
        cfg = deep_update(cfg, {"runtime": {"device": args.device}})
    if args.batch_size is not None:
        cfg = deep_update(cfg, {"training": {"batch_size": int(args.batch_size)}})
    if args.num_workers is not None:
        cfg = deep_update(cfg, {"training": {"num_workers": int(args.num_workers)}})

    seed_everything(int(cfg["training"]["seed"]), bool(cfg["training"]["deterministic"]))
    device = pick_device(cfg["runtime"]["device"])
    print(f"Device: {device}")

    data_root = Path(cfg["data"]["root"])
    train_dir = cfg["data"]["train_dir"]
    extensions = cfg["data"]["extensions"]
    verify_images = bool(cfg["data"].get("verify_images", True))

    paths = list_images(data_root / train_dir, extensions)
    if verify_images:
        paths = filter_corrupted(paths, corrupt_log_path=None)
    if len(paths) == 0:
        raise RuntimeError("No images found for smoke test.")

    img_size = int(cfg["preprocess"]["img_size"])
    tf = build_transform(img_size=img_size, train=False, augment=AugmentConfig(enabled=False))
    ds = GrayscaleImageDataset(paths, transform=tf, labels=None)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=make_worker_init_fn(int(cfg["training"]["seed"]) + 30_000),
        drop_last=False,
    )

    model_cfg = ModelConfig(
        img_size=img_size,
        base_channels=int(cfg["model"]["base_channels"]),
        num_downsamples=int(cfg["model"]["num_downsamples"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
    )
    model = ConvAutoencoder(model_cfg).to(device).eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    x, _paths = next(iter(loader))
    x = x.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    imgs_per_sec = x.shape[0] / max(dt, 1e-9)
    print(f"Batch size: {x.shape[0]}")
    print(f"Batch time (s): {dt:.4f}")
    print(f"Images/sec: {imgs_per_sec:.2f}")


if __name__ == "__main__":
    main()

