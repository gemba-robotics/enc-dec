from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from encdec.config_utils import deep_update, load_yaml, save_yaml
from encdec.data import AugmentConfig, GrayscaleImageDataset, filter_corrupted, list_images, save_json, build_transform
from encdec.model import ConvAutoencoder, ModelConfig
from encdec.runtime import make_run_dir, make_worker_init_fn, pick_device, save_checkpoint, seed_everything
from encdec.scoring import anomaly_score_mse, build_loss
from encdec.thresholding import select_threshold
from encdec.vis import save_recon_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train convolutional autoencoder for anomaly detection.")
    p.add_argument("--config", type=str, default="config.yaml")

    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--loss_type", type=str, default=None, choices=["mse", "smoothl1"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", default=None)
    p.add_argument("--no_deterministic", action="store_true", default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--amp", action="store_true", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    overrides = {}
    if args.data_root is not None:
        overrides = deep_update(overrides, {"data": {"root": args.data_root}})
    if args.img_size is not None:
        overrides = deep_update(overrides, {"preprocess": {"img_size": int(args.img_size)}})
    if args.epochs is not None:
        overrides = deep_update(overrides, {"training": {"epochs": int(args.epochs)}})
    if args.batch_size is not None:
        overrides = deep_update(overrides, {"training": {"batch_size": int(args.batch_size)}})
    if args.num_workers is not None:
        overrides = deep_update(overrides, {"training": {"num_workers": int(args.num_workers)}})
    if args.lr is not None:
        overrides = deep_update(overrides, {"training": {"lr": float(args.lr)}})
    if args.latent_dim is not None:
        overrides = deep_update(overrides, {"model": {"latent_dim": int(args.latent_dim)}})
    if args.loss_type is not None:
        overrides = deep_update(overrides, {"training": {"loss_type": args.loss_type}})
    if args.seed is not None:
        overrides = deep_update(overrides, {"training": {"seed": int(args.seed)}})
    if args.device is not None:
        overrides = deep_update(overrides, {"runtime": {"device": args.device}})
    if args.amp is not None:
        overrides = deep_update(overrides, {"training": {"amp": bool(args.amp)}})
    if args.no_deterministic:
        overrides = deep_update(overrides, {"training": {"deterministic": False}})
    if args.deterministic:
        overrides = deep_update(overrides, {"training": {"deterministic": True}})

    cfg = deep_update(cfg, overrides)

    data_root = Path(cfg["data"]["root"])
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]
    extensions = cfg["data"]["extensions"]
    verify_images = bool(cfg["data"].get("verify_images", True))

    img_size = int(cfg["preprocess"]["img_size"])
    if cfg["preprocess"].get("resize_policy", "direct") != "direct":
        raise ValueError("This implementation is configured for resize_policy=direct only.")

    augment_cfg = AugmentConfig(**cfg["preprocess"]["augment"])

    run_dir = make_run_dir(cfg["runtime"]["runs_dir"])
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "recons").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    save_yaml(run_dir / "config.yaml", cfg)

    seed = int(cfg["training"]["seed"])
    deterministic = bool(cfg["training"]["deterministic"])
    seed_everything(seed, deterministic)

    device = pick_device(cfg["runtime"]["device"])
    print(f"Device: {device}")

    train_paths = list_images(data_root / train_dir, extensions)
    val_paths = list_images(data_root / val_dir, extensions)

    if verify_images:
        train_paths = filter_corrupted(train_paths, run_dir / "logs" / "corrupt_train.txt")
        val_paths = filter_corrupted(val_paths, run_dir / "logs" / "corrupt_val.txt")

    if len(train_paths) == 0:
        raise RuntimeError(f"No training images found under: {data_root / train_dir}")
    if len(val_paths) == 0:
        raise RuntimeError(f"No validation images found under: {data_root / val_dir}")

    train_tf = build_transform(img_size=img_size, train=True, augment=augment_cfg)
    val_tf = build_transform(img_size=img_size, train=False, augment=AugmentConfig(enabled=False))

    train_ds = GrayscaleImageDataset(train_paths, transform=train_tf, labels=None)
    val_ds = GrayscaleImageDataset(val_paths, transform=val_tf, labels=None)

    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=make_worker_init_fn(seed),
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=make_worker_init_fn(seed + 10_000),
        generator=generator,
        drop_last=False,
    )

    model_cfg = ModelConfig(
        img_size=img_size,
        base_channels=int(cfg["model"]["base_channels"]),
        num_downsamples=int(cfg["model"]["num_downsamples"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
    )
    model = ConvAutoencoder(model_cfg).to(device)

    loss_fn = build_loss(cfg["training"]["loss_type"])
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    sched_cfg = cfg["training"]["lr_scheduler"]
    scheduler = None
    if sched_cfg["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(sched_cfg["cosine_tmax"]))
    elif sched_cfg["type"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(sched_cfg["step_size"]), gamma=float(sched_cfg["gamma"])
        )
    elif sched_cfg["type"] != "none":
        raise ValueError("training.lr_scheduler.type must be one of: none, cosine, step")

    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_epoch = -1
    rows = []

    epochs = int(cfg["training"]["epochs"])
    save_recon_every = int(cfg["training"]["save_recon_every_n_epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, _path in tqdm(train_loader, desc=f"train {epoch}/{epochs}"):
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                recon = model(x)
                loss = loss_fn(recon, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        val_scores = []
        val_paths_batch = []
        saved_recon = False
        with torch.no_grad():
            for x, paths in tqdm(val_loader, desc=f"val {epoch}/{epochs}"):
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    recon = model(x)
                    vloss = loss_fn(recon, x)
                val_losses.append(float(vloss.detach().cpu().item()))
                scores = anomaly_score_mse(x, recon).detach().cpu().numpy()
                val_scores.append(scores)
                val_paths_batch.extend(list(paths))

                if (not saved_recon) and save_recon_every > 0 and epoch % save_recon_every == 0:
                    save_recon_grid(run_dir / "recons" / f"epoch_{epoch:04d}.png", x, recon)
                    saved_recon = True

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        all_val_scores = np.concatenate(val_scores, axis=0) if len(val_scores) else np.array([], dtype=np.float32)

        score_stats = {
            "val_score_mean": float(all_val_scores.mean()) if all_val_scores.size else None,
            "val_score_std": float(all_val_scores.std(ddof=0)) if all_val_scores.size else None,
            "val_score_p99": float(np.percentile(all_val_scores, 99)) if all_val_scores.size else None,
        }

        lr_now = float(opt.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            **score_stats,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "logs" / "metrics.csv", index=False)

        print(json.dumps(row, indent=None))

        if scheduler is not None:
            scheduler.step()

        payload = {
            "epoch": epoch,
            "model_cfg": model_cfg.__dict__,
            "config": cfg,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
        }
        save_checkpoint(run_dir / "checkpoints" / "last.pt", payload)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(run_dir / "checkpoints" / "best.pt", payload)

    print(f"Best epoch: {best_epoch} (val_loss={best_val:.6f})")

    # Compute threshold from val scores using the best checkpoint
    best_ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state"])
    model.to(device)
    model.eval()

    all_scores = []
    all_paths = []
    with torch.no_grad():
        for x, paths in tqdm(val_loader, desc="val scores (best)"):
            x = x.to(device, non_blocking=True)
            recon = model(x)
            scores = anomaly_score_mse(x, recon).detach().cpu().numpy()
            all_scores.append(scores)
            all_paths.extend(list(paths))
    scores_np = np.concatenate(all_scores, axis=0)

    val_scores_df = pd.DataFrame({"path": all_paths, "score": scores_np})
    val_scores_df.to_csv(run_dir / "val_scores.csv", index=False)

    thr_cfg = cfg["threshold"]
    thr_res = select_threshold(
        scores=scores_np,
        method=str(thr_cfg["method"]),
        percentile=float(thr_cfg["percentile"]),
        mean_std_k=float(thr_cfg["mean_std_k"]),
    )
    save_json(
        run_dir / "threshold.json",
        {
            "threshold": thr_res.threshold,
            "method": thr_res.method,
            "params": thr_res.params,
            "val_score_stats": thr_res.stats,
            "best_epoch": best_epoch,
        },
    )
    print(f"Threshold: {thr_res.threshold:.6g} ({thr_res.method})")


if __name__ == "__main__":
    main()
