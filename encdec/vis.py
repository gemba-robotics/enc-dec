from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from torchvision.utils import make_grid, save_image
except Exception:  # pragma: no cover
    make_grid = None
    save_image = None


def save_recon_grid(path: str | Path, x: torch.Tensor, recon: torch.Tensor, max_items: int = 16) -> None:
    if make_grid is None or save_image is None:
        raise RuntimeError("torchvision is required for saving reconstruction grids.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = x[:max_items].detach().cpu()
    recon = recon[:max_items].detach().cpu()
    grid = torch.cat([x, recon], dim=0)  # 2N images
    grid = make_grid(grid, nrow=x.shape[0], padding=2)
    save_image(grid, path)


def save_triplet_images(
    out_dir: str | Path,
    stem: str,
    x: torch.Tensor,
    recon: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_np = x.detach().cpu().numpy()[0]  # (H,W)
    r_np = recon.detach().cpu().numpy()[0]
    err = np.abs(x_np - r_np)

    plt.imsave(out_dir / f"{stem}_input.png", x_np, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(out_dir / f"{stem}_recon.png", r_np, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(out_dir / f"{stem}_error.png", err, cmap="inferno")

