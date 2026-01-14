from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
except Exception:  # pragma: no cover
    T = None
    InterpolationMode = None


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = False
    rotation_deg: float = 5.0
    translate_frac: float = 0.02
    brightness: float = 0.05
    contrast: float = 0.05


def _is_image_path(path: Path, extensions: Sequence[str]) -> bool:
    return path.is_file() and any(str(path).endswith(ext) for ext in extensions)


def list_images(folder: str | Path, extensions: Sequence[str], recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []

    if recursive:
        paths = [p for p in folder.rglob("*") if _is_image_path(p, extensions)]
    else:
        paths = [p for p in folder.iterdir() if _is_image_path(p, extensions)]
    return sorted(paths)


def try_decode_grayscale(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("L")
            img.load()
            return img
    except Exception:
        return None


def filter_corrupted(
    paths: Iterable[Path],
    corrupt_log_path: str | Path | None,
) -> list[Path]:
    valid: list[Path] = []
    corrupted: list[str] = []

    for p in paths:
        img = try_decode_grayscale(p)
        if img is None:
            corrupted.append(str(p))
        else:
            valid.append(p)

    if corrupt_log_path is not None:
        corrupt_log_path = Path(corrupt_log_path)
        corrupt_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corrupt_log_path, "w", encoding="utf-8") as f:
            for item in corrupted:
                f.write(item + "\n")

    return valid


def build_transform(
    img_size: int,
    train: bool,
    augment: AugmentConfig,
) -> Callable[[Image.Image], torch.Tensor]:
    if T is None:
        raise RuntimeError("torchvision is required for transforms (install torchvision).")

    ops: list[Callable] = []

    if train and augment.enabled:
        affine_common = dict(
            degrees=augment.rotation_deg,
            translate=(augment.translate_frac, augment.translate_frac),
        )
        try:
            ops.append(
                T.RandomAffine(
                    **affine_common,
                    interpolation=InterpolationMode.BILINEAR if InterpolationMode else Image.BILINEAR,
                    fill=0,
                )
            )
        except TypeError:
            try:
                ops.append(
                    T.RandomAffine(
                        **affine_common,
                        resample=Image.BILINEAR,
                        fillcolor=0,
                    )
                )
            except TypeError:
                ops.append(T.RandomAffine(**affine_common))
        ops.append(T.ColorJitter(brightness=augment.brightness, contrast=augment.contrast))

    resize_kwargs = {
        "size": (img_size, img_size),
        "interpolation": InterpolationMode.BILINEAR if InterpolationMode else Image.BILINEAR,
    }
    try:
        ops.append(T.Resize(**resize_kwargs, antialias=True))
    except TypeError:  # older torchvision
        ops.append(T.Resize(**resize_kwargs))
    ops.append(T.ToTensor())  # float32 in [0,1], shape (1,H,W) for mode "L"

    transform = T.Compose(ops)
    return transform


class GrayscaleImageDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        transform: Callable[[Image.Image], torch.Tensor],
        labels: Sequence[int] | None = None,
    ) -> None:
        if labels is not None and len(labels) != len(paths):
            raise ValueError("labels must be same length as paths")
        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = try_decode_grayscale(path)
        if img is None:
            raise RuntimeError(f"Failed to decode image (should have been filtered): {path}")
        tensor = self.transform(img)
        path_str = str(path)
        if self.labels is None:
            return tensor, path_str
        return tensor, path_str, int(self.labels[idx])


def split_test_paths(
    data_root: str | Path,
    test_good_dir: str,
    test_bad_dir: str,
    extensions: Sequence[str],
    verify_images: bool,
    corrupt_log_path: str | Path | None,
) -> tuple[list[Path], list[int]]:
    data_root = Path(data_root)
    good_paths = list_images(data_root / test_good_dir, extensions)
    bad_paths = list_images(data_root / test_bad_dir, extensions)

    paths = good_paths + bad_paths
    labels = [0] * len(good_paths) + [1] * len(bad_paths)

    if verify_images:
        paths = filter_corrupted(paths, corrupt_log_path)
        kept = {str(p) for p in paths}
        labels = [lbl for p, lbl in zip(good_paths + bad_paths, labels) if str(p) in kept]

    return paths, labels


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    out = {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(scores, p))
    return out


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
