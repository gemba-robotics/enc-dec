from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    stats: dict[str, float]
    method: str
    params: dict[str, Any]


def compute_stats(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    stats: dict[str, float] = {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(scores, p))
    return stats


def select_threshold(scores: np.ndarray, method: str, percentile: float, mean_std_k: float) -> ThresholdResult:
    if scores.size == 0:
        raise ValueError("No scores provided for threshold selection.")

    stats = compute_stats(scores)
    if method == "percentile":
        thr = float(np.percentile(scores, percentile))
        return ThresholdResult(
            threshold=thr,
            stats=stats,
            method=method,
            params={"percentile": float(percentile)},
        )
    if method == "mean_std":
        thr = float(stats["mean"] + mean_std_k * stats["std"])
        return ThresholdResult(
            threshold=thr,
            stats=stats,
            method=method,
            params={"k": float(mean_std_k)},
        )

    raise ValueError("threshold.method must be one of: percentile, mean_std")

