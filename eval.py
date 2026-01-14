from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate anomaly scores and threshold.")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--threshold_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--tpr_target", type=float, default=None, help="Optional: report FPR at this TPR.")
    return p.parse_args()


def _auto_threshold_file(csv_path: Path) -> Path | None:
    for parent in [csv_path.parent, csv_path.parent.parent]:
        cand = parent / "threshold.json"
        if cand.exists():
            return cand
    return None


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold
    if threshold is None:
        thr_file = Path(args.threshold_file) if args.threshold_file else _auto_threshold_file(csv_path)
        if thr_file is not None and thr_file.exists():
            with open(thr_file, "r", encoding="utf-8") as f:
                threshold = float(json.load(f)["threshold"])

    if threshold is not None:
        df["predicted_bad"] = (df["score"] > float(threshold)).astype(int)

    df.to_csv(out_dir / "scored.csv", index=False)

    has_labels = "label" in df.columns and df["label"].notna().all() and set(df["label"].unique()).issubset({0, 1})

    summary: dict[str, object] = {
        "n": int(len(df)),
        "threshold": float(threshold) if threshold is not None else None,
    }

    plt.figure(figsize=(8, 5))
    if "label" in df.columns and has_labels:
        for y, name in [(0, "good"), (1, "bad")]:
            sub = df[df["label"] == y]["score"].to_numpy()
            if sub.size:
                plt.hist(sub, bins=50, alpha=0.6, label=name, density=True)
    else:
        plt.hist(df["score"].to_numpy(), bins=50, alpha=0.8, label="scores", density=True)
    if threshold is not None:
        plt.axvline(float(threshold), color="red", linestyle="--", label="threshold")
    plt.xlabel("anomaly score")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_hist.png", dpi=150)
    plt.close()

    if has_labels and df["label"].nunique() == 2:
        y_true = df["label"].astype(int).to_numpy()
        y_score = df["score"].astype(float).to_numpy()

        summary["auroc"] = float(roc_auc_score(y_true, y_score))
        summary["auprc"] = float(average_precision_score(y_true, y_score))

        fpr, tpr, _thr = roc_curve(y_true, y_score)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUROC={summary['auroc']:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc.png", dpi=150)
        plt.close()

        prec, rec, _thr2 = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(rec, prec))
        summary["pr_auc"] = pr_auc
        plt.figure(figsize=(6, 6))
        plt.plot(rec, prec, label=f"AUPRC={summary['auprc']:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "pr.png", dpi=150)
        plt.close()

        if threshold is not None:
            y_pred = (y_score > float(threshold)).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            summary["confusion_matrix"] = cm.tolist()

        if args.tpr_target is not None:
            target = float(args.tpr_target)
            idx = int(np.argmin(np.abs(tpr - target)))
            summary["fpr_at_tpr"] = {"tpr": float(tpr[idx]), "fpr": float(fpr[idx])}

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
