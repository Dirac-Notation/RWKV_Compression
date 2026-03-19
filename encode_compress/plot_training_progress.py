import argparse
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train/val loss from history.jsonl.")
    parser.add_argument(
        "--history",
        type=str,
        default="./encode_compress/checkpoints/history.jsonl",
        help="Path to training history jsonl file.",
    )
    return parser.parse_args()


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (14, 10),
            "figure.dpi": 150,
            "font.size": 22,
            "axes.labelsize": 26,
            "axes.titlesize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        }
    )


def load_history(history_path: str):
    # Keep latest row when epoch appears multiple times.
    epoch_to_row = OrderedDict()
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            epoch = int(row["epoch"])
            epoch_to_row[epoch] = row

    if not epoch_to_row:
        raise RuntimeError(f"No valid rows found in history file: {history_path}")

    epochs = np.array(sorted(epoch_to_row.keys()), dtype=np.int64)
    train_loss = np.array([float(epoch_to_row[e]["train_loss"]) for e in epochs], dtype=np.float64)
    val_loss = np.array([float(epoch_to_row[e]["val_loss"]) for e in epochs], dtype=np.float64)
    return epochs, train_loss, val_loss


def main():
    args = parse_args()
    if not os.path.isfile(args.history):
        raise FileNotFoundError(f"History file not found: {args.history}")

    apply_plot_style()
    epochs, train_loss, val_loss = load_history(args.history)

    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, figsize=(14, 10))

    ax1.plot(
        epochs,
        train_loss,
        marker="o",
        linewidth=4,
        markersize=8,
        color="#4C72B0",
        label="Train Loss",
        zorder=5,
    )
    ax1.set_title("Train Loss Over Epochs", pad=20)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.legend(frameon=False, loc="best")
    ax1.margins(x=0.02)

    ax2.plot(
        epochs,
        val_loss,
        marker="o",
        linewidth=4,
        markersize=8,
        color="#55A868",
        label="Validation Loss",
        zorder=5,
    )
    ax2.set_title("Validation Loss Over Epochs", pad=20)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.legend(frameon=False, loc="best")
    ax2.margins(x=0.02)

    output_path = os.path.join(os.path.dirname(os.path.abspath(args.history)), "training_progress.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
