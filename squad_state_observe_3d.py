import argparse
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from tqdm import tqdm

from rwkv_model import default_state_dir, load_validation_state_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Observe RWKV states in 3D plots by layer.")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-7.2b-20260301-ctx8192.pth")
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="./squad_state_observe_3d_outputs")
    return parser.parse_args()


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (12, 8),
            "figure.dpi": 150,
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def flatten_abs_channel(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().abs().reshape(-1).cpu().numpy()


def plot_vector_state(z_matrix: np.ndarray, title: str, out_path: str):
    sample_count, channel_count = z_matrix.shape
    x = np.arange(sample_count, dtype=np.float32)
    y = np.arange(channel_count, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, z_matrix, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("data idx")
    ax.set_ylabel("channel")
    ax.set_zlabel("|state|")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-55)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_state_by_head(z_by_head: np.ndarray, layer_idx: int, out_path: str):
    # z_by_head: [num_samples, num_heads, channels_per_head]
    sample_count, num_heads, channels_per_head = z_by_head.shape
    cols = min(8, num_heads)
    rows = math.ceil(num_heads / cols)
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))

    x = np.arange(sample_count, dtype=np.float32)
    y = np.arange(channels_per_head, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    for h in range(num_heads):
        ax = fig.add_subplot(rows, cols, h + 1, projection="3d")
        zz = z_by_head[:, h, :]
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"head={h}", fontsize=10)
        ax.set_xlabel("data idx")
        ax.set_ylabel("channel")
        ax.set_zlabel("|state|")
        ax.view_init(elev=30, azim=-55)

    fig.suptitle(f"RWKV Matrix State by Head (layer={layer_idx}, slot=1)", y=0.995)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_state_heads_0_to_3_3d(z_by_head: np.ndarray, layer_idx: int, out_path: str):
    # z_by_head: [num_samples, num_heads, channels_per_head]
    sample_count, num_heads, channels_per_head = z_by_head.shape
    selected = [h for h in [0, 1, 2, 3] if h < num_heads]
    if not selected:
        return

    fig = plt.figure(figsize=(12, 10))
    x = np.arange(sample_count, dtype=np.float32)
    y = np.arange(channels_per_head, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    for panel in range(4):
        ax = fig.add_subplot(2, 2, panel + 1, projection="3d")
        if panel < len(selected):
            head_idx = selected[panel]
            zz = z_by_head[:, head_idx, :]
            ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True)
            ax.set_title(f"head={head_idx}", fontsize=10)
            ax.set_xlabel("data idx")
            ax.set_ylabel("channel")
            ax.set_zlabel("|state|")
            ax.view_init(elev=30, azim=-55)
        else:
            ax.set_axis_off()

    fig.suptitle(f"Layer {layer_idx}", y=0.96)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_state_mean_by_head(z_mean_by_head: np.ndarray, layer_idx: int, out_path: str):
    # z_mean_by_head: [num_heads, N, N]
    num_heads, rows_n, cols_n = z_mean_by_head.shape
    cols = min(8, num_heads)
    rows = math.ceil(num_heads / cols)
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))

    for h in range(num_heads):
        ax = fig.add_subplot(rows, cols, h + 1)
        img = ax.imshow(
            z_mean_by_head[h],
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )
        ax.set_title(f"head={h}", fontsize=10)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|state|", fontsize=9)

    # Reserve explicit top margin so suptitle never overlaps.
    fig.subplots_adjust(top=0.90, wspace=0.45, hspace=0.45)
    fig.suptitle(f"RWKV Matrix State Mean by Head (layer={layer_idx}, slot=1)", y=0.97)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_state_mean_heads_0_to_3_2d(z_mean_by_head: np.ndarray, layer_idx: int, out_path: str):
    # z_mean_by_head: [num_heads, N, N]
    num_heads, rows_n, cols_n = z_mean_by_head.shape
    del rows_n, cols_n
    selected = [h for h in [0, 1, 2, 3] if h < num_heads]
    if not selected:
        return

    fig = plt.figure(figsize=(12, 10))
    for panel in range(4):
        ax = fig.add_subplot(2, 2, panel + 1)
        if panel < len(selected):
            head_idx = selected[panel]
            img = ax.imshow(
                z_mean_by_head[head_idx],
                cmap="viridis",
                aspect="auto",
                origin="lower",
            )
            ax.set_title(f"head={head_idx}", fontsize=10)
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("|state|", fontsize=9)
        else:
            ax.set_axis_off()

    fig.subplots_adjust(top=0.90, wspace=0.35, hspace=0.35)
    fig.suptitle(f"Layer {layer_idx}", y=0.96)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_plot_style()
    state_dir = args.state_dir or default_state_dir(args.model_filename)
    dataset = load_validation_state_dataset(state_dir=state_dir, limit=args.num_samples)

    layer_slot_vectors: Dict[int, Dict[int, List[np.ndarray]]] = {}
    layer_slot1_heads: Dict[int, List[np.ndarray]] = {}
    layer_slot1_matrices: Dict[int, List[np.ndarray]] = {}
    num_layers = None

    for idx in tqdm(range(len(dataset)), desc="Load states for observation"):
        row = dataset[idx]
        state = row["state"]
        if num_layers is None:
            num_layers = len(state) // 3
            for layer_idx in range(num_layers):
                layer_slot_vectors[layer_idx] = {0: [], 1: [], 2: []}
                layer_slot1_heads[layer_idx] = []
                layer_slot1_matrices[layer_idx] = []

        for layer_idx in range(num_layers):
            for slot in [0, 1, 2]:
                leaf = state[layer_idx * 3 + slot]
                if slot == 1:
                    # slot 1 is matrix-valued state: [head, N, N]
                    leaf_abs = leaf.detach().float().abs().cpu().numpy()
                    layer_slot1_matrices[layer_idx].append(leaf_abs)
                    per_head_flat = leaf.detach().float().abs().reshape(leaf.shape[0], -1).cpu().numpy()
                    layer_slot1_heads[layer_idx].append(per_head_flat)
                    layer_slot_vectors[layer_idx][slot].append(per_head_flat.reshape(-1))
                else:
                    flat = flatten_abs_channel(leaf)
                    layer_slot_vectors[layer_idx][slot].append(flat)

    saved = []
    for layer_idx in tqdm(range(num_layers), desc="Plot by layer"):
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        z_slot0 = np.stack(layer_slot_vectors[layer_idx][0], axis=0).astype(np.float32)
        path_slot0 = os.path.join(layer_dir, "slot0_time_mixing_shift_state_3d.png")
        plot_vector_state(
            z_slot0,
            title=f"RWKV State 3D | layer={layer_idx} slot=0 (Time Mixing Shift)",
            out_path=path_slot0,
        )
        saved.append(path_slot0)

        z_slot2 = np.stack(layer_slot_vectors[layer_idx][2], axis=0).astype(np.float32)
        path_slot2 = os.path.join(layer_dir, "slot2_channel_mixing_shift_state_3d.png")
        plot_vector_state(
            z_slot2,
            title=f"RWKV State 3D | layer={layer_idx} slot=2 (Channel Mixing Shift)",
            out_path=path_slot2,
        )
        saved.append(path_slot2)

        z_slot1_by_head = np.stack(layer_slot1_heads[layer_idx], axis=0).astype(np.float32)
        path_slot1 = os.path.join(layer_dir, "slot1_matrix_wkv_state_by_head_3d.png")
        plot_matrix_state_by_head(z_slot1_by_head, layer_idx, path_slot1)
        saved.append(path_slot1)
        path_slot1_h0123_3d = os.path.join(layer_dir, "slot1_matrix_wkv_state_heads_0_1_2_3_3d.png")
        plot_matrix_state_heads_0_to_3_3d(z_slot1_by_head, layer_idx, path_slot1_h0123_3d)
        saved.append(path_slot1_h0123_3d)

        z_slot1_matrix_mean = np.stack(layer_slot1_matrices[layer_idx], axis=0).astype(np.float32).mean(axis=0)
        path_slot1_mean = os.path.join(layer_dir, "slot1_matrix_wkv_state_mean_by_head_2d.png")
        plot_matrix_state_mean_by_head(z_slot1_matrix_mean, layer_idx, path_slot1_mean)
        saved.append(path_slot1_mean)
        path_slot1_mean_h0123_2d = os.path.join(layer_dir, "slot1_matrix_wkv_state_mean_heads_0_1_2_3_2d.png")
        plot_matrix_state_mean_heads_0_to_3_2d(z_slot1_matrix_mean, layer_idx, path_slot1_mean_h0123_2d)
        saved.append(path_slot1_mean_h0123_2d)

    print(f"Saved {len(saved)} layer plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
