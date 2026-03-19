import argparse
import math
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from tqdm import tqdm

from state_autoencoder import AutoEncoderConfig, StateStructureAutoEncoder, move_state_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot layer-wise WKV comparison between original and reconstructed states."
    )
    parser.add_argument(
        "--input-checkpoint",
        type=str,
        required=True,
        help="Path to AE checkpoint (e.g., best.pt).",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/data/.cache/data/val_index.json",
        help="Path to validation index json (e.g., val_index.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./encode_compress/plot",
        help="Directory to save plots.",
    )
    return parser.parse_args()


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (12, 8),
            "figure.dpi": 150,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_states_from_index(index_path: str):
    rows = torch.load(index_path) if index_path.endswith(".pt") else None
    if rows is not None:
        raise ValueError("Expected json index file, got .pt")
    import json

    with open(index_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    samples = []
    for row in tqdm(rows, desc="Load validation states"):
        sample = torch.load(row["file"], map_location="cpu", weights_only=False)
        samples.append(sample)
    return samples


def build_ae_from_checkpoint(checkpoint_path: str, sample_state, device: torch.device) -> StateStructureAutoEncoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = ckpt.get("config", {})
    config = AutoEncoderConfig(
        dropout=float(raw_config.get("dropout", 0.0)),
        expected_wkv_shape=tuple(raw_config.get("expected_wkv_shape", [32, 64, 64])),
    )
    ae = StateStructureAutoEncoder(config)
    ae.build_from_state(sample_state)
    ae.load_state_dict(ckpt["model_state_dict"], strict=True)
    ae.to(device)
    ae.eval()
    return ae


@torch.no_grad()
def reconstruct_state(ae: StateStructureAutoEncoder, state, device: torch.device):
    state_device = move_state_to_device(state, device)
    recon = ae.forward_state(state_device)
    return move_state_to_device(recon, torch.device("cpu"))


def assert_state_layout(state):
    if not isinstance(state, list) or len(state) % 3 != 0:
        raise ValueError("Unexpected RWKV state format. Expected list with length n_layer*3.")


def collect_layer_wkv_abs(state) -> List[np.ndarray]:
    assert_state_layout(state)
    num_layers = len(state) // 3
    out = []
    for layer_idx in range(num_layers):
        wkv = state[layer_idx * 3 + 1]
        if not torch.is_tensor(wkv) or wkv.ndim != 3:
            raise ValueError(
                f"Unexpected WKV at layer {layer_idx}. Expected Tensor[head,N,N], got {type(wkv)}"
            )
        out.append(wkv.detach().float().abs().cpu().numpy())
    return out


def flatten_per_head(matrix_by_head: np.ndarray) -> np.ndarray:
    # [num_heads, N, N] -> [num_heads, N*N]
    return matrix_by_head.reshape(matrix_by_head.shape[0], -1)


def plot_matrix_state_heads_0_to_3_3d(z_by_head: np.ndarray, title: str, out_path: str):
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
            ax.set_zlabel("|value|")
            ax.view_init(elev=30, azim=-55)
        else:
            ax.set_axis_off()

    fig.suptitle(title, y=0.96)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_state_mean_by_head(z_mean_by_head: np.ndarray, title: str, out_path: str):
    # z_mean_by_head: [num_heads, N, N]
    num_heads, _, _ = z_mean_by_head.shape
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
        cbar.set_label("|value|", fontsize=9)

    fig.subplots_adjust(top=0.90, wspace=0.45, hspace=0.45)
    fig.suptitle(title, y=0.97)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_states_from_index(args.input_data)
    if not samples:
        raise ValueError(f"No samples found in: {args.input_data}")

    ae = build_ae_from_checkpoint(args.input_checkpoint, samples[0]["state"], device)

    per_layer_orig_matrix: Dict[int, List[np.ndarray]] = {}
    per_layer_recon_matrix: Dict[int, List[np.ndarray]] = {}
    per_layer_orig_flat: Dict[int, List[np.ndarray]] = {}
    per_layer_recon_flat: Dict[int, List[np.ndarray]] = {}
    per_layer_diff_flat: Dict[int, List[np.ndarray]] = {}
    num_layers = None

    for row in tqdm(samples, desc="Reconstruct and collect WKV"):
        state = row["state"]
        recon_state = reconstruct_state(ae, state, device)

        orig_wkv_layers = collect_layer_wkv_abs(state)
        recon_wkv_layers = collect_layer_wkv_abs(recon_state)
        if num_layers is None:
            num_layers = len(orig_wkv_layers)
            for layer_idx in range(num_layers):
                per_layer_orig_matrix[layer_idx] = []
                per_layer_recon_matrix[layer_idx] = []
                per_layer_orig_flat[layer_idx] = []
                per_layer_recon_flat[layer_idx] = []
                per_layer_diff_flat[layer_idx] = []

        for layer_idx in range(num_layers):
            orig = orig_wkv_layers[layer_idx]
            recon = recon_wkv_layers[layer_idx]
            diff = np.abs(orig - recon)

            per_layer_orig_matrix[layer_idx].append(orig)
            per_layer_recon_matrix[layer_idx].append(recon)
            per_layer_orig_flat[layer_idx].append(flatten_per_head(orig))
            per_layer_recon_flat[layer_idx].append(flatten_per_head(recon))
            per_layer_diff_flat[layer_idx].append(flatten_per_head(diff))

    saved = []
    for layer_idx in tqdm(range(num_layers), desc="Plot layer-wise compare"):
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        z_orig = np.stack(per_layer_orig_flat[layer_idx], axis=0).astype(np.float32)
        z_recon = np.stack(per_layer_recon_flat[layer_idx], axis=0).astype(np.float32)
        z_diff = np.stack(per_layer_diff_flat[layer_idx], axis=0).astype(np.float32)

        path_orig_3d = os.path.join(layer_dir, "wkv_original_heads_0_1_2_3_3d.png")
        plot_matrix_state_heads_0_to_3_3d(
            z_orig, f"Layer {layer_idx} | Original WKV", path_orig_3d
        )
        saved.append(path_orig_3d)

        path_recon_3d = os.path.join(layer_dir, "wkv_reconstructed_heads_0_1_2_3_3d.png")
        plot_matrix_state_heads_0_to_3_3d(
            z_recon, f"Layer {layer_idx} | Reconstructed WKV", path_recon_3d
        )
        saved.append(path_recon_3d)

        path_diff_3d = os.path.join(layer_dir, "wkv_abs_error_heads_0_1_2_3_3d.png")
        plot_matrix_state_heads_0_to_3_3d(
            z_diff, f"Layer {layer_idx} | Abs Error |WKV - Recon|", path_diff_3d
        )
        saved.append(path_diff_3d)

        orig_mean = np.stack(per_layer_orig_matrix[layer_idx], axis=0).mean(axis=0).astype(np.float32)
        recon_mean = np.stack(per_layer_recon_matrix[layer_idx], axis=0).mean(axis=0).astype(np.float32)
        diff_mean = np.abs(orig_mean - recon_mean).astype(np.float32)

        path_orig_mean = os.path.join(layer_dir, "wkv_original_mean_by_head_2d.png")
        plot_matrix_state_mean_by_head(orig_mean, f"Layer {layer_idx} | Original WKV Mean", path_orig_mean)
        saved.append(path_orig_mean)

        path_recon_mean = os.path.join(layer_dir, "wkv_reconstructed_mean_by_head_2d.png")
        plot_matrix_state_mean_by_head(
            recon_mean, f"Layer {layer_idx} | Reconstructed WKV Mean", path_recon_mean
        )
        saved.append(path_recon_mean)

        path_diff_mean = os.path.join(layer_dir, "wkv_abs_error_mean_by_head_2d.png")
        plot_matrix_state_mean_by_head(
            diff_mean, f"Layer {layer_idx} | Abs Error Mean |WKV - Recon|", path_diff_mean
        )
        saved.append(path_diff_mean)

    print(f"Saved {len(saved)} plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
