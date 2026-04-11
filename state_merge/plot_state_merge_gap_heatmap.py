import argparse
import math
import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from state_utils import _is_wkv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot layer-head heatmap of average gap between (S_A+S_B)/2 and S_AB."
    )
    parser.add_argument("--dataset-dir", type=str, default="./data")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--metric",
        type=str,
        default="mae",
        choices=["mae", "mse"],
        help="Per-head gap metric after spatial reduction.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Output image path. Default: result/state_merge_obs/{split}_avg_gap_layer_head_heatmap.png",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Limit number of two_state pairs to process (0 means all).",
    )
    return parser.parse_args()


def extract_wkv_by_layer(state: Any) -> List[torch.Tensor]:
    layer_map: Dict[int, torch.Tensor] = {}

    def walk(x: Any, path: tuple):
        if torch.is_tensor(x):
            if _is_wkv_path(path):
                layer_idx = path[0] // 3
                if x.ndim < 3:
                    raise ValueError(f"WKV tensor must be >=3D [head,...,H,W], got {tuple(x.shape)}")
                heads = int(math.prod(x.shape[:-2]))
                h, w = int(x.shape[-2]), int(x.shape[-1])
                layer_map[layer_idx] = x.reshape(heads, h, w).float()
            return
        if isinstance(x, (list, tuple)):
            for i, item in enumerate(x):
                walk(item, path + (i,))
            return
        raise TypeError(f"Unsupported state type: {type(x)}")

    walk(state, ())
    if not layer_map:
        raise ValueError("No WKV tensor leaves found in state.")
    return [layer_map[k] for k in sorted(layer_map.keys())]


def _reduce_gap(avg_state: torch.Tensor, target_state: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "mae":
        return (avg_state - target_state).abs().mean(dim=(-1, -2))
    if metric == "mse":
        return (avg_state - target_state).pow(2).mean(dim=(-1, -2))
    raise ValueError(f"Unsupported metric: {metric}")


def load_one_state_cache(one_state_dir: str) -> Dict[int, List[torch.Tensor]]:
    cache: Dict[int, List[torch.Tensor]] = {}
    names = sorted(name for name in os.listdir(one_state_dir) if name.endswith(".pt"))
    for name in tqdm(names, desc="Load one_state cache"):
        path = os.path.join(one_state_dir, name)
        item = torch.load(path, map_location="cpu")
        idx = int(item["index"])
        cache[idx] = extract_wkv_by_layer(item["state"])
    return cache


def compute_layer_head_gap(
    one_state_cache: Dict[int, List[torch.Tensor]],
    two_state_dir: str,
    metric: str,
    max_pairs: int = 0,
) -> torch.Tensor:
    pair_files = sorted(
        os.path.join(two_state_dir, name) for name in os.listdir(two_state_dir) if name.endswith(".pt")
    )
    if max_pairs > 0:
        pair_files = pair_files[:max_pairs]
    if not pair_files:
        raise ValueError(f"No pair files found in {two_state_dir}")

    sum_map: torch.Tensor | None = None
    cnt_map: torch.Tensor | None = None

    for pair_path in tqdm(pair_files, desc="Aggregate pair gaps"):
        pair_item = torch.load(pair_path, map_location="cpu")
        left_idx = int(pair_item["left_index"])
        right_idx = int(pair_item["right_index"])
        s_ab = extract_wkv_by_layer(pair_item["state"])
        s_a = one_state_cache[left_idx]
        s_b = one_state_cache[right_idx]

        if not (len(s_a) == len(s_b) == len(s_ab)):
            raise ValueError(
                f"Layer count mismatch at pair {pair_path}: A={len(s_a)}, B={len(s_b)}, AB={len(s_ab)}"
            )

        for layer_idx, (la, lb, lab) in enumerate(zip(s_a, s_b, s_ab)):
            if la.shape != lb.shape or la.shape != lab.shape:
                raise ValueError(
                    f"Shape mismatch at layer={layer_idx}, pair={pair_path}: "
                    f"A={tuple(la.shape)} B={tuple(lb.shape)} AB={tuple(lab.shape)}"
                )
            head_count = la.shape[0]
            avg = 0.5 * (la + lb)
            gap = _reduce_gap(avg, lab, metric=metric)

            if sum_map is None or cnt_map is None:
                max_layers = len(s_ab)
                max_heads = max(int(x.shape[0]) for x in s_ab)
                sum_map = torch.zeros(max_layers, max_heads, dtype=torch.float64)
                cnt_map = torch.zeros(max_layers, max_heads, dtype=torch.float64)

            sum_map[layer_idx, :head_count] += gap.to(dtype=torch.float64)
            cnt_map[layer_idx, :head_count] += 1.0

    assert sum_map is not None and cnt_map is not None
    valid = cnt_map > 0
    mean_map = torch.zeros_like(sum_map)
    mean_map[valid] = sum_map[valid] / cnt_map[valid]
    return mean_map


def plot_heatmap(values: torch.Tensor, output_path: str, metric: str, split: str):
    arr = values.cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    im = ax.imshow(arr, aspect="auto", origin="upper", cmap="Blues", vmin=0.0)

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Average {metric.upper()} of ((S_A + S_B)/2 - S_AB) | split={split}")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Gap ({metric.upper()})")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_mean_line(values: torch.Tensor, output_path: str, metric: str, split: str):
    layer_mean = values.mean(dim=1).cpu().numpy()
    x = list(range(len(layer_mean)))

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(x, layer_mean, color="#1f77b4", linewidth=2.0, marker="o", markersize=3.5)
    for xi, yi in zip(x, layer_mean):
        ax.text(
            xi,
            yi,
            f"{yi:.3f}",
            fontsize=8,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Mean Gap ({metric.upper()})")
    ax.set_title(f"Layer-wise Mean Gap of ((S_A + S_B)/2 - S_AB) | split={split}")
    max_x = max(len(layer_mean) - 1, 0)
    ax.set_xlim(-0.5, max_x + 0.5)
    y_min = float(layer_mean.min()) if len(layer_mean) > 0 else 0.0
    y_max = float(layer_mean.max()) if len(layer_mean) > 0 else 1.0
    if y_max > y_min:
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        base = y_max if y_max != 0 else 1.0
        ax.set_ylim(y_min - abs(base) * 0.08, y_max + abs(base) * 0.08)
    ax.margins(x=0.02, y=0.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    split_dir = os.path.join(args.dataset_dir, args.split)
    one_state_dir = os.path.join(split_dir, "one_state")
    two_state_dir = os.path.join(split_dir, "two_state")
    if not os.path.isdir(one_state_dir) or not os.path.isdir(two_state_dir):
        raise FileNotFoundError(
            f"Dataset split not found: one_state={one_state_dir}, two_state={two_state_dir}"
        )

    output_path = args.output_path.strip()
    if not output_path:
        output_path = os.path.join(
            "result",
            "state_merge_obs",
            f"{args.split}_avg_gap_layer_head_heatmap_{args.metric}.png",
        )
    line_output_path = output_path.replace("_heatmap_", "_layer_mean_line_")
    if line_output_path == output_path:
        root, ext = os.path.splitext(output_path)
        line_output_path = f"{root}_layer_mean_line{ext}"

    one_state_cache = load_one_state_cache(one_state_dir)
    mean_map = compute_layer_head_gap(
        one_state_cache=one_state_cache,
        two_state_dir=two_state_dir,
        metric=args.metric,
        max_pairs=args.max_pairs,
    )
    plot_heatmap(mean_map, output_path=output_path, metric=args.metric, split=args.split)
    plot_layer_mean_line(mean_map, output_path=line_output_path, metric=args.metric, split=args.split)
    print(f"Saved heatmap: {output_path}")
    print(f"Saved layer-mean line plot: {line_output_path}")
    print(f"Mean gap over all layer-head cells: {float(mean_map.mean().item()):.8f}")


if __name__ == "__main__":
    main()
