import argparse
import json
import os
import re
from collections import OrderedDict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
from tqdm import tqdm

from rwkv_model import (
    RWKVModel,
    append_jsonl_row,
    default_result_dir,
    default_state_dir,
    init_jsonl_file,
    load_validation_state_dataset,
    squad_em,
)
from state_utils import clone_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RWKV with linear-quantized WKV state: per-matrix vs channel-wise (dataset-calibrated min/max per channel)."
    )
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--outlier-frac", type=float, default=0.1, help="Fraction of elements (by |x|) treated as outliers for scaled mode (default: top 10%%).")
    return parser.parse_args()


def _dtype_bits(t: torch.Tensor) -> int:
    if t.dtype in (torch.float16, torch.bfloat16):
        return 16
    if t.dtype == torch.float32:
        return 32
    return 32


def _linear_quant_dequant_flat(x_flat: torch.Tensor, bits: int) -> torch.Tensor:
    """Asymmetric min–max linear quantize on a 1D float32 tensor, return float32 reconstruction."""
    q_levels = (1 << bits) - 1
    if q_levels <= 0:
        return x_flat
    x_min = x_flat.min()
    x_max = x_flat.max()
    span = (x_max - x_min).clamp_min(1e-12)
    scale = span / float(q_levels)
    q = torch.round((x_flat - x_min) / scale).clamp(0, float(q_levels))
    return q * scale + x_min


def _flat_outlier_work(
    flat: torch.Tensor, outlier_frac: Optional[float]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return work flat tensor; optionally downscale top-|x| outliers (same logic as linear_quant_dequant_matrix)."""
    n = flat.numel()
    if n == 0:
        return flat, None, None
    if outlier_frac is None or outlier_frac <= 0.0 or n <= 1:
        return flat, None, None
    k = max(1, int(n * outlier_frac))
    k = min(k, n - 1)
    abs_flat = flat.abs()
    _, top_idx = torch.topk(abs_flat, k, largest=True)
    mask_flat = torch.zeros(n, dtype=torch.bool, device=flat.device)
    mask_flat[top_idx] = True
    inlier_idx = ~mask_flat
    inlier_vals = flat[inlier_idx]
    outlier_vals = flat[mask_flat]
    t = inlier_vals.abs().max()
    o = outlier_vals.abs().max()
    eps = 1e-12
    if t > eps and o > t:
        scale_s = o / t
        work = flat.clone()
        work[mask_flat] = flat[mask_flat] / scale_s
        return work, mask_flat, scale_s
    return flat, None, None


def linear_quant_dequant_matrix(
    matrix: torch.Tensor, bits: int, outlier_frac: Optional[float]
) -> Tuple[torch.Tensor, float]:
    """
    Per-matrix linear quantize / dequantize. Optionally apply AWQ-style scaling on top-|x| outliers
    before quant, then multiply back after dequant.

    Returns (reconstructed matrix, compression_ratio) where compression_ratio = bits / original_bits.
    """
    orig_dtype = matrix.dtype
    orig_bits = _dtype_bits(matrix)
    x = matrix.float()
    flat = x.flatten()
    n = flat.numel()
    if n == 0:
        return matrix.clone(), float(bits) / float(orig_bits)

    work, mask_flat, scale_s = _flat_outlier_work(flat, outlier_frac)

    recon_flat = _linear_quant_dequant_flat(work, bits)
    if mask_flat is not None and scale_s is not None:
        recon_flat = recon_flat.clone()
        recon_flat[mask_flat] = recon_flat[mask_flat] * scale_s

    recon = recon_flat.view_as(x).to(dtype=orig_dtype)
    compression = float(bits) / float(orig_bits)
    return recon, compression


def _linear_quant_dequant_channelwise_2d(x: torch.Tensor, bits: int, ch_min: torch.Tensor, ch_max: torch.Tensor) -> torch.Tensor:
    """Asymmetric linear quant per row (channel). ch_min/ch_max: shape [H] for x [H, W]."""
    q_levels = (1 << bits) - 1
    if q_levels <= 0:
        return x
    span = (ch_max - ch_min).clamp_min(1e-12)
    scale = span / float(q_levels)
    q = torch.round((x - ch_min.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, float(q_levels))
    return q * scale.unsqueeze(-1) + ch_min.unsqueeze(-1)


def linear_quant_dequant_matrix_channelwise(
    matrix: torch.Tensor,
    bits: int,
    outlier_frac: Optional[float],
    ch_min: torch.Tensor,
    ch_max: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Per-channel (row-wise) quant using fixed ch_min/ch_max from calibration; optional outlier pre-scaling."""
    orig_dtype = matrix.dtype
    orig_bits = _dtype_bits(matrix)
    x = matrix.float()
    flat = x.flatten()
    n = flat.numel()
    if n == 0:
        return matrix.clone(), float(bits) / float(orig_bits)

    work_flat, mask_flat, scale_s = _flat_outlier_work(flat, outlier_frac)
    work_2d = work_flat.view_as(x)
    recon_2d = _linear_quant_dequant_channelwise_2d(work_2d, bits, ch_min, ch_max)
    recon_flat = recon_2d.flatten()
    if mask_flat is not None and scale_s is not None:
        recon_flat = recon_flat.clone()
        recon_flat[mask_flat] = recon_flat[mask_flat] * scale_s

    recon = recon_flat.view_as(x).to(dtype=orig_dtype)
    compression = float(bits) / float(orig_bits)
    return recon, compression


def build_wkv_channel_calibration(
    dataset,
    outlier_frac: Optional[float],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Per layer: (ch_min, ch_max) with shape [num_heads, H] where H is channel dim (dim 0 of each head matrix).
    Min/max are taken over all samples and all positions along the last dim for each channel.
    """
    state0 = dataset[0]["state"]
    if not isinstance(state0, list) or len(state0) % 3 != 0:
        raise RuntimeError("Unexpected state format: expected list length n_layer*3.")

    layer_mins: List[torch.Tensor] = []
    layer_maxs: List[torch.Tensor] = []
    for i in range(0, len(state0), 3):
        wkv0 = state0[i + 1]
        if not torch.is_tensor(wkv0) or wkv0.ndim != 3:
            layer_mins.append(torch.empty(0))
            layer_maxs.append(torch.empty(0))
            continue
        nh, h_ch, _ = wkv0.shape
        dev = wkv0.device
        dt = torch.float32
        layer_mins.append(torch.full((nh, h_ch), float("inf"), device=dev, dtype=dt))
        layer_maxs.append(torch.full((nh, h_ch), float("-inf"), device=dev, dtype=dt))

    for idx in range(len(dataset)):
        state = dataset[idx]["state"]
        layer_idx = 0
        for i in range(0, len(state), 3):
            wkv = state[i + 1]
            if not torch.is_tensor(wkv) or wkv.ndim != 3:
                layer_idx += 1
                continue
            if layer_mins[layer_idx].numel() == 0:
                layer_idx += 1
                continue
            x = wkv.float()
            num_heads = x.shape[0]
            for h in range(num_heads):
                flat = x[h].flatten()
                work_flat, _, _ = _flat_outlier_work(flat, outlier_frac)
                m2 = work_flat.view_as(x[h])
                row_min = m2.min(dim=-1).values
                row_max = m2.max(dim=-1).values
                layer_mins[layer_idx][h] = torch.minimum(layer_mins[layer_idx][h], row_min)
                layer_maxs[layer_idx][h] = torch.maximum(layer_maxs[layer_idx][h], row_max)
            layer_idx += 1

    for li in range(len(layer_mins)):
        if layer_mins[li].numel() == 0:
            continue
        layer_mins[li] = torch.where(
            torch.isfinite(layer_mins[li]), layer_mins[li], torch.zeros_like(layer_mins[li])
        )
        layer_maxs[li] = torch.where(
            torch.isfinite(layer_maxs[li]), layer_maxs[li], torch.zeros_like(layer_maxs[li])
        )
        same = layer_mins[li] >= layer_maxs[li]
        if same.any():
            layer_maxs[li] = torch.where(same, layer_mins[li] + 1e-12, layer_maxs[li])

    return list(zip(layer_mins, layer_maxs))


def apply_linear_quant_to_wkv_state(
    state,
    bits: int,
    outlier_frac: Optional[float],
):
    if not isinstance(state, list) or len(state) % 3 != 0:
        raise RuntimeError("Unexpected state format: expected list length n_layer*3.")

    out = clone_state(state)
    ratio_sum = 0.0
    ratio_count = 0
    for i in range(0, len(out), 3):
        wkv = out[i + 1]
        if not torch.is_tensor(wkv) or wkv.ndim != 3:
            continue
        num_heads = wkv.shape[0]
        recon_heads = []
        for h in range(num_heads):
            recon, compression = linear_quant_dequant_matrix(wkv[h], bits, outlier_frac)
            recon_heads.append(recon)
            ratio_sum += compression
            ratio_count += 1
        out[i + 1] = torch.stack(recon_heads, dim=0)
    return out, ratio_sum, ratio_count


def apply_channelwise_quant_to_wkv_state(
    state,
    bits: int,
    outlier_frac: Optional[float],
    calibration: List[Tuple[torch.Tensor, torch.Tensor]],
):
    if not isinstance(state, list) or len(state) % 3 != 0:
        raise RuntimeError("Unexpected state format: expected list length n_layer*3.")

    out = clone_state(state)
    ratio_sum = 0.0
    ratio_count = 0
    layer_idx = 0
    for i in range(0, len(out), 3):
        wkv = out[i + 1]
        if not torch.is_tensor(wkv) or wkv.ndim != 3:
            layer_idx += 1
            continue
        if layer_idx >= len(calibration):
            layer_idx += 1
            continue
        ch_min_all, ch_max_all = calibration[layer_idx]
        if ch_min_all.numel() == 0:
            layer_idx += 1
            continue
        num_heads = wkv.shape[0]
        recon_heads = []
        for h in range(num_heads):
            ch_min = ch_min_all[h].to(device=wkv.device, dtype=torch.float32)
            ch_max = ch_max_all[h].to(device=wkv.device, dtype=torch.float32)
            recon, compression = linear_quant_dequant_matrix_channelwise(
                wkv[h], bits, outlier_frac, ch_min, ch_max
            )
            recon_heads.append(recon)
            ratio_sum += compression
            ratio_count += 1
        out[i + 1] = torch.stack(recon_heads, dim=0)
        layer_idx += 1
    return out, ratio_sum, ratio_count


def _sanitize_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", name)


def init_mode_record_file(output_dir: str, mode_key: str) -> str:
    path = os.path.join(output_dir, f"{_sanitize_key(mode_key)}_qa_records.jsonl")
    init_jsonl_file(path)
    return path


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 11,
            "ytick.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.4,
        }
    )


def plot_quant_results(summary: OrderedDict, output_path: str):
    labels = list(summary.keys())
    accs = [summary[k]["accuracy"] for k in labels]
    comps = [summary[k]["mean_compression_ratio"] for k in labels]
    x = list(range(len(labels)))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    plt.figure()
    bars = plt.bar(
        x,
        accs,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    for bar, v, c in zip(bars, accs, comps):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}\n({c * 100:.1f}% storage)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylim(0, min(1.0, max(accs + [0.1]) + 0.08))
    plt.ylabel("Accuracy")
    plt.xlabel("Quantization setting")
    plt.title("RWKV WKV-State Quantization (per-matrix vs channel-wise calib.)", pad=12)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_mode(
    mode_key: str,
    rwkv: RWKVModel,
    dataset,
    bits: int,
    outlier_frac: Optional[float],
    max_new_tokens: int,
    record_path: str,
    channel_calibration: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    correct = 0
    ratio_sum = 0.0
    ratio_count = 0
    desc = f"Quant {mode_key}"
    progress = tqdm(range(len(dataset)), desc=desc)
    for idx in progress:
        row = dataset[idx]
        if channel_calibration is not None:
            q_state, rs, rc = apply_channelwise_quant_to_wkv_state(
                row["state"], bits, outlier_frac, channel_calibration
            )
        else:
            q_state, rs, rc = apply_linear_quant_to_wkv_state(row["state"], bits, outlier_frac)
        ratio_sum += rs
        ratio_count += rc
        sample_mean_compression = (rs / rc) if rc > 0 else 0.0
        pred = rwkv.generate(question=row["question"], init_state=q_state, max_new_tokens=max_new_tokens)
        answers = row.get("answers", [])
        em = squad_em(pred, answers)
        correct += em
        append_jsonl_row(
            record_path,
            {
                "index": idx,
                "question": row["question"],
                "prediction": pred,
                "answers": answers,
                "em": int(em),
                "bits": int(bits),
                "outlier_frac": outlier_frac,
                "mean_compression_ratio": sample_mean_compression,
                "channelwise": channel_calibration is not None,
            },
        )
        running_acc = (correct / (idx + 1)) * 100.0
        progress.set_postfix_str(f"acc={running_acc:.2f}%")

    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
    mean_compression = (ratio_sum / ratio_count) if ratio_count > 0 else 0.0
    return {"accuracy": accuracy, "mean_compression_ratio": mean_compression}


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "wkv_quant")
    os.makedirs(output_dir, exist_ok=True)
    apply_plot_style()

    max_new_tokens = 256
    state_dir = args.state_dir or default_state_dir(args.model_filename)
    rwkv = RWKVModel(
        model_path=args.model_path,
        model_filename=args.model_filename,
        strategy=args.strategy,
        tokenizer=args.tokenizer,
    )
    dataset = load_validation_state_dataset(state_dir=state_dir, limit=args.num_samples)

    results: OrderedDict[str, dict] = OrderedDict()
    of = float(args.outlier_frac)

    calib_plain = build_wkv_channel_calibration(dataset, outlier_frac=None)

    for bits in args.bits:
        bits = int(bits)
        # 1) naive per-matrix quantization
        mode_key = f"{bits}bit_plain"
        record_path = init_mode_record_file(output_dir, mode_key)
        result = evaluate_mode(
            mode_key=mode_key,
            rwkv=rwkv,
            dataset=dataset,
            bits=bits,
            outlier_frac=None,
            max_new_tokens=max_new_tokens,
            record_path=record_path,
        )
        results[mode_key] = {
            "bits": bits,
            "outlier_scaled": False,
            "outlier_frac": None,
            "channelwise": False,
            "accuracy": result["accuracy"],
            "mean_compression_ratio": result["mean_compression_ratio"],
        }

        # 2) scaling-based per-matrix quantization
        mode_key = f"{bits}bit_outlier_scale"
        record_path = init_mode_record_file(output_dir, mode_key)
        result = evaluate_mode(
            mode_key=mode_key,
            rwkv=rwkv,
            dataset=dataset,
            bits=bits,
            outlier_frac=of,
            max_new_tokens=max_new_tokens,
            record_path=record_path,
        )
        results[mode_key] = {
            "bits": bits,
            "outlier_scaled": True,
            "outlier_frac": of,
            "channelwise": False,
            "accuracy": result["accuracy"],
            "mean_compression_ratio": result["mean_compression_ratio"],
        }

        # 3) channel-wise quantization with dataset-level calibration
        mode_key = f"{bits}bit_channelwise"
        record_path = init_mode_record_file(output_dir, mode_key)
        result = evaluate_mode(
            mode_key=mode_key,
            rwkv=rwkv,
            dataset=dataset,
            bits=bits,
            outlier_frac=None,
            max_new_tokens=max_new_tokens,
            record_path=record_path,
            channel_calibration=calib_plain,
        )
        results[mode_key] = {
            "bits": bits,
            "outlier_scaled": False,
            "outlier_frac": None,
            "channelwise": True,
            "accuracy": result["accuracy"],
            "mean_compression_ratio": result["mean_compression_ratio"],
        }

    result_json = os.path.join(output_dir, "squad_rwkv_wkv_quant_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "wkv_quant_bar.png")
    plot_quant_results(results, plot_path)

    print(f"Saved summary: {result_json}")
    print(f"Saved plot: {plot_path}")
    for name, row in results.items():
        cw = row.get("channelwise", False)
        print(
            f"{name}: acc={row['accuracy']:.4f}, "
            f"mean_compression={row['mean_compression_ratio']:.4f}, "
            f"bits={row['bits']}, outlier_scaled={row['outlier_scaled']}, channelwise={cw}"
        )


if __name__ == "__main__":
    main()
