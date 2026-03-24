import argparse
import copy
import json
import os
import re
from collections import OrderedDict
from typing import Optional, Tuple

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RWKV with linear-quantized WKV state (plain vs outlier scaling)."
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


def clone_state(state):
    if isinstance(state, list):
        return [clone_state(x) for x in state]
    if isinstance(state, tuple):
        return tuple(clone_state(x) for x in state)
    if torch.is_tensor(state):
        return state.clone()
    return copy.deepcopy(state)


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

    mask_flat: Optional[torch.Tensor] = None
    scale_s: Optional[torch.Tensor] = None

    if outlier_frac is not None and outlier_frac > 0.0 and n > 1:
        k = max(1, int(n * outlier_frac))
        k = min(k, n - 1)  # keep at least one inlier for scaling
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
        else:
            work = flat
            mask_flat = None
            scale_s = None
    else:
        work = flat

    recon_flat = _linear_quant_dequant_flat(work, bits)
    if mask_flat is not None and scale_s is not None:
        recon_flat = recon_flat.clone()
        recon_flat[mask_flat] = recon_flat[mask_flat] * scale_s

    recon = recon_flat.view_as(x).to(dtype=orig_dtype)
    compression = float(bits) / float(orig_bits)
    return recon, compression


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
    plt.title("RWKV WKV-State Linear Quantization", pad=12)
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
):
    correct = 0
    ratio_sum = 0.0
    ratio_count = 0
    desc = f"Quant {mode_key}"
    progress = tqdm(range(len(dataset)), desc=desc)
    for idx in progress:
        row = dataset[idx]
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

    for bits in args.bits:
        bits = int(bits)
        for use_outlier in (False, True):
            mode_key = f"{bits}bit_{'outlier_scale' if use_outlier else 'plain'}"
            record_path = init_mode_record_file(output_dir, mode_key)
            frac = of if use_outlier else None
            result = evaluate_mode(
                mode_key=mode_key,
                rwkv=rwkv,
                dataset=dataset,
                bits=bits,
                outlier_frac=frac,
                max_new_tokens=max_new_tokens,
                record_path=record_path,
            )
            results[mode_key] = {
                "bits": bits,
                "outlier_scaled": bool(use_outlier),
                "outlier_frac": of if use_outlier else None,
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
        print(
            f"{name}: acc={row['accuracy']:.4f}, "
            f"mean_compression={row['mean_compression_ratio']:.4f}, "
            f"bits={row['bits']}, outlier_scaled={row['outlier_scaled']}"
        )


if __name__ == "__main__":
    main()
