import argparse
import json
import os
from collections import OrderedDict

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
from svd_utils import rank_from_threshold, truncated_svd_reconstruct_by_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RWKV with ratio-threshold SVD-compressed WKV state.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--sv-thresholds", type=float, nargs="+", default=[1.00, 0.95, 0.90, 0.85, 0.80, 0.75])
    return parser.parse_args()


def apply_svd_ratio_to_wkv_state(state, threshold: float):
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
            recon, compression = truncated_svd_reconstruct_by_threshold(wkv[h], threshold)
            recon_heads.append(recon)
            ratio_sum += compression
            ratio_count += 1
        out[i + 1] = torch.stack(recon_heads, dim=0)
    return out, ratio_sum, ratio_count


def init_threshold_record_file(output_dir: str, threshold: float) -> str:
    name = f"{threshold:.2f}".replace(".", "_")
    path = os.path.join(output_dir, f"svd_threshold_{name}_qa_records.jsonl")
    init_jsonl_file(path)
    return path


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        }
    )


def plot_ratio_results(summary: OrderedDict, output_path: str):
    keys = list(summary.keys())
    accs = [summary[k]["accuracy"] for k in keys]
    mean_compressions = [summary[k]["mean_compression_ratio"] for k in keys]
    x = list(range(len(keys)))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, accs, color="#4C72B0", edgecolor="white", linewidth=0.8, zorder=5)
    for bar, v, comp in zip(bars, accs, mean_compressions):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}\n({comp * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.ylim(0, min(1.0, max(accs + [0.1]) + 0.08))
    plt.ylabel("Accuracy")
    plt.xlabel("SV Threshold x")
    plt.title("RWKV WKV-State Ratio-threshold SVD Compression", pad=12)
    plt.xticks(x, keys)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_with_threshold(
    rwkv: RWKVModel,
    dataset,
    threshold: float,
    max_new_tokens: int,
    record_path: str,
):
    correct = 0
    ratio_sum = 0.0
    ratio_count = 0
    progress = tqdm(range(len(dataset)), desc=f"Ratio SVD Eval (x={threshold:.2f})")
    for idx in progress:
        row = dataset[idx]
        svd_state, rs, rc = apply_svd_ratio_to_wkv_state(row["state"], threshold)
        ratio_sum += rs
        ratio_count += rc
        sample_mean_compression = (rs / rc) if rc > 0 else 0.0
        pred = rwkv.generate(question=row["question"], init_state=svd_state, max_new_tokens=max_new_tokens)
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
                "sv_threshold": float(threshold),
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
    output_dir = default_result_dir(args.model_filename, "ratio")
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

    results = OrderedDict()
    for threshold in args.sv_thresholds:
        threshold = float(threshold)
        record_path = init_threshold_record_file(output_dir, threshold)
        result = evaluate_with_threshold(
            rwkv=rwkv,
            dataset=dataset,
            threshold=threshold,
            max_new_tokens=max_new_tokens,
            record_path=record_path,
        )
        results[f"{threshold:.2f}"] = {
            "accuracy": result["accuracy"],
            "mean_compression_ratio": result["mean_compression_ratio"],
        }

    result_json = os.path.join(output_dir, "squad_rwkv_wkv_svd_ratio_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "wkv_svd_ratio_bar.png")
    plot_ratio_results(results, plot_path)

    print(f"Saved summary: {result_json}")
    print(f"Saved plot: {plot_path}")
    for threshold, row in results.items():
        print(
            f"SV threshold {threshold}: "
            f"acc={row['accuracy']:.4f}, mean_compression={row['mean_compression_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
