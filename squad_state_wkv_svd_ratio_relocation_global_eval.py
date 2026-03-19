import argparse
import copy
import json
import os
from collections import OrderedDict
from typing import Dict, Tuple

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
        description="Evaluate RWKV with global relocation + ratio-threshold SVD-compressed WKV state."
    )
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-7.2b-20260301-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--sv-thresholds", type=float, nargs="+", default=[1.00, 0.95, 0.90, 0.85, 0.80, 0.75])
    return parser.parse_args()


def clone_state(state):
    if isinstance(state, list):
        return [clone_state(x) for x in state]
    if isinstance(state, tuple):
        return tuple(clone_state(x) for x in state)
    if torch.is_tensor(state):
        return state.clone()
    return copy.deepcopy(state)


def rank_from_threshold(singular_values: torch.Tensor, threshold: float) -> int:
    total = singular_values.sum()
    if total <= 0:
        return 1
    cumsum = torch.cumsum(singular_values, dim=0)
    target = float(threshold) * float(total.item())
    k = int(torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device), right=False).item()) + 1
    return max(1, min(k, singular_values.shape[0]))


def truncated_svd_reconstruct_by_threshold(matrix: torch.Tensor, threshold: float):
    orig_dtype = matrix.dtype
    work = matrix.float()
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    k = rank_from_threshold(s, threshold)
    u_k = u[:, :k]
    s_k = s[:k]
    vh_k = vh[:k, :]
    recon = (u_k * s_k.unsqueeze(0)) @ vh_k
    d = int(s.shape[0])
    compression = (2.0 * float(k)) / float(d)
    return recon.to(dtype=orig_dtype), compression


def key_to_str(key: Tuple[int, int]) -> str:
    return f"slot{key[0]}_head{key[1]}"


def build_global_relocation_orders(dataset) -> Dict[Tuple[int, int], torch.Tensor]:
    sum_map: Dict[Tuple[int, int], torch.Tensor] = {}
    cnt_map: Dict[Tuple[int, int], int] = {}
    for row in tqdm(dataset, desc="Build global relocation index"):
        state = row["state"]
        if not isinstance(state, list) or len(state) % 3 != 0:
            raise RuntimeError("Unexpected state format: expected list length n_layer*3.")
        for i in range(0, len(state), 3):
            wkv = state[i + 1]
            if not torch.is_tensor(wkv) or wkv.ndim != 3:
                continue
            for h in range(wkv.shape[0]):
                key = (i + 1, h)
                matrix = wkv[h].to(torch.float64).cpu()
                if key not in sum_map:
                    sum_map[key] = torch.zeros_like(matrix, dtype=torch.float64)
                    cnt_map[key] = 0
                sum_map[key] += matrix
                cnt_map[key] += 1

    if not sum_map:
        raise RuntimeError("No WKV tensors found to build global relocation index.")

    order_map: Dict[Tuple[int, int], torch.Tensor] = {}
    for key, mat_sum in sum_map.items():
        mean_matrix = mat_sum / max(cnt_map[key], 1)
        flat_mean = mean_matrix.reshape(-1)
        order = torch.argsort(flat_mean, descending=True)
        order_map[key] = order.to(torch.long).cpu()
    return order_map


def relocate_matrix_with_order(matrix: torch.Tensor, order: torch.Tensor):
    flat = matrix.reshape(-1)
    order_dev = order.to(device=flat.device)
    relocated_flat = flat.index_select(0, order_dev)
    return relocated_flat.reshape_as(matrix)


def restore_matrix_from_relocation(relocated_matrix: torch.Tensor, order: torch.Tensor):
    relocated_flat = relocated_matrix.reshape(-1)
    restored_flat = torch.empty_like(relocated_flat)
    restored_flat.scatter_(0, order.to(device=relocated_flat.device), relocated_flat)
    return restored_flat.reshape_as(relocated_matrix)


def apply_global_relocation_svd_ratio_to_wkv_state(state, threshold: float, order_map):
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
            key = (i + 1, h)
            if key not in order_map:
                raise KeyError(f"Missing global relocation index for {key_to_str(key)}")
            order = order_map[key]
            if int(order.numel()) != int(wkv[h].numel()):
                raise RuntimeError(
                    f"Relocation index size mismatch at {key_to_str(key)}: "
                    f"order={int(order.numel())}, matrix={int(wkv[h].numel())}"
                )
            relocated = relocate_matrix_with_order(wkv[h], order)
            recon_relocated, compression = truncated_svd_reconstruct_by_threshold(relocated, threshold)
            recon = restore_matrix_from_relocation(recon_relocated, order)
            recon_heads.append(recon)
            ratio_sum += compression
            ratio_count += 1
        out[i + 1] = torch.stack(recon_heads, dim=0)
    return out, ratio_sum, ratio_count


def init_threshold_record_file(output_dir: str, threshold: float) -> str:
    name = f"{threshold:.2f}".replace(".", "_")
    path = os.path.join(output_dir, f"global_relocation_svd_threshold_{name}_qa_records.jsonl")
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


def plot_ratio_global_relocation_results(summary: OrderedDict, output_path: str):
    keys = list(summary.keys())
    accs = [summary[k]["accuracy"] for k in keys]
    mean_compressions = [summary[k]["mean_compression_ratio"] for k in keys]
    x = list(range(len(keys)))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, accs, color="#C44E52", edgecolor="white", linewidth=0.8, zorder=5)
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
    plt.title("Global Relocation + Ratio-threshold SVD Compression", pad=12)
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
    order_map,
):
    correct = 0
    ratio_sum = 0.0
    ratio_count = 0
    progress = tqdm(range(len(dataset)), desc=f"Global Relocation+SVD Eval (x={threshold:.2f})")
    for idx in progress:
        row = dataset[idx]
        svd_state, rs, rc = apply_global_relocation_svd_ratio_to_wkv_state(
            row["state"], threshold, order_map
        )
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
                "mode": "global_relocation_svd",
            },
        )
        running_acc = (correct / (idx + 1)) * 100.0
        progress.set_postfix_str(f"acc={running_acc:.2f}%")

    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
    mean_compression = (ratio_sum / ratio_count) if ratio_count > 0 else 0.0
    return {"accuracy": accuracy, "mean_compression_ratio": mean_compression}


def save_order_metadata(output_dir: str, order_map, dataset_size: int):
    meta_rows = []
    for key, order in order_map.items():
        meta_rows.append(
            {
                "slot_index": int(key[0]),
                "head_index": int(key[1]),
                "num_elements": int(order.numel()),
            }
        )
    meta_rows = sorted(meta_rows, key=lambda x: (x["slot_index"], x["head_index"]))
    with open(os.path.join(output_dir, "global_relocation_index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"dataset_size": int(dataset_size), "index_count": len(meta_rows), "entries": meta_rows},
            f,
            ensure_ascii=False,
            indent=2,
        )


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

    order_map = build_global_relocation_orders(dataset)
    torch.save(
        {key_to_str(k): v for k, v in order_map.items()},
        os.path.join(output_dir, "global_relocation_index.pt"),
    )
    save_order_metadata(output_dir, order_map, len(dataset))

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
            order_map=order_map,
        )
        results[f"{threshold:.2f}"] = {
            "accuracy": result["accuracy"],
            "mean_compression_ratio": result["mean_compression_ratio"],
        }

    result_json = os.path.join(output_dir, "squad_rwkv_wkv_svd_ratio_global_relocation_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "wkv_svd_ratio_global_relocation_bar.png")
    plot_ratio_global_relocation_results(results, plot_path)

    print(f"Saved summary: {result_json}")
    print(f"Saved index: {os.path.join(output_dir, 'global_relocation_index.pt')}")
    print(f"Saved plot: {plot_path}")
    for threshold, row in results.items():
        print(
            f"Global Relocation SV threshold {threshold}: "
            f"acc={row['accuracy']:.4f}, mean_compression={row['mean_compression_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
