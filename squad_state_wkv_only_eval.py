import argparse
import copy
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RWKV with only Matrix-Valued WKV state kept.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    return parser.parse_args()


def keep_only_wkv_matrix_state(state):
    # RWKV7 state layout per layer:
    # idx*3+0: time-mixing shift vector
    # idx*3+1: matrix-valued WKV state
    # idx*3+2: channel-mixing shift vector
    if not isinstance(state, list) or len(state) % 3 != 0:
        raise RuntimeError("Unexpected state format: expected list length n_layer*3.")
    out = []
    for i in range(0, len(state), 3):
        s0 = state[i + 0]
        s1 = state[i + 1]
        s2 = state[i + 2]
        out.append(torch.zeros_like(s0) if torch.is_tensor(s0) else copy.deepcopy(s0))
        out.append(s1.clone() if torch.is_tensor(s1) else copy.deepcopy(s1))
        out.append(torch.zeros_like(s2) if torch.is_tensor(s2) else copy.deepcopy(s2))
    return out


def init_mode_record_file(output_dir: str, mode_name: str) -> str:
    path = os.path.join(output_dir, f"{mode_name}_qa_records.jsonl")
    init_jsonl_file(path)
    return path


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.4,
        }
    )


def plot_wkv_only_results(results: OrderedDict, output_path: str):
    labels = list(results.keys())
    values = [float(results[k]) for k in labels]
    x = list(range(len(labels)))
    colors = ["#4C72B0", "#55A868"]
    plt.figure()
    bars = plt.bar(
        x,
        values,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.006,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.ylim(0, min(1.0, max(values + [0.1]) + 0.08))
    plt.ylabel("Accuracy")
    plt.xlabel("Evaluation Mode")
    plt.title("RWKV WKV-Only State Evaluation", pad=12)
    plt.xticks(x, labels)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_mode(
    mode_name: str,
    rwkv: RWKVModel,
    dataset,
    state_transform,
    max_new_tokens: int,
    record_path: str,
):
    correct = 0
    progress = tqdm(range(len(dataset)), desc=f"{mode_name} Eval")
    for idx in progress:
        row = dataset[idx]
        state = state_transform(row["state"])
        pred = rwkv.generate(question=row["question"], init_state=state, max_new_tokens=max_new_tokens)
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
                "mode": mode_name,
            },
        )
        progress.set_postfix_str(f"acc={(correct / (idx + 1)) * 100.0:.2f}%")
    return {"accuracy": (correct / len(dataset)) if len(dataset) > 0 else 0.0}


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "wkv_only")
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

    no_change_path = init_mode_record_file(output_dir, "no_change")
    no_change_result = evaluate_mode(
        mode_name="No Change",
        rwkv=rwkv,
        dataset=dataset,
        state_transform=lambda s: s,
        max_new_tokens=max_new_tokens,
        record_path=no_change_path,
    )
    results["No Change"] = no_change_result["accuracy"]

    wkv_only_path = init_mode_record_file(output_dir, "wkv_only")
    wkv_only_result = evaluate_mode(
        mode_name="WKV Only (slot0/2 zero)",
        rwkv=rwkv,
        dataset=dataset,
        state_transform=keep_only_wkv_matrix_state,
        max_new_tokens=max_new_tokens,
        record_path=wkv_only_path,
    )
    results["WKV Only (slot0/2 zero)"] = wkv_only_result["accuracy"]

    result_path = os.path.join(output_dir, "squad_rwkv_wkv_only_eval_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "wkv_only_vs_no_change_bar.png")
    plot_wkv_only_results(results, plot_path)

    print("\n=== RWKV State WKV-Only Evaluation ===")
    print(f"{'Mode':<32} {'Accuracy':>10}")
    print("-" * 45)
    for k, v in results.items():
        print(f"{k:<32} {v:>10.4f}")
    print("-" * 45)
    print(f"Saved summary: {result_path}")
    print(f"Saved records: {no_change_path}, {wkv_only_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
