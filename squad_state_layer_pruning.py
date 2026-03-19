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
    parser = argparse.ArgumentParser(
        description="Evaluate RWKV by pruning front layers of state usage."
    )
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-7.2b-20260301-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument(
        "--prune-front-layers",
        type=int,
        nargs="+",
        default=[0, 4, 8, 12, 16, 20],
        help="Number of front layers to disable (set state to zero).",
    )
    return parser.parse_args()


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (9, 6),
            "figure.dpi": 150,
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.4,
        }
    )


def infer_num_layers(state) -> int:
    if not isinstance(state, list) or len(state) % 3 != 0:
        raise RuntimeError("Unexpected state format: expected list length n_layer*3.")
    return len(state) // 3


def prune_front_layers_from_state(state, prune_count: int):
    if not isinstance(state, list) or len(state) % 3 != 0:
        raise RuntimeError("Unexpected state format: expected list length n_layer*3.")
    out = copy.deepcopy(state)
    num_layers = len(out) // 3
    k = max(0, min(int(prune_count), num_layers))
    for layer_idx in range(k):
        base = layer_idx * 3
        for slot in range(3):
            leaf = out[base + slot]
            if torch.is_tensor(leaf):
                out[base + slot] = torch.zeros_like(leaf)
            else:
                out[base + slot] = copy.deepcopy(leaf)
    return out


def init_pruning_record_file(output_dir: str, prune_count: int) -> str:
    path = os.path.join(output_dir, f"prune_front_{prune_count:02d}_qa_records.jsonl")
    init_jsonl_file(path)
    return path


def evaluate_with_layer_pruning(
    rwkv: RWKVModel,
    dataset,
    prune_count: int,
    max_new_tokens: int,
    record_path: str,
):
    correct = 0
    progress = tqdm(range(len(dataset)), desc=f"Layer Prune Eval (front={prune_count})")
    for idx in progress:
        row = dataset[idx]
        pruned_state = prune_front_layers_from_state(row["state"], prune_count)
        pred = rwkv.generate(question=row["question"], init_state=pruned_state, max_new_tokens=max_new_tokens)
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
                "pruned_front_layers": int(prune_count),
            },
        )
        running_acc = (correct / (idx + 1)) * 100.0
        progress.set_postfix_str(f"acc={running_acc:.2f}%")
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
    return {"accuracy": accuracy}


def plot_layer_pruning_results(summary: OrderedDict, output_path: str):
    labels = list(summary.keys())
    accs = [summary[k]["accuracy"] for k in labels]
    x = list(range(len(labels)))

    plt.figure()
    bars = plt.bar(
        x,
        accs,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    for bar, v in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylim(0, min(1.0, max(accs + [0.1]) + 0.08))
    plt.ylabel("Accuracy")
    plt.xlabel("Front-Pruned Layers")
    plt.title("RWKV State Layer Pruning Evaluation", pad=12)
    plt.xticks(x, labels)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "layer_pruning")
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
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    total_layers = infer_num_layers(dataset[0]["state"])
    prune_candidates = sorted(set(max(0, min(int(k), total_layers)) for k in args.prune_front_layers))

    results = OrderedDict()
    for prune_count in prune_candidates:
        record_path = init_pruning_record_file(output_dir, prune_count)
        result = evaluate_with_layer_pruning(
            rwkv=rwkv,
            dataset=dataset,
            prune_count=prune_count,
            max_new_tokens=max_new_tokens,
            record_path=record_path,
        )
        used_layers = total_layers - prune_count
        label = f"prune{prune_count}_use{used_layers}"
        results[label] = {
            "pruned_front_layers": int(prune_count),
            "used_layers": int(used_layers),
            "accuracy": result["accuracy"],
        }

    result_json = os.path.join(output_dir, "squad_rwkv_layer_pruning_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_path = os.path.join(output_dir, "layer_pruning_accuracy_bar.png")
    plot_layer_pruning_results(results, plot_path)

    print(f"Saved summary: {result_json}")
    print(f"Saved plot: {plot_path}")
    print(f"Total layers: {total_layers}")
    for label, row in results.items():
        print(
            f"{label}: pruned_front={row['pruned_front_layers']}, "
            f"used_layers={row['used_layers']}, acc={row['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
