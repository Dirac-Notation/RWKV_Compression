import argparse
import copy
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Sequence

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
    parser = argparse.ArgumentParser(description="Evaluate RWKV on SQuAD with state merge.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    return parser.parse_args()


def merge_states(states: Sequence):
    merged = copy.deepcopy(states[0])
    for state in states[1:]:
        merged = _add_state(merged, state)
    return _scale_state(merged, 1.0 / len(states))


def _add_state(lhs, rhs):
    if torch.is_tensor(lhs):
        return lhs + rhs
    if isinstance(lhs, tuple):
        return tuple(_add_state(lv, rv) for lv, rv in zip(lhs, rhs))
    if isinstance(lhs, list):
        return [_add_state(lv, rv) for lv, rv in zip(lhs, rhs)]
    raise TypeError(f"Unsupported state type: {type(lhs)}")


def _scale_state(state, factor: float):
    if torch.is_tensor(state):
        return state * factor
    if isinstance(state, tuple):
        return tuple(_scale_state(v, factor) for v in state)
    if isinstance(state, list):
        return [_scale_state(v, factor) for v in state]
    raise TypeError(f"Unsupported state type: {type(state)}")


def init_mode_record_file(output_dir: str, mode_name: str) -> str:
    path = os.path.join(output_dir, f"{mode_name}_qa_records.jsonl")
    init_jsonl_file(path)
    return path


def append_mode_record(path: str, row: Dict[str, object]) -> None:
    append_jsonl_row(path, row)


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        }
    )


def load_records(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_group_size(records: List[Dict], file_path: str) -> int:
    if not records:
        return 0
    first = records[0]
    if "group_size" in first:
        n = int(first["group_size"])
        if n > 0:
            return n
    name = os.path.basename(file_path)
    if name.startswith("no_merge_"):
        return 1
    match = re.search(r"merge_(\d+)_", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer group_size from: {file_path}")


def compute_distribution(records: List[Dict], group_size: int) -> List[float]:
    records = sorted(records, key=lambda x: int(x.get("index", 0)))
    usable = (len(records) // group_size) * group_size
    group_count = usable // group_size
    correct_count_hist = [0] * (group_size + 1)
    for start in range(0, usable, group_size):
        block = records[start : start + group_size]
        correct_in_group = sum(int(row.get("em", 0)) for row in block)
        correct_count_hist[correct_in_group] += 1
    if group_count == 0:
        return [0.0] * (group_size + 1)
    return [count / group_count for count in correct_count_hist]


def compute_accuracy_breakdown(all_ratios: List[float], group_size: int) -> Dict[str, object]:
    contrib_by_wrong = [0.0] * group_size
    for k in range(1, min(len(all_ratios), group_size + 1)):
        w = group_size - k
        contrib_by_wrong[w] += (k / group_size) * all_ratios[k]
    accuracy = sum(contrib_by_wrong)
    return {"contrib_by_wrong": contrib_by_wrong, "accuracy": accuracy}


def mode_sort_key(mode_name: str):
    if mode_name == "No Merge":
        return (0, 0)
    m = re.match(r"Merge (\d+)", mode_name)
    if m:
        return (1, int(m.group(1)))
    return (2, mode_name)


def mode_label_from_file(file_path: str, group_size: int) -> str:
    name = os.path.basename(file_path)
    if name.startswith("no_merge_"):
        return "No Merge"
    return f"Merge {group_size}"


def build_mode_distributions(output_dir: str):
    mode_data = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith("_qa_records.jsonl"):
            continue
        if not (name.startswith("no_merge_") or name.startswith("merge_")):
            continue
        file_path = os.path.join(output_dir, name)
        records = load_records(file_path)
        if not records:
            continue
        group_size = infer_group_size(records, file_path)
        all_ratios = compute_distribution(records, group_size)
        breakdown = compute_accuracy_breakdown(all_ratios, group_size)
        mode_data.append(
            {
                "mode": mode_label_from_file(file_path, group_size),
                "group_size": group_size,
                "ratios": breakdown["contrib_by_wrong"],
                "accuracy": breakdown["accuracy"],
            }
        )
    mode_data.sort(key=lambda x: mode_sort_key(x["mode"]))
    return mode_data


def plot_stacked_distribution(mode_data: List[Dict], output_path: str):
    if not mode_data:
        raise RuntimeError("No mode data to plot.")
    max_group_size = max(int(x["group_size"]) for x in mode_data)
    x_positions = [int(entry["group_size"]) for entry in mode_data]
    x_tick_labels = [str(v) for v in x_positions]
    colors = ["#4C72B0", "#55A868", "#64B5CD", "#8172B3", "#CCB974", "#76B7B2", "#9C755F"]

    plt.figure()
    bottoms = [0.0] * len(mode_data)
    for wrong_count in range(max_group_size - 1, -1, -1):
        heights = []
        for entry in mode_data:
            ratios = entry["ratios"]
            heights.append(ratios[wrong_count] if wrong_count < len(ratios) else 0.0)
        plt.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=colors[wrong_count % len(colors)],
            edgecolor="white",
            linewidth=0.7,
            label=f"{wrong_count} wrong",
            zorder=5,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    plt.ylabel("Ratio")
    plt.xlabel("Number of Merged Group")
    plt.ylim(0, 1)
    plt.xticks(x_positions, x_tick_labels)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    for i, entry in enumerate(mode_data):
        text_y = min(0.98, entry["accuracy"] + 0.02)
        plt.text(
            x_positions[i],
            text_y,
            f"acc={entry['accuracy']:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            fontweight="bold",
            zorder=10,
        )
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    order = sorted(range(len(legend_labels)), key=lambda i: int(legend_labels[i].split()[0]))
    handles = [handles[i] for i in order]
    legend_labels = [legend_labels[i] for i in order]
    plt.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_no_merge(
    rwkv: RWKVModel,
    dataset,
    max_new_tokens,
    record_path: str,
):
    correct = 0
    progress = tqdm(range(len(dataset)), desc="No Merge Eval")
    for idx in progress:
        row = dataset[idx]
        pred = rwkv.generate(question=row["question"], init_state=row["state"], max_new_tokens=max_new_tokens)
        answers = row.get("answers", [])
        em = squad_em(pred, answers)
        correct += em
        append_mode_record(
            record_path,
            {
                "index": idx,
                "question": row["question"],
                "prediction": pred,
                "answers": answers,
                "em": int(em),
            },
        )
        running_acc = (correct / (idx + 1)) * 100.0
        progress.set_postfix_str(f"acc={running_acc:.2f}%")
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
    return {"accuracy": accuracy}


def evaluate_with_state_merge(
    rwkv: RWKVModel,
    dataset,
    group_size,
    max_new_tokens,
    record_path: str,
):
    correct = 0
    total = 0
    progress = tqdm(range(0, len(dataset), group_size), desc=f"Merged Eval (N={group_size})")
    for start in progress:
        if start + group_size > len(dataset):
            break
        group_rows = [dataset[i] for i in range(start, start + group_size)]
        group_states = [row["state"] for row in group_rows]
        merged_state = merge_states(group_states)
        for idx_in_group, row in enumerate(group_rows):
            pred = rwkv.generate(question=row["question"], init_state=merged_state, max_new_tokens=max_new_tokens)
            answers = row.get("answers", [])
            em = squad_em(pred, answers)
            correct += em
            total += 1
            append_mode_record(
                record_path,
                {
                    "index": start + idx_in_group,
                    "question": row["question"],
                    "prediction": pred,
                    "answers": answers,
                    "em": int(em),
                    "group_size": group_size,
                },
            )
            running_acc = (correct / total) * 100.0
            progress.set_postfix_str(f"acc={running_acc:.2f}%")
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy}


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "merge")
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

    acc = OrderedDict()
    no_merge_path = init_mode_record_file(output_dir, "no_merge")
    no_merge_result = evaluate_no_merge(
        rwkv,
        dataset,
        max_new_tokens,
        no_merge_path,
    )
    acc["No Merge"] = no_merge_result["accuracy"]

    for n in sorted(set(args.group_sizes)):
        if n > 1:
            merge_path = init_mode_record_file(output_dir, f"merge_{n}")
            merge_result = evaluate_with_state_merge(
                rwkv,
                dataset,
                n,
                max_new_tokens,
                merge_path,
            )
            acc[f"Merge {n}"] = merge_result["accuracy"]

    result_json = os.path.join(output_dir, "squad_rwkv_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(acc, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "group_outcome_stacked.png")
    mode_data = build_mode_distributions(output_dir)
    plot_stacked_distribution(mode_data, plot_path)
    for k, v in acc.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved summary: {result_json}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
