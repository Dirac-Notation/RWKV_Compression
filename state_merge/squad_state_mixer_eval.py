import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rwkv_model import (
    RWKVModel,
    append_jsonl_row,
    default_result_dir,
    default_state_dir,
    init_jsonl_file,
    load_validation_state_dataset,
    squad_em,
)
from state_merge.mixer import DynamicStateMixer, HeadwiseStateMixer
from state_merge.tiny_rwkv_merger import TinyRWKVMergerConfig, TinyRWKVStateMerger
from state_utils import _is_wkv_path, move_state_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare No-Merge vs Mixer-based 2-state merge on SQuAD."
    )
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--mixer-ckpt", type=str, default="./state_merge/checkpoints/best.pt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def mixer_merge_states(mixer: HeadwiseStateMixer, state_a: Any, state_b: Any):
    sa = move_state_to_device(state_a, "cpu")
    sb = move_state_to_device(state_b, "cpu")
    mixed_wkv = mixer(sa, sb)["mixed"]
    return merge_state_with_avg_non_wkv(sa, sb, mixed_wkv, ())


def merge_state_with_avg_non_wkv(lhs: Any, rhs: Any, mixed_wkv: Any, path: tuple[int, ...]):
    if torch.is_tensor(lhs):
        if _is_wkv_path(path):
            return mixed_wkv
        return (lhs + rhs) * 0.5
    if isinstance(lhs, tuple):
        return tuple(
            merge_state_with_avg_non_wkv(lhs[i], rhs[i], mixed_wkv[i], path + (i,))
            for i in range(len(lhs))
        )
    if isinstance(lhs, list):
        return [
            merge_state_with_avg_non_wkv(lhs[i], rhs[i], mixed_wkv[i], path + (i,))
            for i in range(len(lhs))
        ]
    raise TypeError(f"Unsupported state type: {type(lhs)}")


def init_mode_record_file(output_dir: str, mode_name: str) -> str:
    path = os.path.join(output_dir, f"{mode_name}_qa_records.jsonl")
    init_jsonl_file(path)
    return path


def apply_plot_style():
    rcParams.update(
        {
            "font.family": "serif",
            "figure.figsize": (7, 5),
            "figure.dpi": 150,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.3,
        }
    )


def plot_accuracy_compare(results: OrderedDict, output_path: str):
    labels = [k for k in results.keys() if "Gap" not in k]
    values = [float(results[k]) for k in labels]
    colors = ["#4C72B0", "#55A868"]

    plt.figure()
    bars = plt.bar(range(len(labels)), values, color=colors[: len(labels)], edgecolor="white", linewidth=0.8)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Exact Match (EM)")
    plt.title("No Merge vs Mixer Merge 2")
    plt.xticks(range(len(labels)), labels)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)

    for bar, v in zip(bars, values):
        x = bar.get_x() + bar.get_width() * 0.5
        y = min(0.98, v + 0.02)
        plt.text(x, y, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_records(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_group_size(records, file_path: str) -> int:
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
    if name.startswith("mixer_merge_2"):
        return 2
    match = re.search(r"merge_(\d+)_", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer group_size from: {file_path}")


def compute_distribution(records, group_size: int):
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


def compute_accuracy_breakdown(all_ratios, group_size: int):
    contrib_by_wrong = [0.0] * group_size
    for k in range(1, min(len(all_ratios), group_size + 1)):
        w = group_size - k
        contrib_by_wrong[w] += (k / group_size) * all_ratios[k]
    accuracy = sum(contrib_by_wrong)
    return {"contrib_by_wrong": contrib_by_wrong, "accuracy": accuracy}


def mode_sort_key(mode_name: str):
    if mode_name == "No Merge":
        return (0, 0)
    if mode_name == "Mixer Merge 2":
        return (1, 2)
    m = re.match(r"Merge (\d+)", mode_name)
    if m:
        return (2, int(m.group(1)))
    return (3, mode_name)


def mode_label_from_file(file_path: str, group_size: int):
    name = os.path.basename(file_path)
    if name.startswith("no_merge_"):
        return "No Merge"
    if name.startswith("mixer_merge_2"):
        return "Mixer Merge 2"
    return f"Merge {group_size}"


def build_mode_distributions(output_dir: str):
    mode_data = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith("_qa_records.jsonl"):
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


def plot_stacked_distribution(mode_data, output_path: str):
    if not mode_data:
        raise RuntimeError("No mode data to plot.")
    max_group_size = max(int(x["group_size"]) for x in mode_data)
    x_positions = [i for i in range(len(mode_data))]
    x_tick_labels = [entry["mode"] for entry in mode_data]
    colors = ["#4C72B0", "#55A868", "#64B5CD", "#8172B3", "#CCB974", "#76B7B2", "#9C755F"]

    plt.figure(figsize=(9, 6))
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
    plt.xlabel("Mode")
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
        ncol=min(3, len(legend_labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_no_merge(rwkv: RWKVModel, dataset, max_new_tokens: int, record_path: str):
    correct = 0
    progress = tqdm(range(len(dataset)), desc="No Merge Eval")
    for idx in progress:
        row = dataset[idx]
        pred = rwkv.generate(question=row["question"], init_state=row["state"], max_new_tokens=max_new_tokens)
        answers = row.get("answers", [])
        em = squad_em(pred, answers)
        correct += em
        append_jsonl_row(
            record_path,
            {"index": idx, "question": row["question"], "prediction": pred, "answers": answers, "em": int(em)},
        )
        progress.set_postfix_str(f"acc={(correct / (idx + 1)) * 100.0:.2f}%")
    return {"accuracy": (correct / len(dataset)) if len(dataset) else 0.0}


def evaluate_mixer_merge_2(
    rwkv: RWKVModel, dataset, mixer: HeadwiseStateMixer, max_new_tokens: int, record_path: str
):
    correct = 0
    total = 0
    progress = tqdm(range(0, len(dataset), 2), desc="Mixer Merge Eval (N=2)")
    for start in progress:
        if start + 1 >= len(dataset):
            break
        row_a = dataset[start]
        row_b = dataset[start + 1]
        merged_state = mixer_merge_states(mixer, row_a["state"], row_b["state"])
        for i, row in enumerate([row_a, row_b]):
            idx = start + i
            pred = rwkv.generate(question=row["question"], init_state=merged_state, max_new_tokens=max_new_tokens)
            answers = row.get("answers", [])
            em = squad_em(pred, answers)
            correct += em
            total += 1
            append_jsonl_row(
                record_path,
                {
                    "index": idx,
                    "question": row["question"],
                    "prediction": pred,
                    "answers": answers,
                    "em": int(em),
                    "group_size": 2,
                    "mode": "mixer_2",
                },
            )
            progress.set_postfix_str(f"acc={(correct / total) * 100.0:.2f}%")
    return {"accuracy": (correct / total) if total else 0.0}


def _build_headwise_for_kind(model_kind: str, ckpt_args: dict | None) -> HeadwiseStateMixer:
    """Return an unbuilt HeadwiseStateMixer wired to the right inner mixer class."""
    ckpt_args = ckpt_args or {}
    if model_kind == "conv":
        return HeadwiseStateMixer(
            mixer_cls=DynamicStateMixer,
            mixer_kwargs={
                "d_model": int(ckpt_args.get("d_model", 32)),
                "d_ffn": int(ckpt_args.get("d_ffn", 64)),
                "n_attn_heads": int(ckpt_args.get("n_attn_heads", 2)),
                "delta_rank": int(ckpt_args.get("delta_rank", 1)),
                "max_layers": int(ckpt_args.get("max_layers", 128)),
            },
        )
    if model_kind == "tiny_rwkv":
        config = TinyRWKVMergerConfig(
            d_model=int(ckpt_args.get("d_model", 32)),
            d_ffn=int(ckpt_args.get("d_ffn", 64)),
            max_layers=int(ckpt_args.get("max_layers", 128)),
            use_low_rank_mask=not bool(ckpt_args.get("no_low_rank_mask", False)),
            dropout=float(ckpt_args.get("dropout", 0.0)),
        )
        return HeadwiseStateMixer(
            mixer_cls=TinyRWKVStateMerger,
            mixer_kwargs={"config": config},
        )
    raise ValueError(f"Unknown model_kind in checkpoint: {model_kind!r}")


def load_mixer(ckpt_path: str, sample_state: Any) -> HeadwiseStateMixer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint format mismatch: expected dict with 'model_state_dict', got {ckpt_path}."
        )
    state_dict = ckpt["model_state_dict"]
    # Back-compat: older checkpoints (pre-pivot) have no 'model_kind' key.
    # Those were all conv-based.
    model_kind = ckpt.get("model_kind", "conv")
    ckpt_args = ckpt.get("args")
    model = _build_headwise_for_kind(model_kind, ckpt_args)
    model.build_from_state(sample_state)

    if not any(k.startswith("_mixer.") for k in state_dict.keys()):
        raise RuntimeError(
            f"Checkpoint state_dict does not contain '_mixer.*' keys: {ckpt_path}. "
            "Please use a unified checkpoint saved by the current train_mixer.py."
        )
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"[load_mixer] kind={model_kind} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    return model


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "mixer_merge_eval")
    os.makedirs(output_dir, exist_ok=True)
    apply_plot_style()
    state_dir = args.state_dir or default_state_dir(args.model_filename)

    rwkv = RWKVModel(
        model_path=args.model_path,
        model_filename=args.model_filename,
        strategy=args.strategy,
        tokenizer=args.tokenizer,
    )
    dataset = load_validation_state_dataset(state_dir=state_dir, limit=args.num_samples)
    if len(dataset) == 0:
        raise ValueError("Empty evaluation dataset.")
    mixer = load_mixer(args.mixer_ckpt, dataset[0]["state"])

    results = OrderedDict()
    no_merge_path = init_mode_record_file(output_dir, "no_merge")
    mixer_merge_path = init_mode_record_file(output_dir, "mixer_merge_2")
    no_merge_result = evaluate_no_merge(rwkv, dataset, args.max_new_tokens, no_merge_path)
    mixer_result = evaluate_mixer_merge_2(rwkv, dataset, mixer, args.max_new_tokens, mixer_merge_path)

    results["No Merge"] = no_merge_result["accuracy"]
    results["Mixer Merge 2"] = mixer_result["accuracy"]
    results["Accuracy Gap (Mixer2 - NoMerge)"] = mixer_result["accuracy"] - no_merge_result["accuracy"]

    result_json = os.path.join(output_dir, "squad_mixer_merge_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "mixer_merge_group_outcome_stacked.png")
    mode_data = build_mode_distributions(output_dir)
    plot_stacked_distribution(mode_data, plot_path)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved summary: {result_json}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved no-merge records: {no_merge_path}")
    print(f"Saved mixer-merge records: {mixer_merge_path}")


if __name__ == "__main__":
    main()
