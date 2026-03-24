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
    parser = argparse.ArgumentParser(description="Evaluate RWKV robustness with Gaussian state noise.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    parser.add_argument("--noise-seed", type=int, default=1234)
    return parser.parse_args()


def add_gaussian_noise_to_state(state, std: float, generator: torch.Generator):
    if isinstance(state, list):
        return [add_gaussian_noise_to_state(x, std, generator) for x in state]
    if isinstance(state, tuple):
        return tuple(add_gaussian_noise_to_state(x, std, generator) for x in state)
    if torch.is_tensor(state):
        if std <= 0:
            return state.clone()
        if not torch.is_floating_point(state):
            return state.clone()
        noise = torch.randn(
            state.shape,
            generator=generator,
            device=state.device,
            dtype=torch.float32,
        ) * std
        return state + noise.to(dtype=state.dtype)
    return copy.deepcopy(state)


def init_noise_record_file(output_dir: str, noise_std: float) -> str:
    noise_name = str(noise_std).replace(".", "_")
    path = os.path.join(output_dir, f"noise_{noise_name}_qa_records.jsonl")
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


def plot_noise_robustness(acc_by_noise: OrderedDict, output_path: str):
    stds = list(acc_by_noise.keys())
    accs = [acc_by_noise[s] for s in stds]
    x = list(range(len(stds)))
    labels = [str(s) for s in stds]
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
            fontsize=11,
        )
    plt.ylim(0, min(1.0, max(accs + [0.1]) + 0.08))
    plt.ylabel("Accuracy")
    plt.xlabel("Gaussian Noise Std")
    plt.title("RWKV State Robustness Under Gaussian Noise", pad=12)
    plt.xticks(x, labels)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_with_noise(
    rwkv: RWKVModel,
    dataset,
    noise_std: float,
    max_new_tokens: int,
    record_path: str,
    noise_seed: int,
):
    correct = 0
    progress = tqdm(range(len(dataset)), desc=f"Noise Eval (std={noise_std})")
    for idx in progress:
        row = dataset[idx]
        g = torch.Generator(device="cpu")
        g.manual_seed(noise_seed + idx)
        noisy_state = add_gaussian_noise_to_state(row["state"], noise_std, g)
        pred = rwkv.generate(question=row["question"], init_state=noisy_state, max_new_tokens=max_new_tokens)
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
                "noise_std": noise_std,
            },
        )
        running_acc = (correct / (idx + 1)) * 100.0
        progress.set_postfix_str(f"acc={running_acc:.2f}%")
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
    return {"accuracy": accuracy}


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "noise")
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
    for noise_std in args.noise_stds:
        record_path = init_noise_record_file(output_dir, noise_std)
        result = evaluate_with_noise(
            rwkv=rwkv,
            dataset=dataset,
            noise_std=float(noise_std),
            max_new_tokens=max_new_tokens,
            record_path=record_path,
            noise_seed=args.noise_seed,
        )
        acc[float(noise_std)] = result["accuracy"]

    result_json = os.path.join(output_dir, "squad_rwkv_noise_eval_results.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(acc, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(output_dir, "noise_robustness_bar.png")
    plot_noise_robustness(acc, plot_path)

    print(f"Saved summary: {result_json}")
    print(f"Saved plot: {plot_path}")
    for std, value in acc.items():
        print(f"Noise std {std}: {value:.4f}")


if __name__ == "__main__":
    main()
