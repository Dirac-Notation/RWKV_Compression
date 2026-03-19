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

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from state_autoencoder import (
    AutoEncoderConfig,
    StateStructureAutoEncoder,
    add_states,
    clone_state,
    move_state_to_cpu,
    move_state_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate latent-level state merge on SQuAD.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-7.2b-20260301-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--state-index", type=str, default="/data/.cache/data/val_index.json")
    parser.add_argument("--ae-checkpoint", type=str, default="./encode_compress/checkpoints/best.pt")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--output-dir", type=str, default="./encode_compress/eval_outputs")
    return parser.parse_args()


def resolve_rwkv_model_path(model_path: str, model_filename: str = "") -> str:
    if os.path.isfile(model_path):
        if model_path.endswith(".pth"):
            return model_path[:-4]
        return model_path
    if os.path.isfile(model_path + ".pth"):
        return model_path
    if model_path.endswith(".pth"):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if "/" not in model_path:
        raise FileNotFoundError(
            f"Invalid model path '{model_path}'. "
            "Use local .pth path or Hugging Face repo id like 'BlinkDL/rwkv7-g1'."
        )
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for auto download. "
            "Install it with: pip install huggingface_hub"
        ) from e
    repo_files = list_repo_files(repo_id=model_path)
    pth_files = [f for f in repo_files if f.endswith(".pth")]
    if not pth_files:
        raise FileNotFoundError(f"No .pth file found in Hugging Face repo: {model_path}")

    target_file = model_filename or sorted(pth_files)[-1]
    if target_file not in pth_files:
        raise FileNotFoundError(
            f"'{target_file}' not found in repo {model_path}. Available: {pth_files}"
        )
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=model_path,
        filename=target_file,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    if not local_path.endswith(".pth"):
        raise RuntimeError(f"Downloaded file is not a .pth model: {local_path}")
    return local_path[:-4]


def build_question_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def strip_think_block(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    return text.strip()


@torch.no_grad()
def generate_from_state(model, pipeline, gen_args, init_state, question: str, max_new_tokens: int) -> str:
    prompt = build_question_prompt(question)
    state = clone_state(init_state)
    tokens = pipeline.encode(prompt)
    out, state = model.forward(tokens, state)

    generated = []
    seen_think_open = False
    seen_think_close = False

    for _ in range(max_new_tokens):
        token = pipeline.sample_logits(
            out, temperature=gen_args.temperature, top_p=gen_args.top_p, top_k=gen_args.top_k
        )
        candidate = pipeline.decode(generated + [token])
        if "�" in candidate:
            break
        generated.append(token)
        lower = candidate.lower()
        if not seen_think_open and "<think>" in lower:
            seen_think_open = True
        if not seen_think_open and len(generated) >= 32:
            break
        if "</think>" in lower:
            seen_think_close = True
            out, state = model.forward([token], state)
            break
        out, state = model.forward([token], state)

    if seen_think_open and not seen_think_close:
        forced = pipeline.encode("</think>")
        for t in forced:
            candidate = pipeline.decode(generated + [t])
            if "�" in candidate:
                break
            generated.append(t)
            out, state = model.forward([t], state)
        seen_think_close = "</think>" in pipeline.decode(generated).lower()

    if seen_think_close:
        for _ in range(32):
            token = pipeline.sample_logits(
                out, temperature=gen_args.temperature, top_p=gen_args.top_p, top_k=gen_args.top_k
            )
            candidate = pipeline.decode(generated + [token])
            if "�" in candidate:
                break
            generated.append(token)
            out, state = model.forward([token], state)

    return strip_think_block(pipeline.decode(generated).strip())


@torch.no_grad()
def answer_from_state(model, pipeline, gen_args, state, question: str, max_new_tokens: int) -> str:
    infer_state = move_state_to_device(state, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return generate_from_state(model, pipeline, gen_args, infer_state, question, max_new_tokens)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def squad_em(prediction: str, answers: Sequence[str]) -> int:
    if not answers:
        return 0
    norm_pred = normalize_answer(prediction)
    return int(any(normalize_answer(a) in norm_pred for a in answers))


def init_mode_record_file(output_dir: str, mode_name: str) -> str:
    path = os.path.join(output_dir, f"{mode_name}_qa_records.jsonl")
    with open(path, "w", encoding="utf-8"):
        pass
    return path


def append_mode_record(path: str, row: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


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
    if name.startswith("no_merge_ae_"):
        return 1
    match = re.search(r"merge_latent_sum_(\d+)_", name)
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
    if mode_name == "No Merge (AE)":
        return (0, 0)
    m = re.match(r"Merge Latent Sum (\d+)", mode_name)
    if m:
        return (1, int(m.group(1)))
    return (2, mode_name)


def mode_label_from_file(file_path: str, group_size: int) -> str:
    name = os.path.basename(file_path)
    if name.startswith("no_merge_ae_"):
        return "No Merge (AE)"
    return f"Merge Latent Sum {group_size}"


def build_mode_distributions(output_dir: str):
    mode_data = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith("_qa_records.jsonl"):
            continue
        if not (name.startswith("no_merge_ae_") or name.startswith("merge_latent_sum_")):
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
    labels = [x["mode"] for x in mode_data]
    x_positions = list(range(len(mode_data)))
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
    plt.xlabel("Evaluation Mode")
    plt.ylim(0, 1)
    plt.xticks(x_positions, labels)
    plt.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
    for i, entry in enumerate(mode_data):
        text_y = min(0.98, entry["accuracy"] + 0.02)
        plt.text(
            i,
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


def load_state_records_from_index(index_path: str):
    with open(index_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    records = []
    for row in tqdm(rows, desc="Load eval states"):
        sample = torch.load(row["file"], map_location="cpu", weights_only=False)
        records.append(sample)
    return records


def build_ae_from_checkpoint(ckpt_path: str, sample_state, device: torch.device) -> StateStructureAutoEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_config = ckpt.get("config", {})
    config = AutoEncoderConfig(
        dropout=float(raw_config.get("dropout", 0.0)),
    )
    ae = StateStructureAutoEncoder(config)
    ae.build_from_state(sample_state)
    ae.load_state_dict(ckpt["model_state_dict"], strict=True)
    ae.to(device)
    ae.eval()
    return ae


@torch.no_grad()
def reconstruct_state_with_ae(ae: StateStructureAutoEncoder, state, device: torch.device):
    state_device = move_state_to_device(state, device)
    latent = ae.encode_state(state_device)
    recon = ae.decode_state(latent)
    return move_state_to_cpu(recon)


@torch.no_grad()
def encode_state_with_ae(ae: StateStructureAutoEncoder, state, device: torch.device):
    state_device = move_state_to_device(state, device)
    latent = ae.encode_state(state_device)
    return move_state_to_cpu(latent)


@torch.no_grad()
def decode_latent_with_ae(ae: StateStructureAutoEncoder, latent_state, device: torch.device):
    latent_device = move_state_to_device(latent_state, device)
    recon = ae.decode_state(latent_device)
    return move_state_to_cpu(recon)


def evaluate_no_merge_ae(
    model,
    pipeline,
    gen_args,
    ae: StateStructureAutoEncoder,
    states,
    max_new_tokens: int,
    record_path: str,
    device: torch.device,
):
    correct = 0
    pbar = tqdm(states, desc="No Merge (AE)")
    for idx, row in enumerate(pbar):
        recon_state = reconstruct_state_with_ae(ae, row["state"], device)
        pred = answer_from_state(model, pipeline, gen_args, recon_state, row["question"], max_new_tokens)
        em = squad_em(pred, row["answers"])
        correct += em
        append_mode_record(
            record_path,
            {
                "index": row["index"],
                "question": row["question"],
                "prediction": pred,
                "answers": row["answers"],
                "em": int(em),
            },
        )
        running_acc = correct / (idx + 1) * 100.0
        pbar.set_postfix_str(f"acc={running_acc:.2f}%")
    return {"accuracy": correct / len(states) if states else 0.0}


def evaluate_merge_ae(
    model,
    pipeline,
    gen_args,
    ae: StateStructureAutoEncoder,
    states,
    group_size: int,
    max_new_tokens: int,
    record_path: str,
    device: torch.device,
):
    correct = 0
    total = 0
    pbar = tqdm(range(0, len(states), group_size), desc=f"Merge Latent Sum (N={group_size})")
    for start in pbar:
        group = states[start : start + group_size]
        if len(group) < group_size:
            break

        latent_group = [encode_state_with_ae(ae, row["state"], device) for row in group]
        merged_latent = copy.deepcopy(latent_group[0])
        for latent in latent_group[1:]:
            merged_latent = add_states(merged_latent, latent)
        merged_state = decode_latent_with_ae(ae, merged_latent, device)

        for row in group:
            pred = answer_from_state(model, pipeline, gen_args, merged_state, row["question"], max_new_tokens)
            em = squad_em(pred, row["answers"])
            correct += em
            total += 1
            append_mode_record(
                record_path,
                {
                    "index": row["index"],
                    "question": row["question"],
                    "prediction": pred,
                    "answers": row["answers"],
                    "em": int(em),
                    "group_size": group_size,
                },
            )
        pbar.set_postfix_str(f"acc={(correct / max(total, 1)) * 100.0:.2f}%")
    return {"accuracy": correct / total if total else 0.0}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_plot_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_records = load_state_records_from_index(args.state_index)
    if not state_records:
        raise ValueError(f"Empty state dataset: {args.state_index}")

    ae = build_ae_from_checkpoint(args.ae_checkpoint, state_records[0]["state"], device)

    resolved_model_path = resolve_rwkv_model_path(args.model_path, args.model_filename)
    model = RWKV(model=resolved_model_path, strategy=args.strategy)
    pipeline = PIPELINE(model, args.tokenizer)
    gen_args = PIPELINE_ARGS(
        temperature=0.0,
        top_p=0.0,
        top_k=1,
        alpha_frequency=0.0,
        alpha_presence=0.0,
        alpha_decay=1.0,
        token_ban=[],
        token_stop=[],
        chunk_len=256,
    )
    max_new_tokens = 256

    results = OrderedDict()
    no_merge_path = init_mode_record_file(args.output_dir, "no_merge_ae")
    no_merge = evaluate_no_merge_ae(
        model, pipeline, gen_args, ae, state_records, max_new_tokens, no_merge_path, device
    )
    results["No Merge (AE)"] = no_merge["accuracy"]

    for n in sorted(set(args.group_sizes)):
        if n > 1:
            merge_path = init_mode_record_file(args.output_dir, f"merge_latent_sum_{n}")
            merged = evaluate_merge_ae(
                model, pipeline, gen_args, ae, state_records, n, max_new_tokens, merge_path, device
            )
            results[f"Merge Latent Sum {n}"] = merged["accuracy"]

    out_json = os.path.join(args.output_dir, "latent_merge_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(args.output_dir, "group_outcome_stacked.png")
    mode_data = build_mode_distributions(args.output_dir)
    plot_stacked_distribution(mode_data, plot_path)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
