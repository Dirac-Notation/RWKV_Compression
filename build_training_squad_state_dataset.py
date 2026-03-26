import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question only using the provided context. "
    "You must always reason in <think>...</think> format first, "
    "then provide the final answer after </think>."
)
DEFAULT_RANDOM_SEED = 42


def move_state_to_cpu(state):
    if isinstance(state, list):
        return [move_state_to_cpu(x) for x in state]
    if isinstance(state, tuple):
        return tuple(move_state_to_cpu(x) for x in state)
    if torch.is_tensor(state):
        return state.detach().cpu()
    return state


@dataclass
class SquadSample:
    context: str
    question: str
    answers: List[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Build RWKV state dataset from SQuAD.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="./data")
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


def load_squad_samples(split: str, limit: int, seed: int) -> List[SquadSample]:
    ds = load_dataset("squad", split=split)
    if limit > 0:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    return [
        SquadSample(
            context=row["context"],
            question=row["question"],
            answers=row["answers"]["text"] if "answers" in row else [],
        )
        for row in ds
    ]


def build_prefill_prompt(system_prompt: str, context: str) -> str:
    return f"System: {system_prompt}\n\nContext: {context}\n\n"


@torch.no_grad()
def prefill_state_from_context(model, pipeline, system_prompt: str, context: str):
    prompt = build_prefill_prompt(system_prompt, context)
    tokens = pipeline.encode(prompt)
    _, state = model.forward(tokens, None)
    return move_state_to_cpu(state)


def summarize_state_shapes(state) -> Dict[str, object]:
    summary = {}

    def walk(x, path):
        key = "root" if not path else ".".join(str(p) for p in path)
        if torch.is_tensor(x):
            summary[key] = {"shape": list(x.shape), "dtype": str(x.dtype)}
            return
        if isinstance(x, list):
            for i, item in enumerate(x):
                walk(item, path + [i])
            return
        if isinstance(x, tuple):
            for i, item in enumerate(x):
                walk(item, path + [i])
            return
        summary[key] = {"type": str(type(x))}

    walk(state, [])
    return summary


def write_state_records(
    model,
    pipeline,
    samples: Sequence[SquadSample],
    system_prompt: str,
    split_name: str,
    output_dir: str,
):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    index_rows = []
    first_state_summary = {}

    for idx, sample in enumerate(tqdm(samples, desc=f"Build {split_name} states")):
        state = prefill_state_from_context(model, pipeline, system_prompt, sample.context)
        row = {
            "index": idx,
            "context": sample.context,
            "question": sample.question,
            "answers": sample.answers,
            "state": state,
        }
        state_path = os.path.join(split_dir, f"{idx:08d}.pt")
        torch.save(row, state_path)
        if idx == 0:
            first_state_summary = summarize_state_shapes(state)
        index_rows.append(
            {
                "index": idx,
                "file": state_path,
            }
        )

    index_path = os.path.join(output_dir, f"{split_name}_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)

    return {
        "count": len(index_rows),
        "index_file": index_path,
        "split_dir": split_dir,
        "state_shape_summary": first_state_summary,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    resolved_model_path = resolve_rwkv_model_path(args.model_path, args.model_filename)
    model = RWKV(model=resolved_model_path, strategy=args.strategy)
    pipeline = PIPELINE(model, args.tokenizer)

    train_samples = load_squad_samples("train", args.train_limit, DEFAULT_RANDOM_SEED)
    val_samples = load_squad_samples("validation", args.val_limit, DEFAULT_RANDOM_SEED)

    train_info = write_state_records(
        model=model,
        pipeline=pipeline,
        samples=train_samples,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        split_name="train",
        output_dir=args.output_dir,
    )
    val_info = write_state_records(
        model=model,
        pipeline=pipeline,
        samples=val_samples,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        split_name="val",
        output_dir=args.output_dir,
    )

    meta = {
        "model_path": args.model_path,
        "model_filename": args.model_filename,
        "strategy": args.strategy,
        "tokenizer": args.tokenizer,
        "seed": DEFAULT_RANDOM_SEED,
        "train_split": "train",
        "train_count": train_info["count"],
        "val_split": "validation",
        "val_count": val_info["count"],
        "val_seed": DEFAULT_RANDOM_SEED,
        "state_shape_summary": train_info["state_shape_summary"],
        "train_split_dir": train_info["split_dir"],
        "val_split_dir": val_info["split_dir"],
        "train_index_file": train_info["index_file"],
        "val_index_file": val_info["index_file"],
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved train index: {train_info['index_file']} ({train_info['count']})")
    print(f"Saved val index: {val_info['index_file']} ({val_info['count']})")


if __name__ == "__main__":
    main()
