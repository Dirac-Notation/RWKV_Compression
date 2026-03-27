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


def build_pair_text(samples: Sequence[SquadSample], left_idx: int, right_idx: int) -> str:
    return f"{samples[left_idx].context} {samples[right_idx].context}".strip()


def write_state_records(
    model,
    pipeline,
    samples: Sequence[SquadSample],
    system_prompt: str,
    split_name: str,
    output_dir: str,
):
    split_dir = os.path.join(output_dir, split_name)
    one_state_dir = os.path.join(split_dir, "one_state")
    two_state_dir = os.path.join(split_dir, "two_state")
    os.makedirs(one_state_dir, exist_ok=True)
    os.makedirs(two_state_dir, exist_ok=True)
    text_rows = []
    one_state_rows = []
    two_state_rows = []
    first_state_summary = {}

    for idx, sample in enumerate(tqdm(samples, desc=f"Build {split_name} one_state")):
        state = prefill_state_from_context(model, pipeline, system_prompt, sample.context)
        one_path = os.path.join(one_state_dir, f"{idx}.pt")
        torch.save({"index": idx, "state": state}, one_path)
        if idx == 0:
            first_state_summary = summarize_state_shapes(state)
        text_rows.append(
            {
                "index": idx,
                "text": sample.context,
                "question": sample.question,
                "answers": sample.answers,
            }
        )
        one_state_rows.append({"index": idx, "file": one_path})

    for left_idx in tqdm(range(max(len(samples) - 1, 0)), desc=f"Build {split_name} two_state"):
        right_idx = left_idx + 1
        pair_text = build_pair_text(samples, left_idx, right_idx)
        pair_state = prefill_state_from_context(model, pipeline, system_prompt, pair_text)
        pair_name = f"{left_idx}_{right_idx}.pt"
        pair_path = os.path.join(two_state_dir, pair_name)
        torch.save(
            {
                "pair": [left_idx, right_idx],
                "left_index": left_idx,
                "right_index": right_idx,
                "text": pair_text,
                "state": pair_state,
            },
            pair_path,
        )
        two_state_rows.append({"pair": [left_idx, right_idx], "file": pair_path})

    texts_path = os.path.join(split_dir, "texts.jsonl")
    with open(texts_path, "w", encoding="utf-8") as f:
        for row in text_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    one_index_path = os.path.join(split_dir, "one_state_index.json")
    with open(one_index_path, "w", encoding="utf-8") as f:
        json.dump(one_state_rows, f, ensure_ascii=False, indent=2)

    two_index_path = os.path.join(split_dir, "two_state_index.json")
    with open(two_index_path, "w", encoding="utf-8") as f:
        json.dump(two_state_rows, f, ensure_ascii=False, indent=2)

    return {
        "count": len(one_state_rows),
        "split_dir": split_dir,
        "texts_file": texts_path,
        "one_state_dir": one_state_dir,
        "two_state_dir": two_state_dir,
        "one_state_index_file": one_index_path,
        "two_state_index_file": two_index_path,
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
        "train_texts_file": train_info["texts_file"],
        "val_texts_file": val_info["texts_file"],
        "train_one_state_dir": train_info["one_state_dir"],
        "val_one_state_dir": val_info["one_state_dir"],
        "train_two_state_dir": train_info["two_state_dir"],
        "val_two_state_dir": val_info["two_state_dir"],
        "train_one_state_index_file": train_info["one_state_index_file"],
        "val_one_state_index_file": val_info["one_state_index_file"],
        "train_two_state_index_file": train_info["two_state_index_file"],
        "val_two_state_index_file": val_info["two_state_index_file"],
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved train one_state: {train_info['one_state_dir']} ({train_info['count']})")
    print(f"Saved train two_state: {train_info['two_state_dir']}")
    print(f"Saved val one_state: {val_info['one_state_dir']} ({val_info['count']})")
    print(f"Saved val two_state: {val_info['two_state_dir']}")


if __name__ == "__main__":
    main()
