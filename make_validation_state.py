import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm

from rwkv_model import DEFAULT_SYSTEM_PROMPT, RWKVModel, default_state_dir

DEFAULT_RANDOM_SEED = 42


@dataclass
class SquadSample:
    context: str
    question: str
    answers: List[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Build prefilled validation states for RWKV experiments.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


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


def main():
    args = parse_args()
    state_dir = default_state_dir(args.model_filename, dataset_root="./dataset")
    os.makedirs(state_dir, exist_ok=True)

    rwkv = RWKVModel(
        model_path=args.model_path,
        model_filename=args.model_filename,
        strategy=args.strategy,
        tokenizer=args.tokenizer,
    )
    samples = load_squad_samples(split=args.split, limit=args.num_samples, seed=DEFAULT_RANDOM_SEED)

    index_rows = []
    for idx, sample in enumerate(tqdm(samples, desc="Build Prefilled Validation States")):
        state = rwkv.prefill_state(sample.context, system_prompt=args.system_prompt)
        file_name = f"{idx:08d}.pt"
        torch.save(
            {
                "index": idx,
                "context": sample.context,
                "question": sample.question,
                "answers": sample.answers,
                "state": state,
            },
            os.path.join(state_dir, file_name),
        )
        index_rows.append(
            {
                "index": idx,
                "file": file_name,
            }
        )

    meta = {
        "model_path": args.model_path,
        "model_filename": args.model_filename,
        "strategy": args.strategy,
        "tokenizer": args.tokenizer,
        "split": args.split,
        "num_samples": len(index_rows),
        "seed": DEFAULT_RANDOM_SEED,
        "system_prompt": args.system_prompt,
    }
    index_path = os.path.join(state_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)

    meta_path = os.path.join(state_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved state dir: {state_dir}")
    print(f"Saved records: {len(index_rows)}")
    print(f"Saved index: {index_path}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
