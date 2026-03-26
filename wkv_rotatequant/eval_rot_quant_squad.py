import argparse
import json
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

from rot_quant_projector import RotationQuantProjector, move_state_to_device
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
    parser = argparse.ArgumentParser(description="Evaluate RWKV accuracy with learned rotate->STE quant->inverse pipeline.")
    parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv7-g1")
    parser.add_argument("--model-filename", type=str, default="rwkv7-g1e-1.5b-20260309-ctx8192.pth")
    parser.add_argument("--strategy", type=str, default="cuda fp16")
    parser.add_argument("--tokenizer", type=str, default="rwkv_vocab_v20230424")
    parser.add_argument("--state-dir", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--projector-ckpt", type=str, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def evaluate_plain(rwkv: RWKVModel, dataset, max_new_tokens: int, record_path: str) -> float:
    correct = 0
    for idx in tqdm(range(len(dataset)), desc="Eval plain"):
        row = dataset[idx]
        pred = rwkv.generate(question=row["question"], init_state=row["state"], max_new_tokens=max_new_tokens)
        em = squad_em(pred, row.get("answers", []))
        correct += em
        append_jsonl_row(
            record_path,
            {
                "index": idx,
                "mode": "plain",
                "question": row["question"],
                "prediction": pred,
                "answers": row.get("answers", []),
                "em": int(em),
            },
        )
    return correct / max(len(dataset), 1)


def evaluate_rot_quant(
    rwkv: RWKVModel,
    dataset,
    projector: RotationQuantProjector,
    bits: int,
    max_new_tokens: int,
    record_path: str,
) -> float:
    correct = 0
    device = next(projector.parameters()).device
    projector.eval()
    for idx in tqdm(range(len(dataset)), desc=f"Eval rot-quant {bits}bit"):
        row = dataset[idx]
        state = move_state_to_device(row["state"], device)
        with torch.no_grad():
            rec_state = projector.transform_state(state, bits=bits)
        rec_state = move_state_to_device(rec_state, torch.device("cpu"))

        pred = rwkv.generate(question=row["question"], init_state=rec_state, max_new_tokens=max_new_tokens)
        em = squad_em(pred, row.get("answers", []))
        correct += em
        append_jsonl_row(
            record_path,
            {
                "index": idx,
                "mode": "rot_quant",
                "bits": int(bits),
                "question": row["question"],
                "prediction": pred,
                "answers": row.get("answers", []),
                "em": int(em),
            },
        )
    return correct / max(len(dataset), 1)


def main():
    args = parse_args()
    output_dir = default_result_dir(args.model_filename, "wkv_rotatequant")
    os.makedirs(output_dir, exist_ok=True)

    rwkv = RWKVModel(
        model_path=args.model_path,
        model_filename=args.model_filename,
        strategy=args.strategy,
        tokenizer=args.tokenizer,
    )
    state_dir = args.state_dir or default_state_dir(args.model_filename)
    dataset = load_validation_state_dataset(state_dir=state_dir, limit=args.num_samples)

    projector = RotationQuantProjector()
    projector.build_from_state(dataset[0]["state"])
    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    projector.load_state_dict(ckpt["model_state_dict"])
    projector.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    plain_path = os.path.join(output_dir, "plain_records.jsonl")
    quant_path = os.path.join(output_dir, "rot_quant_records.jsonl")
    init_jsonl_file(plain_path)
    init_jsonl_file(quant_path)

    plain_acc = evaluate_plain(
        rwkv=rwkv,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        record_path=plain_path,
    )
    rot_quant_acc = evaluate_rot_quant(
        rwkv=rwkv,
        dataset=dataset,
        projector=projector,
        bits=args.bits,
        max_new_tokens=args.max_new_tokens,
        record_path=quant_path,
    )

    results = OrderedDict(
        [
            ("plain", {"accuracy": plain_acc}),
            ("rot_quant", {"accuracy": rot_quant_acc, "bits": int(args.bits), "checkpoint": args.projector_ckpt}),
        ]
    )
    out_json = os.path.join(output_dir, "rot_quant_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {out_json}")
    print(f"plain acc={plain_acc:.4f}")
    print(f"rot-quant acc={rot_quant_acc:.4f}")


if __name__ == "__main__":
    main()
