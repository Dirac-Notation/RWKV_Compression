import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rot_quant_projector import RotationQuantProjector, move_state_to_device, stack_states, wkv_only_mse


class StateOnlyDataset(Dataset):
    def __init__(self, index_file: str):
        with open(index_file, "r", encoding="utf-8") as f:
            self.records: List[Dict[str, object]] = json.load(f)
        if not self.records:
            raise ValueError(f"Empty index file: {index_file}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        sample = torch.load(row["file"], map_location="cpu")
        return sample["state"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train bilateral rotation projector with STE quantization.")
    parser.add_argument("--train-index", type=str, default="./data/train_index.json")
    parser.add_argument("--val-index", type=str, default="./data/val_index.json")
    parser.add_argument("--output-dir", type=str, default="./wkv_rotatequant/checkpoints")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser.parse_args()


def collate_states(batch):
    return stack_states(batch)


@torch.no_grad()
def evaluate(model: RotationQuantProjector, loader: DataLoader, device: torch.device, bits: int) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_state_to_device(batch, device)
        pred = model.transform_state(batch, bits=bits)
        loss = wkv_only_mse(pred, batch)
        total += float(loss.detach().cpu().item())
        n += 1
    return total / max(n, 1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = StateOnlyDataset(args.train_index)
    val_ds = StateOnlyDataset(args.val_index)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_states,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_states,
    )

    model = RotationQuantProjector()
    model.build_from_state(train_ds[0])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_val = float("inf")
    history_path = os.path.join(args.output_dir, "history.jsonl")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            batch = move_state_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.transform_state(batch, bits=args.bits)
            loss = wkv_only_mse(pred, batch)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu().item())
            steps += 1
            pbar.set_postfix(loss=f"{running / max(steps, 1):.6f}")

        train_loss = running / max(steps, 1)
        val_loss = evaluate(model, val_loader, device=device, bits=args.bits)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "bits": int(args.bits)}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "model_state_dict": model.state_dict(),
                    "bits": int(args.bits),
                },
                os.path.join(args.output_dir, "best.pt"),
            )
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")


if __name__ == "__main__":
    main()
