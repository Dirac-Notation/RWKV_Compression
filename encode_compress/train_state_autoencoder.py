import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from state_autoencoder import (
    AutoEncoderConfig,
    StateStructureAutoEncoder,
    move_state_to_device,
    peak_aware_wkv_loss,
    stack_states,
)


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
    parser = argparse.ArgumentParser(description="Train RWKV state AutoEncoder.")
    parser.add_argument("--train-index", type=str, default="/data/.cache/data/train_index.json")
    parser.add_argument("--val-index", type=str, default="/data/.cache/data/val_index.json")
    parser.add_argument("--output-dir", type=str, default="./encode_compress/checkpoints")
    return parser.parse_args()


def collate_states(batch):
    return stack_states(batch)


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def set_optimizer_lr(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = new_lr


@torch.no_grad()
def evaluate(
    model: StateStructureAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    topk_weight: float,
    topk_ratio: float,
) -> Dict[str, float]:
    model.eval()
    totals: List[float] = []
    mses: List[float] = []
    topk_mses: List[float] = []
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_state_to_device(batch, device)
        recon = model.forward_state(batch)
        _, metrics = peak_aware_wkv_loss(
            recon,
            batch,
            topk_weight=topk_weight,
            topk_ratio=topk_ratio,
        )
        totals.append(metrics["total"])
        mses.append(metrics["mse"])
        topk_mses.append(metrics["topk_mse"])

    if not totals:
        return {"total": 0.0, "mse": 0.0, "topk_mse": 0.0}
    return {
        "total": float(sum(totals) / len(totals)),
        "mse": float(sum(mses) / len(mses)),
        "topk_mse": float(sum(topk_mses) / len(topk_mses)),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed training defaults to keep CLI minimal.
    epochs = 1000
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-2
    dropout = 0.0
    min_lr = 1e-7
    topk_weight = 0.1
    topk_ratio = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = StateOnlyDataset(args.train_index)
    val_ds = StateOnlyDataset(args.val_index)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_states,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_states,
    )

    config = AutoEncoderConfig(
        dropout=dropout,
    )
    model = StateStructureAutoEncoder(config)
    # Build module cache from first batch state structure before loading/optimizing.
    model.build_from_state(train_ds[0])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_schedule": "loss_ratio_to_first_epoch_val_loss",
        "min_lr": min_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "topk_weight": topk_weight,
        "topk_ratio": topk_ratio,
    }
    args_path = os.path.join(args.output_dir, "train_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cli_args": vars(args),
                "fixed_train_config": run_config,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    history_path = os.path.join(args.output_dir, "history.jsonl")
    best_val_loss = float("inf")
    reference_val_loss = None
    current_lr = lr
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        running_total = 0.0
        running_mse = 0.0
        running_topk_mse = 0.0
        step_count = 0

        for batch in pbar:
            batch = move_state_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            recon = model.forward_state(batch)
            loss, metrics = peak_aware_wkv_loss(
                recon,
                batch,
                topk_weight=topk_weight,
                topk_ratio=topk_ratio,
            )
            loss.backward()
            gn = grad_norm(model)
            optimizer.step()

            running_total += metrics["total"]
            running_mse += metrics["mse"]
            running_topk_mse += metrics["topk_mse"]
            step_count += 1
            pbar.set_postfix(
                loss=f"{running_total / step_count:.6f}",
                mse=f"{running_mse / step_count:.6f}",
                topk=f"{running_topk_mse / step_count:.6f}",
                grad_norm=f"{gn:.4f}",
            )

        denom = max(step_count, 1)
        train_total = running_total / denom
        train_mse = running_mse / denom
        train_topk_mse = running_topk_mse / denom
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            topk_weight=topk_weight,
            topk_ratio=topk_ratio,
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_total,
            "train_mse": train_mse,
            "train_topk_mse": train_topk_mse,
            "val_loss": val_metrics["total"],
            "val_mse": val_metrics["mse"],
            "val_topk_mse": val_metrics["topk_mse"],
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        best_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "best_val_loss": best_val_loss,
        }

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_ckpt["best_val_loss"] = best_val_loss
            torch.save(best_ckpt, os.path.join(args.output_dir, "best.pt"))

        if reference_val_loss is None:
            reference_val_loss = max(val_metrics["total"], 1e-12)
        loss_ratio = val_metrics["total"] / reference_val_loss
        target_lr = max(min_lr, lr * loss_ratio)
        # Keep LR monotonically non-increasing even if loss temporarily rises.
        current_lr = min(current_lr, target_lr)
        set_optimizer_lr(optimizer, current_lr)

        print(
            f"[Epoch {epoch}] "
            f"lr={current_lr:.8f} "
            f"train_loss={train_total:.6f} train_mse={train_mse:.6f} "
            f"val_loss={val_metrics['total']:.6f} val_mse={val_metrics['mse']:.6f} "
            f"val_topk={val_metrics['topk_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
