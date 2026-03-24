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
    rmse_plus_mae_wkv_only,
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
    parser.add_argument("--train-index", type=str, default="./encode_compress/data/train_index.json")
    parser.add_argument("--val-index", type=str, default="./encode_compress/data/val_index.json")
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
) -> Dict[str, float]:
    model.eval()
    totals: List[float] = []
    rmses: List[float] = []
    maes: List[float] = []
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_state_to_device(batch, device)
        recon = model.forward_state(batch)
        _, m = rmse_plus_mae_wkv_only(recon, batch)
        totals.append(m["total"])
        rmses.append(m["rmse"])
        maes.append(m["mae"])

    if not totals:
        return {"total": 0.0, "rmse": 0.0, "mae": 0.0}
    n = len(totals)
    return {
        "total": float(sum(totals) / n),
        "rmse": float(sum(rmses) / n),
        "mae": float(sum(maes) / n),
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
        atch_size=batch_size,
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
        "wkv_rotation": "cayley_per_layer_per_head_R1_M_R2T",
        "loss": "rmse_plus_mae_wkv_only",
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
        running_rmse = 0.0
        running_mae = 0.0
        step_count = 0

        for batch in pbar:
            batch = move_state_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            recon = model.forward_state(batch)
            loss, m = rmse_plus_mae_wkv_only(recon, batch)
            loss.backward()
            gn = grad_norm(model)
            optimizer.step()

            running_total += m["total"]
            running_rmse += m["rmse"]
            running_mae += m["mae"]
            step_count += 1
            denom_s = max(step_count, 1)
            pbar.set_postfix(
                loss=f"{running_total / denom_s:.6f}",
                rmse=f"{running_rmse / denom_s:.6f}",
                mae=f"{running_mae / denom_s:.6f}",
                grad_norm=f"{gn:.4f}",
            )

        denom = max(step_count, 1)
        train_loss = running_total / denom
        train_rmse = running_rmse / denom
        train_mae = running_mae / denom
        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "val_loss": val_metrics["total"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
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
            f"train_loss={train_loss:.6f} train_rmse={train_rmse:.6f} train_mae={train_mae:.6f} "
            f"val_loss={val_metrics['total']:.6f} val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
        )


if __name__ == "__main__":
    main()
