import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from state_merge.mixer import DynamicStateMixer, count_layers_heads_from_state
from state_merge.tiny_rwkv_merger import TinyRWKVMergerConfig, TinyRWKVStateMerger
from state_utils import _is_wkv_path


MODEL_KINDS = ("tiny_rwkv", "conv")


def parse_args():
    parser = argparse.ArgumentParser(description="Train state merger (tiny RWKV or conv).")
    parser.add_argument(
        "--model",
        type=str,
        default="tiny_rwkv",
        choices=MODEL_KINDS,
        help="Merger architecture: 'tiny_rwkv' (default) or legacy 'conv'.",
    )
    parser.add_argument("--dataset-dir", type=str, default="./data")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./state_merge/checkpoints")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=0.2)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--rec-tol", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Tiny RWKV hyperparameters (only used when --model=tiny_rwkv).
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ffn", type=int, default=64)
    parser.add_argument("--max-layers", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--no-low-rank-mask",
        action="store_true",
        help="Disable low-rank per-element mask (use only per-head scalar gate).",
    )
    return parser.parse_args()


def build_merger(args, num_layers: int, heads: int, h: int, w: int) -> torch.nn.Module:
    """Construct the merger model requested by `--model`."""
    if args.model == "conv":
        return DynamicStateMixer(num_layers, heads, h, w)
    if args.model == "tiny_rwkv":
        config = TinyRWKVMergerConfig(
            d_model=args.d_model,
            d_ffn=args.d_ffn,
            max_layers=max(args.max_layers, num_layers),
            use_low_rank_mask=not args.no_low_rank_mask,
            dropout=args.dropout,
        )
        return TinyRWKVStateMerger(num_layers, heads, h, w, config=config)
    raise ValueError(f"Unknown --model={args.model}")


def _split_dirs(dataset_dir: str, split_name: str):
    split_dir = os.path.join(dataset_dir, split_name)
    return os.path.join(split_dir, "one_state"), os.path.join(split_dir, "two_state")


class PairStateDataset(Dataset):
    def __init__(self, one_state_dir: str, two_state_dir: str):
        self.two_state_files = sorted(
            os.path.join(two_state_dir, name)
            for name in os.listdir(two_state_dir)
            if name.endswith(".pt")
        )
        self.one_state_cache = {}
        self.samples = []
        self.num_groups = 0

        # Load all one_state tensors into CPU memory first.
        one_files = sorted(
            os.path.join(one_state_dir, name)
            for name in os.listdir(one_state_dir)
            if name.endswith(".pt")
        )
        for path in tqdm(one_files, desc=f"RAM preload one_state ({os.path.basename(one_state_dir)})"):
            item = torch.load(path, map_location="cpu")
            self.one_state_cache[int(item["index"])] = extract_wkv_units(item["state"]).to(
                dtype=torch.float16, device="cpu"
            )

        # Build all training pairs in RAM, so no SSD I/O during training.
        for pair_path in tqdm(self.two_state_files, desc=f"RAM preload two_state ({os.path.basename(two_state_dir)})"):
            pair_item = torch.load(pair_path, map_location="cpu")
            left_index = int(pair_item["left_index"])
            right_index = int(pair_item["right_index"])
            left_wkv = self.one_state_cache[left_index]
            right_wkv = self.one_state_cache[right_index]
            target_wkv = extract_wkv_units(pair_item["state"]).to(dtype=torch.float16, device="cpu")
            if left_wkv.shape != right_wkv.shape or left_wkv.shape != target_wkv.shape:
                raise ValueError(
                    f"WKV shape mismatch: left={tuple(left_wkv.shape)} right={tuple(right_wkv.shape)} target={tuple(target_wkv.shape)}"
                )
            if self.num_groups == 0:
                self.num_groups = left_wkv.shape[0]
            elif self.num_groups != left_wkv.shape[0]:
                raise ValueError(f"Inconsistent group count: {self.num_groups} vs {left_wkv.shape[0]}")

            # x: [2N,H,W] in (A0,B0,A1,B1,...) channel order
            x = torch.stack([left_wkv, right_wkv], dim=1).reshape(
                2 * left_wkv.shape[0], left_wkv.shape[1], left_wkv.shape[2]
            )
            self.samples.append(
                {
                    "x": x.to(dtype=torch.float16, device="cpu"),
                    "target": target_wkv.to(dtype=torch.float16, device="cpu"),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_float32(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([item["x"] for item in batch], dim=0).to(dtype=torch.float32)
    target = torch.stack([item["target"] for item in batch], dim=0).to(dtype=torch.float32)
    return {"x": x, "target": target}


def run_epoch(model, loader, device, optimizer=None, grad_clip: float = 1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_recall = 0.0
    total_count = 0

    pbar = tqdm(loader, desc="train" if is_train else "eval")
    for batch in pbar:
        x = batch["x"].to(device=device)
        target = batch["target"].to(device=device)

        out = model(x)
        pred = out["mixed"]
        loss = tensor_loss(pred, target, model.num_groups)
        rec = tensor_recall(pred, target)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        total_recall += float(rec)
        total_count += 1
        pbar.set_postfix(
            loss=f"{(total_loss / max(total_count, 1)):.8f}",
            recall=f"{(total_recall / max(total_count, 1)):.4f}",
        )

    return {
        "loss": total_loss / max(total_count, 1),
        "recall": total_recall / max(total_count, 1),
    }


def _to_headwise_state_dict(mixer_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Unify checkpoint format with HeadwiseStateMixer expected keys.
    if any(k.startswith("_mixer.") for k in mixer_state_dict.keys()):
        return mixer_state_dict
    return {f"_mixer.{k}": v for k, v in mixer_state_dict.items()}


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_one, train_two = _split_dirs(args.dataset_dir, args.train_split)
    val_one, val_two = _split_dirs(args.dataset_dir, args.val_split)
    train_ds = PairStateDataset(train_one, train_two)
    val_ds = PairStateDataset(val_one, val_two)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_float32,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_float32,
    )

    if len(train_ds) == 0:
        raise ValueError("Empty training dataset.")
    one_files = sorted(
        os.path.join(train_one, name) for name in os.listdir(train_one) if name.endswith(".pt")
    )
    if not one_files:
        raise ValueError(f"No one_state .pt files in {train_one}")
    first_item = torch.load(one_files[0], map_location="cpu")
    num_layers, heads, h_meta, w_meta = count_layers_heads_from_state(first_item["state"])
    h = int(train_ds[0]["x"].shape[-2])
    w = int(train_ds[0]["x"].shape[-1])
    if h_meta != h or w_meta != w:
        raise ValueError(
            f"Spatial size mismatch: state tree {(h_meta, w_meta)} vs dataset x {(h, w)}"
        )
    if train_ds.num_groups != num_layers * heads:
        raise ValueError(
            f"WKV unit count mismatch: dataset num_groups={train_ds.num_groups} "
            f"vs num_layers*heads={num_layers * heads}"
        )
    model = build_merger(args, num_layers, heads, h, w)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[init] model={args.model} layers={num_layers} heads_per_layer={heads} "
        f"total_units={train_ds.num_groups} trainable_params={num_params:,}"
    )
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model, train_loader, args.device, optimizer=optimizer, grad_clip=args.grad_clip
        )
        val_stats = run_epoch(model, val_loader, args.device, optimizer=None, grad_clip=args.grad_clip)
        scheduler.step()

        train_loss = train_stats["loss"]
        val_loss = val_stats["loss"]
        train_recall = train_stats["recall"]
        val_recall = val_stats["recall"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_recall": train_recall,
                "val_recall": val_recall,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[epoch {epoch}] train_loss={train_loss:.8f} val_loss={val_loss:.8f} "
            f"train_recall={train_recall:.4f} val_recall={val_recall:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        ckpt = {
            "epoch": epoch,
            "model_kind": args.model,
            "model_state_dict": _to_headwise_state_dict(model.state_dict()),
            # Keep raw mixer keys for optional direct mixer-only loading.
            "mixer_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_shape": {
                "num_layers": num_layers,
                "heads_per_layer": heads,
                "height": h,
                "width": w,
            },
            "num_parameters": num_params,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_recall": train_recall,
            "val_recall": val_recall,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def extract_wkv_units(state: Any) -> torch.Tensor:
    chunks: List[torch.Tensor] = []

    def walk(x: Any, path: Tuple[int, ...]):
        if torch.is_tensor(x):
            if _is_wkv_path(path):
                if x.ndim < 2:
                    raise ValueError(f"WKV tensor must be >=2D, got shape={tuple(x.shape)}")
                n = int(math.prod(x.shape[:-2]))
                h, w = x.shape[-2], x.shape[-1]
                chunks.append(x.reshape(n, h, w).float())
            return
        if isinstance(x, (list, tuple)):
            for i, item in enumerate(x):
                walk(item, path + (i,))
            return
        raise TypeError(f"Unsupported state type: {type(x)}")

    walk(state, ())
    if not chunks:
        raise ValueError("No WKV tensor leaves found in state.")
    return torch.cat(chunks, dim=0)


def tensor_loss(pred: torch.Tensor, target: torch.Tensor, num_groups: int) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    pred_flat = pred.reshape(pred.size(0) * num_groups, -1)
    tgt_flat = target.reshape(target.size(0) * num_groups, -1)
    cos = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=1).mean()
    return mse + 0.2 * l1 + 0.2 * cos


def tensor_recall(pred: torch.Tensor, target: torch.Tensor, tol: float = 1e-3) -> float:
    err = (pred - target).abs().mean(dim=(-1, -2))
    return float((err <= tol).float().mean().item())


if __name__ == "__main__":
    main()
