import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from RL_DIN_reward.data_utils import DINDataset, build_sid_vocab, din_collate_fn, load_sid_rows
from RL_DIN_reward.din_model import DIN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: DIN, dataloader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            history_ids = batch["history_ids"].to(device)
            history_mask = batch["history_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(history_ids, history_mask, target_ids)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == labels).sum().item())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Offline train DIN reward model for MiniOneRec RL.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train CSV with history_item_sid and item_sid.")
    parser.add_argument("--valid_file", type=str, default="", help="Optional path to valid CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for checkpoint and metadata.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--attn_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--num_neg", type=int, default=2, help="Negative sampling number per positive sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_rows = load_sid_rows(args.train_file)
    if len(train_rows) == 0:
        raise ValueError(f"No training rows loaded from {args.train_file}")

    sid2id = build_sid_vocab(train_rows)
    train_dataset = DINDataset(
        rows=train_rows,
        sid2id=sid2id,
        max_seq_len=args.max_seq_len,
        num_neg=args.num_neg,
        seed=args.seed,
        with_negative_sampling=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=din_collate_fn,
    )

    valid_loader = None
    if args.valid_file:
        valid_rows = load_sid_rows(args.valid_file)
        if len(valid_rows) > 0:
            valid_dataset = DINDataset(
                rows=valid_rows,
                sid2id=sid2id,
                max_seq_len=args.max_seq_len,
                num_neg=args.num_neg,
                seed=args.seed + 1,
                with_negative_sampling=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=din_collate_fn,
            )

    device = torch.device(args.device)
    model = DIN(
        num_items=len(sid2id),
        embed_dim=args.embed_dim,
        attn_hidden_dim=args.attn_hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_valid_loss = float("inf")
    best_path = os.path.join(args.output_dir, "din_reward_best.pt")

    print(f"Train rows: {len(train_rows)}")
    print(f"Train examples (pos+neg): {len(train_dataset)}")
    print(f"SID vocab size: {len(sid2id)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0

        for batch in train_loader:
            history_ids = batch["history_ids"].to(device)
            history_mask = batch["history_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(history_ids, history_mask, target_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_total += batch_size

        train_loss = running_loss / max(running_total, 1)
        log_line = f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.6f}"

        if valid_loader is not None:
            valid_metrics = evaluate(model, valid_loader, device, criterion)
            valid_loss = valid_metrics["loss"]
            valid_acc = valid_metrics["accuracy"]
            log_line += f", valid_loss={valid_loss:.6f}, valid_acc={valid_acc:.4f}"

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "sid2id": sid2id,
                        "config": {
                            "num_items": len(sid2id),
                            "embed_dim": args.embed_dim,
                            "attn_hidden_dim": args.attn_hidden_dim,
                            "mlp_hidden_dim": args.mlp_hidden_dim,
                            "max_seq_len": args.max_seq_len,
                        },
                    },
                    best_path,
                )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "sid2id": sid2id,
                    "config": {
                        "num_items": len(sid2id),
                        "embed_dim": args.embed_dim,
                        "attn_hidden_dim": args.attn_hidden_dim,
                        "mlp_hidden_dim": args.mlp_hidden_dim,
                        "max_seq_len": args.max_seq_len,
                    },
                },
                best_path,
            )

        print(log_line)

    with open(os.path.join(args.output_dir, "din_train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"Saved checkpoint: {best_path}")


if __name__ == "__main__":
    main()
