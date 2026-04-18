from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_builder import SASRecEvalDataset, SASRecTrainDataset, build_training_records, load_sid_mappings
from model import SASRec


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SASRec for RL reward scoring.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--info_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--eval_topk", type=str, default="1,3,5,10,20")
    parser.add_argument(
        "--list_size",
        type=int,
        default=-1,
        help="Listwise candidates per sample. -1 means full item set.",
    )
    parser.add_argument(
        "--neg_sampling_strategy",
        type=str,
        default="hard",
        choices=["hard", "random", "mixed"],
        help="Negative sampling strategy for sampled listwise training.",
    )
    parser.add_argument(
        "--hard_pool_size",
        type=int,
        default=1024,
        help="Top-k pool size for hard negatives (from model logits).",
    )
    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.8,
        help="When strategy=mixed, fraction of negatives from hard pool.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_topk(topk_str: str) -> List[int]:
    return [int(x.strip()) for x in topk_str.split(",") if x.strip()]


def evaluate(
    model: SASRec,
    dataloader: DataLoader,
    item_num: int,
    topk: List[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    hit = np.zeros(len(topk), dtype=np.float64)
    ndcg = np.zeros(len(topk), dtype=np.float64)
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            len_seq = batch["len_seq"].to(device)
            target = batch["target_item"].to(device)

            logits = model.forward_eval(seq, len_seq)
            rank = logits.argsort(dim=1, descending=True)
            total += target.size(0)

            for i, k in enumerate(topk):
                topk_items = rank[:, :k]
                match = topk_items.eq(target.view(-1, 1))
                hit[i] += match.any(dim=1).float().sum().item()

                rows, cols = torch.where(match)
                if rows.numel() > 0:
                    ndcg[i] += (1.0 / torch.log2(cols.float() + 2.0)).sum().item()

    metrics = {}
    for i, k in enumerate(topk):
        metrics[f"HR@{k}"] = hit[i] / max(total, 1)
        metrics[f"NDCG@{k}"] = ndcg[i] / max(total, 1)
    return metrics


def train_one_epoch(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.CrossEntropyLoss,
    item_num: int,
    list_size: int,
    rng: random.Random,
    neg_sampling_strategy: str,
    hard_pool_size: int,
    hard_ratio: float,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    effective_list_size = list_size if list_size is not None else -1
    use_sampled_list = effective_list_size > 0 and effective_list_size < item_num

    for batch in tqdm(dataloader, desc="Train", leave=False):
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"].to(device)
        target_item = batch["target_item"].to(device)

        logits = model(seq, len_seq)

        if not use_sampled_list:
            loss = ce_loss(logits, target_item.long())
        else:
            batch_size = target_item.size(0)
            target_list_indices = []
            candidate_ids_per_sample = []
            neg_count = max(1, effective_list_size - 1)
            hard_ratio = min(max(hard_ratio, 0.0), 1.0)

            def sample_random_negs(target_id: int, need: int, existing: set[int]) -> List[int]:
                if need <= 0:
                    return []
                sampled = []
                max_need = min(need, item_num - 1 - len(existing))
                while len(sampled) < max_need:
                    neg_id = rng.randint(0, item_num - 1)
                    if neg_id != target_id and neg_id not in existing:
                        sampled.append(neg_id)
                        existing.add(neg_id)
                return sampled

            for i in range(batch_size):
                target_id = int(target_item[i].item())
                chosen_neg_ids: List[int] = []
                chosen_set: set[int] = set()
                max_neg = min(neg_count, item_num - 1)

                hard_need = 0
                if neg_sampling_strategy == "hard":
                    hard_need = max_neg
                elif neg_sampling_strategy == "mixed":
                    hard_need = int(round(max_neg * hard_ratio))
                    hard_need = min(max_neg, max(hard_need, 0))

                if hard_need > 0:
                    pool_k = min(item_num, max(hard_pool_size, hard_need + 8))
                    top_ids = torch.topk(logits[i], k=pool_k, dim=0).indices.detach().cpu().tolist()
                    for neg_id in top_ids:
                        if neg_id != target_id and neg_id not in chosen_set:
                            chosen_neg_ids.append(neg_id)
                            chosen_set.add(neg_id)
                            if len(chosen_neg_ids) >= hard_need:
                                break

                random_need = max_neg - len(chosen_neg_ids)
                if random_need > 0:
                    chosen_neg_ids.extend(sample_random_negs(target_id, random_need, chosen_set))

                candidates = [target_id] + chosen_neg_ids
                rng.shuffle(candidates)
                candidate_ids_per_sample.append(candidates)
                target_list_indices.append(candidates.index(target_id))

            candidate_tensor = torch.tensor(candidate_ids_per_sample, dtype=torch.long, device=device)
            sampled_logits = torch.gather(logits, 1, candidate_tensor)
            label_tensor = torch.tensor(target_list_indices, dtype=torch.long, device=device)
            loss = ce_loss(sampled_logits, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps += 1

    return running_loss / max(steps, 1)


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    topk = parse_topk(args.eval_topk)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    item_sids, sid2id = load_sid_mappings(args.info_file)
    item_num = len(item_sids)
    print(f"item_num={item_num}")

    train_records = build_training_records(args.train_file, sid2id, args.max_seq_len)
    valid_records = build_training_records(args.valid_file, sid2id, args.max_seq_len)
    test_records = build_training_records(args.test_file, sid2id, args.max_seq_len)

    train_dataset = SASRecTrainDataset(train_records, item_num=item_num, max_seq_len=args.max_seq_len)
    valid_dataset = SASRecEvalDataset(valid_records, item_num=item_num, max_seq_len=args.max_seq_len)
    test_dataset = SASRecEvalDataset(test_records, item_num=item_num, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    model = SASRec(
        hidden_size=args.hidden_size,
        item_num=item_num,
        state_size=args.max_seq_len,
        dropout=args.dropout,
        device=device,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    ce_loss = nn.CrossEntropyLoss()
    rng = random.Random(args.seed)

    best_ndcg = -math.inf
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            ce_loss=ce_loss,
            item_num=item_num,
            list_size=args.list_size,
            rng=rng,
            neg_sampling_strategy=args.neg_sampling_strategy,
            hard_pool_size=args.hard_pool_size,
            hard_ratio=args.hard_ratio,
            device=device,
        )
        valid_metrics = evaluate(model, valid_loader, item_num=item_num, topk=topk, device=device)
        test_metrics = evaluate(model, test_loader, item_num=item_num, topk=topk, device=device)
        monitor_key = f"NDCG@{topk[-1]}"
        monitor_value = valid_metrics[monitor_key]

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }
        history.append(epoch_result)
        print(
            f"[Epoch {epoch}] loss={train_loss:.6f} "
            f"valid_{monitor_key}={monitor_value:.6f} "
            f"test_{monitor_key}={test_metrics[monitor_key]:.6f}"
        )

        if monitor_value > best_ndcg:
            best_ndcg = monitor_value
            best_epoch = epoch
            wait = 0
            ckpt_path = os.path.join(args.output_dir, "sasrec_state_dict.pth")
            torch.save(model.state_dict(), ckpt_path)
            save_json(
                os.path.join(args.output_dir, "best_metrics.json"),
                {
                    "best_epoch": best_epoch,
                    "best_valid_metrics": valid_metrics,
                    "corresponding_test_metrics": test_metrics,
                },
            )
            print(f"[Checkpoint] saved to {ckpt_path}")
        else:
            wait += 1
            if wait >= args.early_stop:
                print(f"[EarlyStop] no improvement for {args.early_stop} epochs.")
                break

    save_json(os.path.join(args.output_dir, "train_history.json"), {"history": history})
    save_json(
        os.path.join(args.output_dir, "train_config.json"),
        {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "info_file": args.info_file,
            "output_dir": args.output_dir,
            "max_seq_len": args.max_seq_len,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "num_heads": args.num_heads,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "item_num": item_num,
            "best_epoch": best_epoch,
            "best_valid_ndcg": best_ndcg,
            "eval_topk": topk,
            "loss_type": "listwise_ce",
            "list_size": args.list_size,
            "neg_sampling_strategy": args.neg_sampling_strategy,
            "hard_pool_size": args.hard_pool_size,
            "hard_ratio": args.hard_ratio,
        },
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
