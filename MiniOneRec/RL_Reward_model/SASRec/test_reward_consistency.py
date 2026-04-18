from __future__ import annotations

import argparse
import ast
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from model import SASRec


@dataclass
class EvalRecord:
    history_ids: List[int]
    target_id: int

"""
    “reward 一致性评测脚本”（专门测 cf_reward 对多 completion 的排序质量）。    
"""

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate cf_reward ranking consistency on multi-completion candidates (test set)."
    )
    parser.add_argument("--test_file", type=str, required=True, help="Test CSV with history_item_sid/item_sid.")
    parser.add_argument("--info_file", type=str, required=True, help="SID mapping file used by RL.")
    parser.add_argument("--cf_path", type=str, required=True, help="SASRec state dict path.")
    parser.add_argument("--output_path", type=str, default="", help="Optional JSON report output path.")

    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_candidates", type=int, default=16, help="Candidates per prompt (1 positive + negatives).")
    parser.add_argument("--eval_topk", type=str, default="1,3,5,10")
    parser.add_argument("--sample_size", type=int, default=-1, help="Use -1 for all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_examples", type=int, default=20)
    return parser.parse_args()


def parse_topk(topk_str: str) -> List[int]:
    return [int(x.strip()) for x in topk_str.split(",") if x.strip()]


def load_sid_mapping(info_file: str) -> Tuple[List[str], Dict[str, int]]:
    with open(info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    item_sids = [line.split("\t")[0].strip() for line in lines]
    sid2id = {sid: idx for idx, sid in enumerate(item_sids)}
    return item_sids, sid2id


def parse_sid_list(value: str) -> List[str]:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def build_eval_records(test_file: str, sid2id: Dict[str, int], max_seq_len: int) -> List[EvalRecord]:
    df = pd.read_csv(test_file)
    required = {"history_item_sid", "item_sid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{test_file} missing columns: {missing}")

    records: List[EvalRecord] = []
    dropped = 0
    for _, row in df.iterrows():
        target_sid = str(row["item_sid"]).strip()
        if target_sid not in sid2id:
            dropped += 1
            continue

        history_sids = parse_sid_list(row["history_item_sid"])
        history_ids = [sid2id[sid] for sid in history_sids if sid in sid2id]
        if len(history_ids) == 0:
            dropped += 1
            continue
        history_ids = history_ids[-max_seq_len:]
        records.append(EvalRecord(history_ids=history_ids, target_id=sid2id[target_sid]))

    if len(records) == 0:
        raise ValueError("No valid evaluation records were built.")
    if dropped > 0:
        print(f"[Evaluator] Dropped {dropped} invalid rows.")
    print(f"[Evaluator] Built {len(records)} valid eval records.")
    return records


def score_candidates(
    model: SASRec,
    history_ids: List[int],
    candidate_ids: List[int],
    item_num: int,
    max_seq_len: int,
    device: torch.device,
) -> List[float]:
    seq = history_ids[-max_seq_len:]
    seq_len = len(seq)
    if seq_len < max_seq_len:
        seq = seq + [item_num] * (max_seq_len - seq_len)

    seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
    len_tensor = torch.tensor([seq_len], dtype=torch.long, device=device)
    cand_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device).view(1, -1)

    with torch.no_grad():
        logits = model.forward_eval(seq_tensor, len_tensor)  # [1, item_num]
        cand_scores = torch.gather(logits, 1, cand_tensor).view(-1).cpu().tolist()
    return cand_scores


def sample_negative_ids(target_id: int, item_num: int, num_neg: int, rng: random.Random) -> List[int]:
    if item_num <= 1:
        return []
    available = item_num - 1
    take = min(num_neg, available)
    negatives = set()
    while len(negatives) < take:
        neg = rng.randint(0, item_num - 1)
        if neg != target_id:
            negatives.add(neg)
    return list(negatives)


def compute_rank_desc(scores: List[float], target_idx: int) -> int:
    order = np.argsort(np.array(scores))[::-1]
    pos = int(np.where(order == target_idx)[0][0])
    return pos + 1


def main():
    args = parse_args()
    set_seed(args.seed)
    topk = parse_topk(args.eval_topk)
    rng = random.Random(args.seed)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    item_sids, sid2id = load_sid_mapping(args.info_file)
    item_num = len(item_sids)

    model = SASRec(
        hidden_size=args.hidden_size,
        item_num=item_num,
        state_size=args.max_seq_len,
        dropout=args.dropout,
        device=device,
        num_heads=args.num_heads,
    ).to(device)
    model.load_state_dict(torch.load(args.cf_path, map_location=device))
    model.eval()

    records = build_eval_records(args.test_file, sid2id, args.max_seq_len)
    if args.sample_size > 0 and args.sample_size < len(records):
        records = rng.sample(records, args.sample_size)
        print(f"[Evaluator] Sampled {len(records)} records for evaluation.")

    num_candidates = max(2, args.num_candidates)
    num_neg = num_candidates - 1

    hit = {k: 0 for k in topk}
    ndcg = {k: 0.0 for k in topk}
    mrr = 0.0
    auc = 0.0
    total = 0
    example_rows = []

    for rec in records:
        neg_ids = sample_negative_ids(rec.target_id, item_num, num_neg, rng)
        candidate_ids = [rec.target_id] + neg_ids
        rng.shuffle(candidate_ids)

        target_idx = candidate_ids.index(rec.target_id)
        cand_scores = score_candidates(
            model=model,
            history_ids=rec.history_ids,
            candidate_ids=candidate_ids,
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            device=device,
        )

        rank = compute_rank_desc(cand_scores, target_idx)
        total += 1
        mrr += 1.0 / rank

        pos_score = cand_scores[target_idx]
        neg_scores = [s for i, s in enumerate(cand_scores) if i != target_idx]
        if len(neg_scores) > 0:
            auc += sum(1.0 if pos_score > ns else 0.5 if pos_score == ns else 0.0 for ns in neg_scores) / len(neg_scores)

        for k in topk:
            if rank <= k:
                hit[k] += 1
                ndcg[k] += 1.0 / np.log2(rank + 1.0)

        if len(example_rows) < args.save_examples:
            order = np.argsort(np.array(cand_scores))[::-1]
            top_pred_id = candidate_ids[int(order[0])]
            example_rows.append(
                {
                    "history_len": len(rec.history_ids),
                    "target_sid": item_sids[rec.target_id],
                    "top1_sid": item_sids[top_pred_id],
                    "target_rank_in_candidates": rank,
                    "target_score": pos_score,
                }
            )

    metrics = {
        "num_records": total,
        "num_candidates_per_prompt": num_candidates,
        "random_hit_at_1_baseline": 1.0 / num_candidates,
        "MRR": mrr / max(total, 1),
        "AUC_pos_vs_neg": auc / max(total, 1),
    }
    for k in topk:
        metrics[f"Hit@{k}"] = hit[k] / max(total, 1)
        metrics[f"NDCG@{k}"] = ndcg[k] / max(total, 1)

    report = {
        "config": {
            "test_file": args.test_file,
            "info_file": args.info_file,
            "cf_path": args.cf_path,
            "max_seq_len": args.max_seq_len,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "num_heads": args.num_heads,
            "device": str(device),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "eval_topk": topk,
        },
        "metrics": metrics,
        "examples": example_rows,
    }

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Evaluator] Saved report to {args.output_path}")


if __name__ == "__main__":
    main()
