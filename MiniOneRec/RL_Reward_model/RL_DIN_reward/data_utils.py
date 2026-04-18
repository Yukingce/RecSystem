import ast
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def parse_history_sids(history_value) -> List[str]:
    if isinstance(history_value, list):
        return [str(x).strip() for x in history_value if str(x).strip()]
    if not isinstance(history_value, str):
        return []
    value = history_value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (SyntaxError, ValueError):
        pass
    if "::" in value:
        return [x.strip() for x in value.split("::") if x.strip()]
    return [value]


def parse_target_sid(target_value) -> str:
    return str(target_value).strip()


def load_sid_rows(csv_path: str) -> List[Tuple[List[str], str]]:
    df = pd.read_csv(csv_path)
    rows: List[Tuple[List[str], str]] = []
    for _, row in df.iterrows():
        history = parse_history_sids(row.get("history_item_sid", ""))
        target = parse_target_sid(row.get("item_sid", ""))
        if target:
            rows.append((history, target))
    return rows


def build_sid_vocab(rows: Sequence[Tuple[List[str], str]]) -> Dict[str, int]:
    sid2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for history, target in rows:
        for sid in history:
            if sid not in sid2id:
                sid2id[sid] = len(sid2id)
        if target not in sid2id:
            sid2id[target] = len(sid2id)
    return sid2id


def encode_history(history_sids: Sequence[str], sid2id: Dict[str, int], max_seq_len: int) -> List[int]:
    encoded = [sid2id.get(sid, UNK_ID) for sid in history_sids][-max_seq_len:]
    return encoded


@dataclass
class DINExample:
    history_ids: List[int]
    target_id: int
    label: float


class DINDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[List[str], str]],
        sid2id: Dict[str, int],
        max_seq_len: int = 50,
        num_neg: int = 2,
        seed: int = 42,
        with_negative_sampling: bool = True,
    ):
        self.sid2id = sid2id
        self.max_seq_len = max_seq_len
        self.num_neg = max(0, int(num_neg))
        self.with_negative_sampling = with_negative_sampling
        self.rng = random.Random(seed)
        self.valid_item_ids = list(range(2, len(sid2id)))
        self.examples: List[DINExample] = []
        self._build_examples(rows)

    def _sample_negative(self, avoid_ids: set) -> int:
        if not self.valid_item_ids:
            return UNK_ID
        neg_id = self.rng.choice(self.valid_item_ids)
        max_retry = 20
        retry = 0
        while neg_id in avoid_ids and retry < max_retry:
            neg_id = self.rng.choice(self.valid_item_ids)
            retry += 1
        return neg_id

    def _build_examples(self, rows: Sequence[Tuple[List[str], str]]):
        for history_sids, target_sid in rows:
            history_ids = encode_history(history_sids, self.sid2id, self.max_seq_len)
            target_id = self.sid2id.get(target_sid, UNK_ID)
            self.examples.append(DINExample(history_ids=history_ids, target_id=target_id, label=1.0))

            if not self.with_negative_sampling or self.num_neg <= 0:
                continue

            avoid_ids = set(history_ids)
            avoid_ids.add(target_id)
            for _ in range(self.num_neg):
                neg_id = self._sample_negative(avoid_ids)
                self.examples.append(DINExample(history_ids=history_ids, target_id=neg_id, label=0.0))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DINExample:
        return self.examples[idx]


def din_collate_fn(batch: Sequence[DINExample]):
    batch_size = len(batch)
    max_len = max((len(x.history_ids) for x in batch), default=1)
    history_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    history_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    target_ids = torch.zeros((batch_size,), dtype=torch.long)
    labels = torch.zeros((batch_size,), dtype=torch.float32)

    for i, example in enumerate(batch):
        seq = example.history_ids
        if seq:
            history_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            history_mask[i, : len(seq)] = True
        target_ids[i] = int(example.target_id)
        labels[i] = float(example.label)

    return {
        "history_ids": history_ids,
        "history_mask": history_mask,
        "target_ids": target_ids,
        "labels": labels,
    }
