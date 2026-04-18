from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceRecord:
    seq: List[int]
    len_seq: int
    target: int


def load_sid_mappings(info_file: str) -> Tuple[List[str], Dict[str, int]]:
    """
    读取 info 文件，构建 SID -> item_id 的映射。
    """
    with open(info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    item_sids = [line.split("\t")[0].strip() for line in lines]
    sid2id = {sid: idx for idx, sid in enumerate(item_sids)}
    return item_sids, sid2id


def _safe_parse_list(raw_value: str) -> List[str]:
    if isinstance(raw_value, list):
        return raw_value
    try:
        value = ast.literal_eval(raw_value)
        if isinstance(value, list):
            return value
        return []
    except (ValueError, SyntaxError):
        return []


def build_training_records(
    csv_file: str,
    sid2id: Dict[str, int],
    max_seq_len: int,
) -> List[SequenceRecord]:
    """
    从 rl.py 使用的 SidDataset 对应 CSV 中构造训练样本:
    - 输入：history_item_sid
    - 正样本：item_sid
    """
    df = pd.read_csv(csv_file)
    required_cols = {"history_item_sid", "item_sid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_file} 缺失字段: {missing}")

    records: List[SequenceRecord] = []
    dropped = 0
    for _, row in df.iterrows():
        history_sids = _safe_parse_list(str(row["history_item_sid"]))
        target_sid = str(row["item_sid"]).strip()
        if target_sid not in sid2id:
            dropped += 1
            continue

        seq_ids = [sid2id[sid] for sid in history_sids if sid in sid2id]
        if len(seq_ids) == 0:
            dropped += 1
            continue

        seq_ids = seq_ids[-max_seq_len:]
        len_seq = len(seq_ids)
        target = sid2id[target_sid]
        records.append(SequenceRecord(seq=seq_ids, len_seq=len_seq, target=target))

    if len(records) == 0:
        raise ValueError(f"从 {csv_file} 未构建出有效样本。")
    if dropped > 0:
        print(f"[DataBuilder] {csv_file} 丢弃 {dropped} 条无效样本。")
    print(f"[DataBuilder] {csv_file} 保留 {len(records)} 条样本。")
    return records


class SASRecTrainDataset(Dataset):
    """
    训练集：每条样本输出 (seq, len_seq, target_item)
    用于 listwise-CE 训练。
    """

    def __init__(self, records: List[SequenceRecord], item_num: int, max_seq_len: int):
        self.records = records
        self.item_num = item_num
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def _pad_seq(self, seq: List[int]) -> List[int]:
        if len(seq) < self.max_seq_len:
            return seq + [self.item_num] * (self.max_seq_len - len(seq))
        return seq[-self.max_seq_len :]

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        seq = self._pad_seq(rec.seq)
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "len_seq": torch.tensor(rec.len_seq, dtype=torch.long),
            "target_item": torch.tensor(rec.target, dtype=torch.long),
        }


class SASRecEvalDataset(Dataset):
    """
    验证/测试集：输出 (seq, len_seq, target_item)，用于全量打分评估 Recall/NDCG。
    """

    def __init__(self, records: List[SequenceRecord], item_num: int, max_seq_len: int):
        self.records = records
        self.item_num = item_num
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def _pad_seq(self, seq: List[int]) -> List[int]:
        if len(seq) < self.max_seq_len:
            return seq + [self.item_num] * (self.max_seq_len - len(seq))
        return seq[-self.max_seq_len :]

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        seq = self._pad_seq(rec.seq)
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "len_seq": torch.tensor(rec.len_seq, dtype=torch.long),
            "target_item": torch.tensor(rec.target, dtype=torch.long),
        }
