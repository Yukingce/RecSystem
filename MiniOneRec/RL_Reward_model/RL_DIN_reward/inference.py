from typing import Dict, List, Sequence

import torch

from RL_DIN_reward.data_utils import UNK_ID, encode_history
from RL_DIN_reward.din_model import DIN


class DINRewardModel:
    def __init__(self, model: DIN, sid2id: Dict[str, int], max_seq_len: int, device: torch.device):
        self.model = model
        self.sid2id = sid2id
        self.max_seq_len = max_seq_len
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device):
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint["config"]
        sid2id = checkpoint["sid2id"]
        model = DIN(
            num_items=int(config["num_items"]),
            embed_dim=int(config["embed_dim"]),
            attn_hidden_dim=int(config["attn_hidden_dim"]),
            mlp_hidden_dim=int(config["mlp_hidden_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model=model, sid2id=sid2id, max_seq_len=int(config.get("max_seq_len", 50)), device=device)

    def _encode_target(self, sid: str) -> int:
        sid = str(sid).strip()
        return self.sid2id.get(sid, UNK_ID)

    def score_batch(self, history_sid_lists: Sequence[Sequence[str]], candidate_sids: Sequence[str]) -> torch.Tensor:
        batch_size = len(candidate_sids)
        if len(history_sid_lists) != batch_size:
            raise ValueError("history_sid_lists and candidate_sids must have the same length.")

        encoded_histories: List[List[int]] = [
            encode_history(history, self.sid2id, self.max_seq_len) for history in history_sid_lists
        ]
        max_len = max((len(x) for x in encoded_histories), default=1)

        history_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        history_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        target_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        for i, history_ids_i in enumerate(encoded_histories):
            if history_ids_i:
                seq_tensor = torch.tensor(history_ids_i, dtype=torch.long, device=self.device)
                history_ids[i, : seq_tensor.numel()] = seq_tensor
                history_mask[i, : seq_tensor.numel()] = True
            target_ids[i] = self._encode_target(candidate_sids[i])

        with torch.no_grad():
            logits = self.model(history_ids, history_mask, target_ids)
            probs = torch.sigmoid(logits)
        return probs
