from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(queries)
        k = self.linear_k(keys)
        v = self.linear_v(keys)

        split_size = self.hidden_size // self.num_heads
        q_ = torch.cat(torch.split(q, split_size, dim=2), dim=0)
        k_ = torch.cat(torch.split(k, split_size, dim=2), dim=0)
        v_ = torch.cat(torch.split(v, split_size, dim=2), dim=0)

        attn_logits = torch.bmm(q_, k_.transpose(1, 2)) / (self.hidden_size ** 0.5)

        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)
        key_mask = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        key_paddings = torch.ones_like(attn_logits) * (-2**32 + 1)
        attn_logits = torch.where(torch.eq(key_mask, 0), key_paddings, attn_logits)

        diag_vals = torch.ones_like(attn_logits[0, :, :])
        tril = torch.tril(diag_vals)
        causality_mask = tril.unsqueeze(0).repeat(attn_logits.shape[0], 1, 1)
        causality_paddings = torch.ones_like(causality_mask) * (-2**32 + 1)
        attn_logits = torch.where(torch.eq(causality_mask, 0), causality_paddings, attn_logits)

        attn = self.softmax(attn_logits)

        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])
        attn = attn * query_mask
        attn = self.dropout(attn)

        output = torch.bmm(attn, v_)
        output = torch.cat(torch.split(output, output.shape[0] // self.num_heads, dim=0), dim=2)
        return output + queries


class SASRec(nn.Module):
    """
    与 rl.py 中 reward_type == "sasrec" 兼容的 SASRec 定义。
    - padding item id 固定为 `item_num`
    - forward_eval 输出 shape: [B, item_num]
    """

    def __init__(
        self,
        hidden_size: int,
        item_num: int,
        state_size: int,
        dropout: float,
        device: torch.device,
        num_heads: int = 1,
    ):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.device = device

        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        inputs_emb = self.item_embeddings(states)
        pos_idx = torch.arange(self.state_size, device=states.device)
        inputs_emb = inputs_emb + self.positional_embeddings(pos_idx)
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        seq = seq * mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out = ff_out * mask
        ff_out = self.ln_3(ff_out)
        return ff_out

    def forward(self, states: torch.Tensor, len_states: torch.Tensor) -> torch.Tensor:
        ff_out = self._encode(states)
        indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        logits = self.s_fc(state_hidden).squeeze(1)
        return logits

    def forward_eval(self, states: torch.Tensor, len_states: torch.Tensor) -> torch.Tensor:
        return self.forward(states, len_states)
