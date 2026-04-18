import torch
import torch.nn as nn


class DIN(nn.Module):
    def __init__(self, num_items: int, embed_dim: int = 64, attn_hidden_dim: int = 128, mlp_hidden_dim: int = 128):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.pad_id = 0

        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim, padding_idx=self.pad_id)

        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1),
        )

        self.prediction_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        if self.item_embedding.padding_idx is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.item_embedding.padding_idx].fill_(0.0)

        for module in list(self.attention_mlp) + list(self.prediction_mlp):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, history_ids: torch.Tensor, history_mask: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_ids: LongTensor, shape [B, T]
            history_mask: BoolTensor, shape [B, T], True for valid positions
            target_ids: LongTensor, shape [B]
        Returns:
            logits: FloatTensor, shape [B]
        """
        history_emb = self.item_embedding(history_ids)  # [B, T, D]
        target_emb = self.item_embedding(target_ids)  # [B, D]
        target_expand = target_emb.unsqueeze(1).expand_as(history_emb)  # [B, T, D]

        attn_input = torch.cat(
            [history_emb, target_expand, history_emb - target_expand, history_emb * target_expand],
            dim=-1,
        )  # [B, T, 4D]

        attn_scores = self.attention_mlp(attn_input).squeeze(-1)  # [B, T]
        attn_scores = attn_scores.masked_fill(~history_mask, float("-inf"))

        all_invalid = history_mask.sum(dim=1) == 0
        if all_invalid.any():
            attn_scores[all_invalid] = 0.0

        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T]
        attn_weights = attn_weights * history_mask.float()
        norm = attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attn_weights = attn_weights / norm

        user_interest = torch.sum(attn_weights.unsqueeze(-1) * history_emb, dim=1)  # [B, D]

        pred_input = torch.cat(
            [user_interest, target_emb, user_interest - target_emb, user_interest * target_emb],
            dim=-1,
        )  # [B, 4D]
        logits = self.prediction_mlp(pred_input).squeeze(-1)  # [B]
        return logits
