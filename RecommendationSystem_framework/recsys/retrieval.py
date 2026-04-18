from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import Item, User

"""召回检索模块，包含简单的向量检索实现。"""


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算余弦相似度。"""
    if not left or not right:
        return 0.0
    dot = sum(l * r for l, r in zip(left, right))
    left_norm = sqrt(sum(l * l for l in left))
    right_norm = sqrt(sum(r * r for r in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class VectorIndex:
    """基于特征向量的内存检索索引。"""

    embedding_keys: List[str]
    item_vectors: Dict[str, List[float]]
    item_lookup: Dict[str, Item]

    @classmethod
    def build(cls, items: Iterable[Item], embedding_keys: Sequence[str]) -> "VectorIndex":
        """从物料构建索引。"""
        vectors: Dict[str, List[float]] = {}
        lookup: Dict[str, Item] = {}
        keys = list(embedding_keys)
        for item in items:
            vector = [item.features.get(key, 0.0) for key in keys]
            vectors[item.item_id] = vector
            lookup[item.item_id] = item
        return cls(embedding_keys=keys, item_vectors=vectors, item_lookup=lookup)

    def query(self, user: User, top_n: int) -> List[Tuple[Item, float]]:
        """使用用户向量检索相似物料。"""
        user_vector = [user.features.get(key, 0.0) for key in self.embedding_keys]
        scored = [
            (self.item_lookup[item_id], cosine_similarity(user_vector, vector))
            for item_id, vector in self.item_vectors.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]
