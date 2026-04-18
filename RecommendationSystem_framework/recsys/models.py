from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence

from .retrieval import VectorIndex
from .types import Candidate, Item, RankedItem, User

"""模型接口与基础实现，支持召回、粗排、精排可插拔组合。"""


class RecallModel(ABC):
    """召回模型接口。"""

    name: str

    @abstractmethod
    def recall(self, user: User, items: Sequence[Item], max_candidates: int) -> List[Candidate]:
        """根据用户与物料池返回候选列表。"""
        raise NotImplementedError


class CoarseRankModel(ABC):
    """粗排模型接口。"""

    name: str

    @abstractmethod
    def score(self, user: User, item: Item, recall_score: float) -> float:
        """粗排阶段打分。"""
        raise NotImplementedError


class RerankModel(ABC):
    """精排模型接口。"""

    name: str

    @abstractmethod
    def score(self, user: User, item: Item, coarse_score: float) -> float:
        """精排阶段打分。"""
        raise NotImplementedError


class PopularityRecall(RecallModel):
    """基于热度的召回。"""

    def __init__(self, name: str = "popularity_recall", top_n: int = 200):
        self.name = name
        self.top_n = top_n

    def recall(self, user: User, items: Sequence[Item], max_candidates: int) -> List[Candidate]:
        limit = min(self.top_n, max_candidates)
        scored = sorted(
            items,
            key=lambda item: item.features.get("popularity", 0.0),
            reverse=True,
        )
        return [
            Candidate(item=item, score=item.features.get("popularity", 0.0), sources=[self.name])
            for item in scored[:limit]
        ]


class KeywordRecall(RecallModel):
    """关键词召回，根据用户兴趣标签召回相似物料。"""

    def __init__(self, name: str = "keyword_recall", top_n: int = 200):
        self.name = name
        self.top_n = top_n

    def recall(self, user: User, items: Sequence[Item], max_candidates: int) -> List[Candidate]:
        limit = min(self.top_n, max_candidates)
        keywords = set(user.keywords)
        scored_items: List[Candidate] = []
        for item in items:
            match_count = len(keywords.intersection(item.tags))
            if match_count > 0:
                scored_items.append(
                    Candidate(item=item, score=float(match_count), sources=[self.name])
                )
        scored_items.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored_items[:limit]


class VectorRecall(RecallModel):
    """向量检索召回，依赖 VectorIndex。"""

    def __init__(self, index: VectorIndex, name: str = "vector_recall", top_n: int = 200):
        self.name = name
        self.index = index
        self.top_n = top_n

    def recall(self, user: User, items: Sequence[Item], max_candidates: int) -> List[Candidate]:
        limit = min(self.top_n, max_candidates)
        scored = self.index.query(user, limit)
        return [
            Candidate(item=item, score=score, sources=[self.name])
            for item, score in scored
        ]


class LinearCoarseRank(CoarseRankModel):
    """线性粗排，根据物料特征与召回分加权。"""

    def __init__(
        self,
        name: str = "linear_coarse_rank",
        feature_weights: Dict[str, float] | None = None,
        recall_weight: float = 0.3,
    ):
        self.name = name
        self.feature_weights = feature_weights or {"quality": 0.5, "popularity": 0.2, "newness": 0.3}
        self.recall_weight = recall_weight

    def score(self, user: User, item: Item, recall_score: float) -> float:
        feature_score = sum(
            weight * item.features.get(feature, 0.0) for feature, weight in self.feature_weights.items()
        )
        return feature_score + self.recall_weight * recall_score


class LinearRerank(RerankModel):
    """线性精排，基于更细粒度特征与粗排分。"""

    def __init__(
        self,
        name: str = "linear_rerank",
        feature_weights: Dict[str, float] | None = None,
        coarse_weight: float = 0.4,
    ):
        self.name = name
        self.feature_weights = feature_weights or {"quality": 0.6, "freshness": 0.2, "ctr": 0.2}
        self.coarse_weight = coarse_weight

    def score(self, user: User, item: Item, coarse_score: float) -> float:
        feature_score = sum(
            weight * item.features.get(feature, 0.0) for feature, weight in self.feature_weights.items()
        )
        return feature_score + self.coarse_weight * coarse_score


class TagBoostRerank(RerankModel):
    """对用户兴趣标签匹配的物料做额外提升。"""

    def __init__(self, name: str = "tag_boost_rerank", boost: float = 0.2, coarse_weight: float = 0.5):
        self.name = name
        self.boost = boost
        self.coarse_weight = coarse_weight

    def score(self, user: User, item: Item, coarse_score: float) -> float:
        match_count = len(set(user.keywords).intersection(item.tags))
        tag_boost = self.boost * match_count
        return coarse_score * self.coarse_weight + tag_boost


def merge_candidates(candidate_lists: Iterable[Sequence[Candidate]]) -> List[Candidate]:
    """将多个召回结果合并为去重候选列表。"""
    merged: Dict[str, Candidate] = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            item_id = candidate.item.item_id
            if item_id not in merged:
                merged[item_id] = Candidate(
                    item=candidate.item,
                    score=candidate.score,
                    sources=list(candidate.sources),
                )
            else:
                existing = merged[item_id]
                if candidate.score > existing.score:
                    existing.score = candidate.score
                for source in candidate.sources:
                    if source not in existing.sources:
                        existing.sources.append(source)
    return list(merged.values())


def rank_candidates(
    candidates: Sequence[Candidate],
    model: CoarseRankModel | RerankModel,
    user: User,
) -> List[RankedItem]:
    """将候选通过指定排序模型打分并排序。"""
    ranked = [
        RankedItem(item=candidate.item, score=model.score(user, candidate.item, candidate.score))
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked
