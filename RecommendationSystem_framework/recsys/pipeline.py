from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .features import FeatureStore
from .models import CoarseRankModel, RerankModel, RecallModel, merge_candidates, rank_candidates
from .policy import ItemFilter, PostProcessor
from .types import Candidate, Item, RankedItem, User

"""推荐流水线，负责组织召回、粗排、精排与特征补全。"""


@dataclass
class RecallStage:
    """召回阶段，多个召回模型并行输出候选。"""

    models: Sequence[RecallModel]
    max_candidates: int = 500

    def run(self, user: User, items: Sequence[Item]) -> List[Candidate]:
        candidate_lists = [model.recall(user, items, self.max_candidates) for model in self.models]
        merged = merge_candidates(candidate_lists)
        merged.sort(key=lambda candidate: candidate.score, reverse=True)
        return merged[: self.max_candidates]


@dataclass
class CoarseRankStage:
    """粗排阶段，过滤掉大部分候选。"""

    model: CoarseRankModel
    top_k: int = 200

    def run(self, user: User, candidates: Sequence[Candidate]) -> List[RankedItem]:
        ranked = rank_candidates(candidates, self.model, user)
        return ranked[: self.top_k]


@dataclass
class RerankStage:
    """精排阶段，输出最终展示列表。"""

    model: RerankModel
    top_k: int = 50

    def run(self, user: User, ranked_items: Sequence[RankedItem]) -> List[RankedItem]:
        candidates = [Candidate(item=item.item, score=item.score, sources=[]) for item in ranked_items]
        reranked = rank_candidates(candidates, self.model, user)
        return reranked[: self.top_k]


@dataclass
class RecommendationPipeline:
    """推荐主链路，可选接入特征补全。"""

    recall_stage: RecallStage
    coarse_rank_stage: CoarseRankStage
    rerank_stage: RerankStage
    feature_store: FeatureStore | None = None
    pre_filters: Sequence[ItemFilter] | None = None
    post_processors: Sequence[PostProcessor] | None = None

    def recommend(self, user: User, items: Sequence[Item]) -> List[RankedItem]:
        if self.feature_store:
            user = self.feature_store.enrich_user(user)
            items = self.feature_store.enrich_items(items)
        if self.pre_filters:
            for item_filter in self.pre_filters:
                items = item_filter.apply(user, items)
        recalled = self.recall_stage.run(user, items)
        coarse_ranked = self.coarse_rank_stage.run(user, recalled)
        ranked = self.rerank_stage.run(user, coarse_ranked)
        if self.post_processors:
            for processor in self.post_processors:
                ranked = processor.apply(user, ranked)
        return ranked


def build_default_pipeline(
    recall_models: Iterable[RecallModel],
    coarse_model: CoarseRankModel,
    rerank_model: RerankModel,
    recall_max: int = 500,
    coarse_top_k: int = 200,
    rerank_top_k: int = 50,
    feature_store: FeatureStore | None = None,
    pre_filters: Sequence[ItemFilter] | None = None,
    post_processors: Sequence[PostProcessor] | None = None,
) -> RecommendationPipeline:
    """构建标准三阶段流水线。"""
    return RecommendationPipeline(
        recall_stage=RecallStage(models=list(recall_models), max_candidates=recall_max),
        coarse_rank_stage=CoarseRankStage(model=coarse_model, top_k=coarse_top_k),
        rerank_stage=RerankStage(model=rerank_model, top_k=rerank_top_k),
        feature_store=feature_store,
        pre_filters=pre_filters,
        post_processors=post_processors,
    )
