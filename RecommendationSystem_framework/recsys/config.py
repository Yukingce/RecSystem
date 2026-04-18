from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping

from .models import (
    KeywordRecall,
    LinearCoarseRank,
    LinearRerank,
    PopularityRecall,
    TagBoostRerank,
    VectorRecall,
)
from .pipeline import CoarseRankStage, RecallStage, RecommendationPipeline, RerankStage

"""配置与注册表模块，用于构建推荐流水线。"""


@dataclass
class ModelRegistry:
    """模型工厂注册表。"""

    recall_factories: Dict[str, Callable[..., object]] = field(default_factory=dict)
    coarse_factories: Dict[str, Callable[..., object]] = field(default_factory=dict)
    rerank_factories: Dict[str, Callable[..., object]] = field(default_factory=dict)

    def register_recall(self, name: str, factory: Callable[..., object]) -> None:
        """注册召回模型工厂。"""
        self.recall_factories[name] = factory

    def register_coarse(self, name: str, factory: Callable[..., object]) -> None:
        """注册粗排模型工厂。"""
        self.coarse_factories[name] = factory

    def register_rerank(self, name: str, factory: Callable[..., object]) -> None:
        """注册精排模型工厂。"""
        self.rerank_factories[name] = factory

    def create_recall(self, name: str, **params) -> object:
        """创建召回模型实例。"""
        if name not in self.recall_factories:
            raise ValueError(f"未知召回模型: {name}")
        return self.recall_factories[name](**params)

    def create_coarse(self, name: str, **params) -> object:
        """创建粗排模型实例。"""
        if name not in self.coarse_factories:
            raise ValueError(f"未知粗排模型: {name}")
        return self.coarse_factories[name](**params)

    def create_rerank(self, name: str, **params) -> object:
        """创建精排模型实例。"""
        if name not in self.rerank_factories:
            raise ValueError(f"未知精排模型: {name}")
        return self.rerank_factories[name](**params)


def register_default_models(registry: ModelRegistry) -> None:
    """注册默认模型工厂。"""
    registry.register_recall("popularity_recall", PopularityRecall)
    registry.register_recall("keyword_recall", KeywordRecall)
    registry.register_recall("vector_recall", VectorRecall)
    registry.register_coarse("linear_coarse_rank", LinearCoarseRank)
    registry.register_rerank("linear_rerank", LinearRerank)
    registry.register_rerank("tag_boost_rerank", TagBoostRerank)


@dataclass(frozen=True)
class ModelConfig:
    """单模型配置。"""

    name: str
    params: Dict[str, object]


@dataclass(frozen=True)
class PipelineConfig:
    """流水线配置。"""

    recall: List[ModelConfig]
    coarse: ModelConfig
    rerank: ModelConfig
    recall_max: int = 500
    coarse_top_k: int = 200
    rerank_top_k: int = 50

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PipelineConfig":
        """从字典构建配置。"""
        recall_models = [
            ModelConfig(name=item["name"], params=dict(item.get("params", {})))
            for item in payload.get("recall", [])
        ]
        coarse_payload = payload.get("coarse", {})
        rerank_payload = payload.get("rerank", {})
        return cls(
            recall=recall_models,
            coarse=ModelConfig(
                name=coarse_payload.get("name", "linear_coarse_rank"),
                params=dict(coarse_payload.get("params", {})),
            ),
            rerank=ModelConfig(
                name=rerank_payload.get("name", "linear_rerank"),
                params=dict(rerank_payload.get("params", {})),
            ),
            recall_max=int(payload.get("recall_max", 500)),
            coarse_top_k=int(payload.get("coarse_top_k", 200)),
            rerank_top_k=int(payload.get("rerank_top_k", 50)),
        )


def build_pipeline_from_config(
    config: PipelineConfig,
    registry: ModelRegistry,
    extra_params: Dict[str, Dict[str, object]] | None = None,
    feature_store=None,
    pre_filters=None,
    post_processors=None,
) -> RecommendationPipeline:
    """根据配置与注册表构建流水线。"""
    extra_params = extra_params or {}
    recall_models = [
        registry.create_recall(model.name, **{**model.params, **extra_params.get(model.name, {})})
        for model in config.recall
    ]
    coarse_model = registry.create_coarse(
        config.coarse.name,
        **{**config.coarse.params, **extra_params.get(config.coarse.name, {})},
    )
    rerank_model = registry.create_rerank(
        config.rerank.name,
        **{**config.rerank.params, **extra_params.get(config.rerank.name, {})},
    )
    return RecommendationPipeline(
        recall_stage=RecallStage(models=recall_models, max_candidates=config.recall_max),
        coarse_rank_stage=CoarseRankStage(model=coarse_model, top_k=config.coarse_top_k),
        rerank_stage=RerankStage(model=rerank_model, top_k=config.rerank_top_k),
        feature_store=feature_store,
        pre_filters=pre_filters,
        post_processors=post_processors,
    )
