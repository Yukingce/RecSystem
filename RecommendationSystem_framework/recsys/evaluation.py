from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Iterable, List, Sequence, Tuple

from .pipeline import RecommendationPipeline
from .types import Interaction, Item, RankedItem, User

"""评估模块，包含离线指标与实验对比。"""


def precision_at_k(recommended: Sequence[RankedItem], ground_truth: set[str], k: int) -> float:
    """计算 Precision@K。"""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hit = sum(1 for item in top_k if item.item.item_id in ground_truth)
    return hit / k


def recall_at_k(recommended: Sequence[RankedItem], ground_truth: set[str], k: int) -> float:
    """计算 Recall@K。"""
    if not ground_truth or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hit = sum(1 for item in top_k if item.item.item_id in ground_truth)
    return hit / len(ground_truth)


def ndcg_at_k(recommended: Sequence[RankedItem], ground_truth: set[str], k: int) -> float:
    """计算 NDCG@K。"""
    if k <= 0 or not ground_truth:
        return 0.0
    top_k = recommended[:k]
    dcg = 0.0
    for index, ranked in enumerate(top_k):
        if ranked.item.item_id in ground_truth:
            dcg += 1.0 / (index + 2) ** 0.5
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / (index + 2) ** 0.5 for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvaluationResult:
    """整体评估结果。"""

    metrics: Dict[str, float]
    per_user: Dict[str, Dict[str, float]]


class OfflineEvaluator:
    """离线评估器，基于交互标签评估推荐效果。"""

    def __init__(self, k: int = 10):
        self.k = k

    def evaluate(
        self,
        pipeline: RecommendationPipeline,
        users: Sequence[User],
        items: Sequence[Item],
        interactions: Iterable[Interaction],
    ) -> EvaluationResult:
        """对全量用户进行离线评估并汇总指标。"""
        labels_by_user: Dict[str, set[str]] = {}
        for interaction in interactions:
            if interaction.label <= 0:
                continue
            labels_by_user.setdefault(interaction.user_id, set()).add(interaction.item_id)

        per_user_metrics: Dict[str, Dict[str, float]] = {}
        precision_sum = recall_sum = ndcg_sum = 0.0
        evaluated_users = 0

        for user in users:
            ground_truth = labels_by_user.get(user.user_id, set())
            if not ground_truth:
                continue
            recommended = pipeline.recommend(user, items)
            precision = precision_at_k(recommended, ground_truth, self.k)
            recall = recall_at_k(recommended, ground_truth, self.k)
            ndcg = ndcg_at_k(recommended, ground_truth, self.k)
            per_user_metrics[user.user_id] = {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
            }
            precision_sum += precision
            recall_sum += recall
            ndcg_sum += ndcg
            evaluated_users += 1

        if evaluated_users == 0:
            metrics = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
        else:
            metrics = {
                "precision": precision_sum / evaluated_users,
                "recall": recall_sum / evaluated_users,
                "ndcg": ndcg_sum / evaluated_users,
            }

        return EvaluationResult(metrics=metrics, per_user=per_user_metrics)


@dataclass(frozen=True)
class ExperimentConfig:
    """实验配置。"""

    name: str
    traffic_split: float = 0.5
    k: int = 10


@dataclass
class ExperimentResult:
    """实验评估结果。"""

    baseline_metrics: Dict[str, float]
    variant_metrics: Dict[str, float]
    uplift: Dict[str, float]


class TrafficAssigner:
    """基于用户 ID 的稳定分流器。"""

    def assign(self, user_id: str, split: float) -> str:
        """按 split 分流，返回 baseline 或 variant。"""
        digest = md5(user_id.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 16**8
        return "variant" if value < split else "baseline"


class Experiment:
    """实验容器，管理分流与对比评估。"""

    def __init__(
        self,
        config: ExperimentConfig,
        baseline: RecommendationPipeline,
        variant: RecommendationPipeline,
        assigner: TrafficAssigner | None = None,
    ) -> None:
        self.config = config
        self.baseline = baseline
        self.variant = variant
        self.assigner = assigner or TrafficAssigner()

    def recommend(self, user: User, items: List[Item]) -> Tuple[str, List[RankedItem]]:
        """根据分流结果输出推荐列表。"""
        bucket = self.assigner.assign(user.user_id, self.config.traffic_split)
        pipeline = self.variant if bucket == "variant" else self.baseline
        return bucket, pipeline.recommend(user, items)

    def evaluate(
        self,
        users: Iterable[User],
        items: List[Item],
        interactions: Iterable[Interaction],
    ) -> ExperimentResult:
        """基于离线评估对比 baseline 和 variant。"""
        evaluator = OfflineEvaluator(k=self.config.k)
        baseline_result = evaluator.evaluate(self.baseline, users, items, interactions)
        variant_result = evaluator.evaluate(self.variant, users, items, interactions)
        uplift = {}
        for metric, base_value in baseline_result.metrics.items():
            var_value = variant_result.metrics.get(metric, 0.0)
            uplift[metric] = var_value - base_value
        return ExperimentResult(
            baseline_metrics=baseline_result.metrics,
            variant_metrics=variant_result.metrics,
            uplift=uplift,
        )
