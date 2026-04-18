from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .evaluation import Experiment
from .pipeline import RecommendationPipeline
from .serving_support import LRUCache, OnlineMetrics
from .types import Item, RankedItem, User

"""服务层抽象，支持单请求与批量推荐。"""


@dataclass(frozen=True)
class RecommendRequest:
    """推荐请求。"""

    user: User
    items: Sequence[Item]


@dataclass
class RecommendResponse:
    """推荐响应。"""

    items: List[RankedItem]
    bucket: str | None = None


class ServingEngine:
    """推荐服务引擎，可选接入实验分流。"""

    def __init__(
        self,
        pipeline: RecommendationPipeline | None = None,
        experiment: Experiment | None = None,
        metrics: OnlineMetrics | None = None,
        cache: LRUCache[str, RecommendResponse] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.experiment = experiment
        self.metrics = metrics
        self.cache = cache

    def _build_cache_key(self, request: RecommendRequest) -> str:
        """构建缓存 key。"""
        item_ids = ",".join(item.item_id for item in request.items)
        return f"{request.user.user_id}:{item_ids}"

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """处理单次推荐请求。"""
        if self.cache:
            cached = self.cache.get(self._build_cache_key(request))
            if cached:
                return cached
        if self.experiment:
            bucket, ranked = self.experiment.recommend(request.user, list(request.items))
            if self.metrics:
                self.metrics.log_impressions(ranked)
            response = RecommendResponse(items=ranked, bucket=bucket)
            if self.cache:
                self.cache.set(self._build_cache_key(request), response)
            return response
        if not self.pipeline:
            raise ValueError("ServingEngine 需要 pipeline 或 experiment")
        ranked = self.pipeline.recommend(request.user, list(request.items))
        if self.metrics:
            self.metrics.log_impressions(ranked)
        response = RecommendResponse(items=ranked)
        if self.cache:
            self.cache.set(self._build_cache_key(request), response)
        return response

    def log_click(self, user_id: str, item_id: str) -> None:
        """记录点击事件并更新在线指标。"""
        if self.metrics:
            self.metrics.log_click(item_id)

    def batch_recommend(self, requests: Iterable[RecommendRequest]) -> List[RecommendResponse]:
        """批量处理推荐请求。"""
        responses = []
        for request in requests:
            responses.append(self.recommend(request))
        return responses
