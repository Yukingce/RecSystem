from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, List, TypeVar

from .features import FeatureStore
from .types import RankedItem

"""Serving 支撑组件：缓存、在线指标与反馈闭环。"""

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class LRUCache(Generic[K, V]):
    """固定容量的 LRU 缓存。"""

    capacity: int = 128
    store: Dict[K, V] = field(default_factory=dict)
    order: List[K] = field(default_factory=list)

    def get(self, key: K) -> V | None:
        """读取缓存并更新访问顺序。"""
        if key not in self.store:
            return None
        self._touch(key)
        return self.store[key]

    def set(self, key: K, value: V) -> None:
        """写入缓存，必要时淘汰最旧元素。"""
        if key in self.store:
            self.store[key] = value
            self._touch(key)
            return
        if len(self.store) >= self.capacity:
            oldest = self.order.pop(0)
            self.store.pop(oldest, None)
        self.store[key] = value
        self.order.append(key)

    def _touch(self, key: K) -> None:
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)

    def bulk_get(self, keys: Iterable[K]) -> List[V | None]:
        """批量读取缓存。"""
        return [self.get(key) for key in keys]


@dataclass
class OnlineMetrics:
    """记录曝光与点击用于在线指标统计。"""

    impressions: int = 0
    clicks: int = 0
    position_sum: int = 0
    item_impressions: Dict[str, int] = field(default_factory=dict)
    item_clicks: Dict[str, int] = field(default_factory=dict)

    def log_impressions(self, ranked_items: Iterable[RankedItem]) -> None:
        """记录曝光。"""
        for position, ranked in enumerate(ranked_items, start=1):
            item_id = ranked.item.item_id
            self.impressions += 1
            self.position_sum += position
            self.item_impressions[item_id] = self.item_impressions.get(item_id, 0) + 1

    def log_click(self, item_id: str) -> None:
        """记录点击。"""
        self.clicks += 1
        self.item_clicks[item_id] = self.item_clicks.get(item_id, 0) + 1

    def ctr(self) -> float:
        """计算整体 CTR。"""
        return self.clicks / self.impressions if self.impressions else 0.0

    def avg_position(self) -> float:
        """计算平均曝光位置。"""
        return self.position_sum / self.impressions if self.impressions else 0.0

    def item_ctr(self) -> Dict[str, float]:
        """计算物料级 CTR。"""
        result: Dict[str, float] = {}
        for item_id, impressions in self.item_impressions.items():
            clicks = self.item_clicks.get(item_id, 0)
            result[item_id] = clicks / impressions if impressions else 0.0
        return result

    def report(self) -> Dict[str, object]:
        """生成指标汇总。"""
        return {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "ctr": self.ctr(),
            "avg_position": self.avg_position(),
            "item_ctr": self.item_ctr(),
        }


@dataclass(frozen=True)
class FeedbackEvent:
    """反馈事件定义。"""

    user_id: str
    item_id: str
    event_type: str
    weight: float = 1.0


@dataclass
class FeedbackLogger:
    """反馈日志收集器。"""

    events: List[FeedbackEvent] = field(default_factory=list)

    def log_impression(self, user_id: str, item_id: str, weight: float = 1.0) -> None:
        """记录曝光事件。"""
        self.events.append(FeedbackEvent(user_id=user_id, item_id=item_id, event_type="impression", weight=weight))

    def log_click(self, user_id: str, item_id: str, weight: float = 1.0) -> None:
        """记录点击事件。"""
        self.events.append(FeedbackEvent(user_id=user_id, item_id=item_id, event_type="click", weight=weight))

    def drain(self) -> List[FeedbackEvent]:
        """读取并清空事件。"""
        events = list(self.events)
        self.events.clear()
        return events


@dataclass
class FeedbackUpdater:
    """根据反馈更新特征仓库。"""

    click_weight: float = 2.0

    def apply(self, store: FeatureStore, events: Iterable[FeedbackEvent]) -> Dict[str, float]:
        """更新物料热度特征并返回新热度。"""
        counts: Dict[str, float] = {}
        for event in events:
            weight = event.weight
            if event.event_type == "click":
                weight *= self.click_weight
            counts[event.item_id] = counts.get(event.item_id, 0.0) + weight

        max_count = max(counts.values()) if counts else 1.0
        popularity: Dict[str, float] = {}
        for item_id, count in counts.items():
            value = count / max_count
            store.update_item_features(item_id, {"popularity": value})
            popularity[item_id] = value
        return popularity
