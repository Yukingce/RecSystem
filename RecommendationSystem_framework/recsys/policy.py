from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .types import Item, RankedItem, User

"""规则与后处理模块，用于过滤与结果优化。"""


class ItemFilter(ABC):
    """物料过滤器接口。"""

    @abstractmethod
    def apply(self, user: User, items: Sequence[Item]) -> List[Item]:
        """返回过滤后的物料列表。"""
        raise NotImplementedError


@dataclass(frozen=True)
class TagBlacklistFilter(ItemFilter):
    """根据黑名单标签过滤物料。"""

    banned_tags: Iterable[str]

    def apply(self, user: User, items: Sequence[Item]) -> List[Item]:
        banned = set(self.banned_tags)
        return [item for item in items if not banned.intersection(item.tags)]


@dataclass(frozen=True)
class MinFeatureFilter(ItemFilter):
    """按特征阈值过滤物料。"""

    feature: str
    min_value: float

    def apply(self, user: User, items: Sequence[Item]) -> List[Item]:
        return [item for item in items if item.features.get(self.feature, 0.0) >= self.min_value]


@dataclass(frozen=True)
class CompositeFilter(ItemFilter):
    """组合多个过滤器。"""

    filters: Sequence[ItemFilter]

    def apply(self, user: User, items: Sequence[Item]) -> List[Item]:
        filtered = list(items)
        for item_filter in self.filters:
            filtered = item_filter.apply(user, filtered)
        return filtered


class PostProcessor(ABC):
    """后处理器接口。"""

    @abstractmethod
    def apply(self, user: User, ranked_items: Sequence[RankedItem]) -> List[RankedItem]:
        """处理排序结果并返回新的列表。"""
        raise NotImplementedError


@dataclass(frozen=True)
class DeduplicatePostProcessor(PostProcessor):
    """根据 item_id 去重，保留最高分。"""

    def apply(self, user: User, ranked_items: Sequence[RankedItem]) -> List[RankedItem]:
        seen: Dict[str, RankedItem] = {}
        for ranked in ranked_items:
            item_id = ranked.item.item_id
            if item_id not in seen or ranked.score > seen[item_id].score:
                seen[item_id] = ranked
        return list(seen.values())


@dataclass(frozen=True)
class TagDiversityPostProcessor(PostProcessor):
    """按标签进行打散，限制单一标签的最大占比。"""

    max_per_tag: int = 1

    def apply(self, user: User, ranked_items: Sequence[RankedItem]) -> List[RankedItem]:
        tag_counts: Dict[str, int] = {}
        diversified: List[RankedItem] = []
        overflow: List[RankedItem] = []

        for ranked in ranked_items:
            tags = ranked.item.tags or ["_no_tag"]
            allowed = True
            for tag in tags:
                if tag_counts.get(tag, 0) >= self.max_per_tag:
                    allowed = False
                    break
            if allowed:
                diversified.append(ranked)
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            else:
                overflow.append(ranked)

        return diversified + overflow
