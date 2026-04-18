from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .types import Dataset, Interaction, Item, User

"""数据接入层，用于组织用户、物料与交互日志。"""


@dataclass
class InMemoryDataSource:
    """简化的数据源实现，适合 demo 与离线实验。"""

    dataset: Dataset

    def get_users(self) -> Sequence[User]:
        """返回用户列表。"""
        return self.dataset.users

    def get_items(self) -> Sequence[Item]:
        """返回物料列表。"""
        return self.dataset.items

    def get_interactions(self) -> Sequence[Interaction]:
        """返回交互标签。"""
        return self.dataset.interactions


def build_demo_dataset() -> Dataset:
    """构造小型示例数据集。"""
    users = [
        User(user_id="user_1", features={"activity": 0.6}, keywords=["ai", "tech"]),
        User(user_id="user_2", features={"activity": 0.3}, keywords=["sports", "outdoor"]),
        User(user_id="user_3", features={"activity": 0.9}, keywords=["travel", "photo"]),
    ]

    items = [
        Item(
            item_id="item_1",
            features={"quality": 0.7, "newness": 0.3, "freshness": 0.4, "ctr": 0.12},
            tags=["sports", "video"],
        ),
        Item(
            item_id="item_2",
            features={"quality": 0.9, "newness": 0.6, "freshness": 0.7, "ctr": 0.22},
            tags=["tech", "ai"],
        ),
        Item(
            item_id="item_3",
            features={"quality": 0.5, "newness": 0.9, "freshness": 0.9, "ctr": 0.18},
            tags=["travel", "photo"],
        ),
        Item(
            item_id="item_4",
            features={"quality": 0.4, "newness": 0.2, "freshness": 0.3, "ctr": 0.05},
            tags=["sports", "outdoor"],
        ),
        Item(
            item_id="item_5",
            features={"quality": 0.8, "newness": 0.7, "freshness": 0.8, "ctr": 0.26},
            tags=["ai", "productivity"],
        ),
    ]

    interactions = [
        Interaction(user_id="user_1", item_id="item_2", label=1.0),
        Interaction(user_id="user_1", item_id="item_5", label=1.0),
        Interaction(user_id="user_2", item_id="item_1", label=1.0),
        Interaction(user_id="user_2", item_id="item_4", label=1.0),
        Interaction(user_id="user_3", item_id="item_3", label=1.0),
    ]

    return Dataset(users=users, items=items, interactions=interactions)


@dataclass(frozen=True)
class SplitResult:
    """训练/验证切分结果。"""

    train: List[Interaction]
    validation: List[Interaction]


def train_validation_split(
    interactions: Iterable[Interaction],
    validation_ratio: float = 0.2,
) -> SplitResult:
    """按时间顺序简单切分数据。"""
    interactions = list(interactions)
    if not interactions:
        return SplitResult(train=[], validation=[])
    cutoff = int(len(interactions) * (1 - validation_ratio))
    return SplitResult(train=interactions[:cutoff], validation=interactions[cutoff:])


def build_negative_samples(
    positives: Iterable[Interaction],
    all_item_ids: Iterable[str],
    negative_per_positive: int = 1,
) -> List[Interaction]:
    """构造简单的负样本。"""
    positives = list(positives)
    all_item_ids = list(all_item_ids)
    negative_samples: List[Interaction] = []
    if not all_item_ids:
        return negative_samples
    for positive in positives:
        for offset in range(negative_per_positive):
            item_id = all_item_ids[(hash((positive.user_id, offset)) % len(all_item_ids))]
            if item_id == positive.item_id:
                continue
            negative_samples.append(
                Interaction(user_id=positive.user_id, item_id=item_id, label=0.0)
            )
    return negative_samples
