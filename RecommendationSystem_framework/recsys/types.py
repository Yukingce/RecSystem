from dataclasses import dataclass, field
from typing import Dict, List

"""核心数据结构定义，用于贯穿召回、粗排、精排与评估流程。"""


@dataclass(frozen=True)
class User:
    """用户画像与上下文特征。"""

    user_id: str
    features: Dict[str, float] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Item:
    """物料与对应特征。"""

    item_id: str
    features: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class Candidate:
    """召回阶段候选，score 表示召回打分。"""

    item: Item
    score: float
    sources: List[str] = field(default_factory=list)


@dataclass
class RankedItem:
    """排序后结果，score 表示当前阶段的排序分。"""

    item: Item
    score: float


@dataclass(frozen=True)
class Interaction:
    """离线评估的用户交互标签。"""

    user_id: str
    item_id: str
    label: float


@dataclass
class Dataset:
    """离线数据集封装。"""

    users: List[User]
    items: List[Item]
    interactions: List[Interaction]
