from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import Interaction, Item, User

"""特征模块，包含特征仓库与特征工程逻辑。"""


@dataclass
class FeatureStore:
    """内存版特征仓库，用于补充用户和物料特征。"""

    user_features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    item_features: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def enrich_user(self, user: User) -> User:
        """补充用户特征，优先保留传入特征。"""
        extra_features = self.user_features.get(user.user_id, {})
        if not extra_features:
            return user
        merged = {**extra_features, **user.features}
        return User(user_id=user.user_id, features=merged, keywords=list(user.keywords))

    def enrich_items(self, items: Sequence[Item]) -> List[Item]:
        """补充物料特征，优先保留传入特征。"""
        enriched: List[Item] = []
        for item in items:
            extra_features = self.item_features.get(item.item_id, {})
            if extra_features:
                merged = {**extra_features, **item.features}
                enriched.append(Item(item_id=item.item_id, features=merged, tags=list(item.tags)))
            else:
                enriched.append(item)
        return enriched

    def update_user_features(self, user_id: str, features: Dict[str, float]) -> None:
        """写入用户特征，覆盖已有字段。"""
        existing = self.user_features.get(user_id, {})
        self.user_features[user_id] = {**existing, **features}

    def update_item_features(self, item_id: str, features: Dict[str, float]) -> None:
        """写入物料特征，覆盖已有字段。"""
        existing = self.item_features.get(item_id, {})
        self.item_features[item_id] = {**existing, **features}

    def bulk_update_item_features(self, updates: Iterable[tuple[str, Dict[str, float]]]) -> None:
        """批量写入物料特征。"""
        for item_id, features in updates:
            self.update_item_features(item_id, features)


@dataclass
class FeatureEngineer:
    """基于交互数据构建基础特征的工程器。"""

    def build_feature_store(
        self,
        users: Iterable[User],
        items: Iterable[Item],
        interactions: Iterable[Interaction],
    ) -> FeatureStore:
        """构建 FeatureStore，包含用户活跃度与物料热度特征。"""
        store = FeatureStore()
        user_counts: Dict[str, int] = {}
        item_counts: Dict[str, int] = {}

        for interaction in interactions:
            if interaction.label <= 0:
                continue
            user_counts[interaction.user_id] = user_counts.get(interaction.user_id, 0) + 1
            item_counts[interaction.item_id] = item_counts.get(interaction.item_id, 0) + 1

        max_user = max(user_counts.values()) if user_counts else 1
        max_item = max(item_counts.values()) if item_counts else 1

        for user in users:
            activity = user_counts.get(user.user_id, 0) / max_user
            store.update_user_features(user.user_id, {"activity": activity})

        for item in items:
            popularity = item_counts.get(item.item_id, 0) / max_item
            store.update_item_features(item.item_id, {"popularity": popularity})

        return store

    def apply_default_item_features(self, items: Iterable[Item]) -> List[Item]:
        """补齐物料的缺省特征，避免模型打分空值。"""
        normalized: List[Item] = []
        for item in items:
            features = {
                "quality": item.features.get("quality", 0.5),
                "newness": item.features.get("newness", 0.5),
                "freshness": item.features.get("freshness", 0.5),
                "ctr": item.features.get("ctr", 0.1),
            }
            merged = {**features, **item.features}
            normalized.append(Item(item_id=item.item_id, features=merged, tags=list(item.tags)))
        return normalized

    def build_tag_embeddings(
        self,
        users: Sequence[User],
        items: Sequence[Item],
    ) -> Tuple[List[User], List[Item], List[str]]:
        """根据标签与关键词构造简单 one-hot 嵌入特征。"""
        vocabulary = sorted({tag for item in items for tag in item.tags})
        embedding_keys = [f"emb_tag_{tag}" for tag in vocabulary]

        def encode_tags(tags: Iterable[str]) -> Dict[str, float]:
            tag_set = set(tags)
            return {f"emb_tag_{tag}": 1.0 for tag in tag_set}

        encoded_items: List[Item] = []
        for item in items:
            encoded = encode_tags(item.tags)
            merged = {**encoded, **item.features}
            encoded_items.append(Item(item_id=item.item_id, features=merged, tags=list(item.tags)))

        encoded_users: List[User] = []
        for user in users:
            encoded = encode_tags(user.keywords)
            merged = {**encoded, **user.features}
            encoded_users.append(User(user_id=user.user_id, features=merged, keywords=list(user.keywords)))

        return encoded_users, encoded_items, embedding_keys
