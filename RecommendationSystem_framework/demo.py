from recsys import (
    DeduplicatePostProcessor,
    Experiment,
    ExperimentConfig,
    FeatureEngineer,
    FeedbackLogger,
    FeedbackUpdater,
    InMemoryDataSource,
    LRUCache,
    MinFeatureFilter,
    ModelRegistry,
    OnlineMetrics,
    PipelineConfig,
    RecommendationPipeline,
    RecommendRequest,
    ServingEngine,
    TagBlacklistFilter,
    TagDiversityPostProcessor,
    VectorIndex,
    build_demo_dataset,
    build_negative_samples,
    build_pipeline_from_config,
    register_default_models,
    train_validation_split,
)
from recsys.types import Item, User

"""推荐系统 demo，包含数据、特征工程、实验分流与离线评估示例。"""


def build_pipeline(
    registry: ModelRegistry,
    vector_index: VectorIndex,
    feature_store,
    use_tag_boost: bool,
    pre_filters,
    post_processors,
) -> RecommendationPipeline:
    """通过配置与注册表构建流水线。"""
    payload = {
        "recall": [
            {"name": "popularity_recall", "params": {"top_n": 4}},
            {"name": "keyword_recall", "params": {"top_n": 4}},
            {"name": "vector_recall", "params": {"top_n": 4}},
        ],
        "coarse": {
            "name": "linear_coarse_rank",
            "params": {"feature_weights": {"quality": 0.6, "popularity": 0.2, "newness": 0.2}, "recall_weight": 0.4},
        },
        "rerank": {
            "name": "tag_boost_rerank" if use_tag_boost else "linear_rerank",
            "params": {"boost": 0.25, "coarse_weight": 0.6} if use_tag_boost else {
                "feature_weights": {"quality": 0.5, "freshness": 0.3, "ctr": 0.2},
                "coarse_weight": 0.5,
            },
        },
        "recall_max": 5,
        "coarse_top_k": 4,
        "rerank_top_k": 3,
    }
    config = PipelineConfig.from_dict(payload)
    return build_pipeline_from_config(
        config=config,
        registry=registry,
        feature_store=feature_store,
        pre_filters=pre_filters,
        post_processors=post_processors,
        extra_params={"vector_recall": {"index": vector_index}},
    )


def print_section(title: str) -> None:
    """打印模块标题。"""
    line = "=" * 8
    print(f"{line} {title} {line}")


def print_ranked(title: str, user: User, items: list[Item], pipeline: RecommendationPipeline) -> None:
    """打印推荐结果。"""
    results = pipeline.recommend(user, items)
    print_section(title)
    for rank, ranked in enumerate(results, start=1):
        print(f"{rank:>2}. {ranked.item.item_id:<6} score={ranked.score:.4f} tags={ranked.item.tags}")
    print()


def main() -> None:
    data_source = InMemoryDataSource(dataset=build_demo_dataset())
    users = list(data_source.get_users())
    items = list(data_source.get_items())
    interactions = list(data_source.get_interactions())

    engineer = FeatureEngineer()
    items = engineer.apply_default_item_features(items)
    feature_store = engineer.build_feature_store(users, items, interactions)
    users, items, embedding_keys = engineer.build_tag_embeddings(users, items)
    vector_index = VectorIndex.build(items, embedding_keys)

    registry = ModelRegistry()
    register_default_models(registry)

    pre_filters = [
        TagBlacklistFilter(banned_tags={"outdoor"}),
        MinFeatureFilter(feature="quality", min_value=0.3),
    ]
    post_processors = [
        DeduplicatePostProcessor(),
        TagDiversityPostProcessor(max_per_tag=1),
    ]

    baseline_pipeline = build_pipeline(
        registry,
        vector_index,
        feature_store,
        use_tag_boost=False,
        pre_filters=pre_filters,
        post_processors=post_processors,
    )
    variant_pipeline = build_pipeline(
        registry,
        vector_index,
        feature_store,
        use_tag_boost=True,
        pre_filters=pre_filters,
        post_processors=post_processors,
    )

    print_ranked("Baseline 推荐结果", users[0], items, baseline_pipeline)
    print_ranked("Variant 推荐结果", users[0], items, variant_pipeline)

    experiment = Experiment(
        config=ExperimentConfig(name="tag_boost_exp", traffic_split=0.5, k=3),
        baseline=baseline_pipeline,
        variant=variant_pipeline,
    )

    metrics = OnlineMetrics()
    cache = LRUCache(capacity=32)
    engine = ServingEngine(experiment=experiment, metrics=metrics, cache=cache)
    response = engine.recommend(RecommendRequest(user=users[0], items=items))
    print_section("Serving 分流与结果")
    print("Serving bucket:", response.bucket)
    for rank, ranked in enumerate(response.items, start=1):
        print(f"{rank:>2}. {ranked.item.item_id:<6} score={ranked.score:.4f}")
    print()

    feedback_logger = FeedbackLogger()
    for ranked in response.items:
        feedback_logger.log_impression(users[0].user_id, ranked.item.item_id)
    if response.items:
        feedback_logger.log_click(users[0].user_id, response.items[0].item.item_id)
    updater = FeedbackUpdater()
    popularity = updater.apply(feature_store, feedback_logger.drain())
    print_section("反馈闭环更新")
    print("Updated popularity:", popularity)
    print()

    if response.items:
        metrics.log_click(response.items[0].item.item_id)
    print_section("在线指标")
    print(metrics.report())
    print()

    exp_result = experiment.evaluate(users, items, interactions)
    print_section("实验对比")
    print("Baseline:", exp_result.baseline_metrics)
    print("Variant :", exp_result.variant_metrics)
    print("Uplift  :", exp_result.uplift)
    print()

    split = train_validation_split(interactions, validation_ratio=0.4)
    negatives = build_negative_samples(split.train, [item.item_id for item in items], negative_per_positive=1)
    print_section("训练数据切分")
    print("Train size     :", len(split.train))
    print("Validation size:", len(split.validation))
    print("Negative size  :", len(negatives))


if __name__ == "__main__":
    main()
