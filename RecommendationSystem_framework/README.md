# RecommendationSystem

一个轻量级推荐系统框架示例，覆盖召回、粗排、精排、特征工程、离线评估、实验分流与在线指标统计。结构按真实推荐链路拆分，支持快速接入自研模型与生成完整 demo。

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pipeline](https://img.shields.io/badge/Recommendation-Pipeline-brightgreen)
![Demo](https://img.shields.io/badge/Demo-Ready-orange)

## 亮点
- 端到端链路齐全：召回、粗排、精排、后处理、在线指标
- 结构工程化：清晰模块划分，便于替换自研模型
- Demo 即跑即用：单文件脚本展示全链路
- 可配置可扩展：注册表 + 配置构建流水线

## 适用场景
- 快速搭建推荐系统框架原型
- 教学/面试/技术分享演示
- 验证召回/排序/特征方案组合

## 项目结构
```
RecommendationSystem/
├── demo.py
└── recsys/
    ├── __init__.py
    ├── types.py
    ├── models.py
    ├── pipeline.py
    ├── features.py           # 特征工程与特征仓库
    ├── retrieval.py          # 向量检索与相似度
    ├── config.py             # 配置与模型注册表
    ├── policy.py             # 规则过滤与后处理
    ├── serving_support.py    # 缓存、指标、反馈闭环
    ├── data.py               # 数据接入与切分
    ├── evaluation.py         # 离线评估与实验对比
    └── serving.py            # 在线 Serving 抽象
```

## 快速开始
在终端执行：
```bash
python3 demo.py
```

你将看到完整链路输出：
- baseline / variant 结果对比
- serving 分流与实时结果
- 反馈闭环与在线指标
- 实验指标与 uplift
- 训练切分与负采样规模

## Demo 输出示例
```text
======== Baseline 推荐结果 ========
 1. item_2 score=1.5340 tags=['tech', 'ai']
 2. item_3 score=1.0960 tags=['travel', 'photo']
 3. item_5 score=1.3020 tags=['ai', 'productivity']

======== Serving 分流与结果 ========
Serving bucket: variant
 1. item_2 score=1.4960
 2. item_1 score=0.6480
 3. item_5 score=0.9820

======== 在线指标 ========
{'impressions': 3, 'clicks': 1, 'ctr': 0.3333333333333333, 'avg_position': 2.0}
```

## 架构一览
```
用户/上下文
   │
召回 → 粗排 → 精排 → 规则过滤/多样性 → 返回结果
   │                             │
特征工程/特征仓库                在线指标/反馈闭环
```

## 使用说明

### 1. 自定义召回/粗排/精排模型
实现对应接口并替换到 pipeline：

```python
from recsys.models import RecallModel, CoarseRankModel, RerankModel

class MyRecall(RecallModel):
    name = "my_recall"
    def recall(self, user, items, max_candidates):
        ...

class MyCoarse(CoarseRankModel):
    name = "my_coarse"
    def score(self, user, item, recall_score):
        ...

class MyRerank(RerankModel):
    name = "my_rerank"
    def score(self, user, item, coarse_score):
        ...
```

### 2. 构建流水线
```python
from recsys.pipeline import RecommendationPipeline, RecallStage, CoarseRankStage, RerankStage

pipeline = RecommendationPipeline(
    recall_stage=RecallStage(models=[MyRecall()], max_candidates=500),
    coarse_rank_stage=CoarseRankStage(model=MyCoarse(), top_k=200),
    rerank_stage=RerankStage(model=MyRerank(), top_k=50),
)
```

### 3. 特征工程与补全
```python
from recsys import FeatureEngineer

engineer = FeatureEngineer()
feature_store = engineer.build_feature_store(users, items, interactions)
items = engineer.apply_default_item_features(items)
```

### 4. 离线评估
```python
from recsys import OfflineEvaluator

evaluator = OfflineEvaluator(k=10)
result = evaluator.evaluate(pipeline, users, items, interactions)
print(result.metrics)
```

### 5. 实验分流
```python
from recsys import Experiment, ExperimentConfig

experiment = Experiment(
    config=ExperimentConfig(name="exp", traffic_split=0.5, k=10),
    baseline=baseline_pipeline,
    variant=variant_pipeline,
)
```

### 6. Serving 调用
```python
from recsys import ServingEngine, RecommendRequest

engine = ServingEngine(experiment=experiment)
response = engine.recommend(RecommendRequest(user=user, items=items))
print(response.bucket, response.items[:3])
```

### 7. 向量召回与配置化
```python
from recsys import ModelRegistry, PipelineConfig, VectorIndex, build_pipeline_from_config, register_default_models

registry = ModelRegistry()
register_default_models(registry)

vector_index = VectorIndex.build(items, embedding_keys)
config = PipelineConfig.from_dict({
    "recall": [{"name": "vector_recall", "params": {"top_n": 100}}],
    "coarse": {"name": "linear_coarse_rank", "params": {}},
    "rerank": {"name": "linear_rerank", "params": {}},
})
pipeline = build_pipeline_from_config(
    config=config,
    registry=registry,
    extra_params={"vector_recall": {"index": vector_index}},
)
```

### 8. 在线指标统计
```python
from recsys import OnlineMetrics

metrics = OnlineMetrics()
metrics.log_impressions(response.items)
metrics.log_click(response.items[0].item.item_id)
print(metrics.report())
```

### 9. 规则过滤与后处理
```python
from recsys import TagBlacklistFilter, MinFeatureFilter, DeduplicatePostProcessor, TagDiversityPostProcessor

pre_filters = [
    TagBlacklistFilter(banned_tags={"outdoor"}),
    MinFeatureFilter(feature="quality", min_value=0.3),
]
post_processors = [
    DeduplicatePostProcessor(),
    TagDiversityPostProcessor(max_per_tag=1),
]
```

### 10. 缓存与反馈闭环
```python
from recsys import LRUCache, FeedbackLogger, FeedbackUpdater

cache = LRUCache(capacity=128)
logger = FeedbackLogger()
logger.log_impression(user.user_id, "item_1")
logger.log_click(user.user_id, "item_1")
updater = FeedbackUpdater()
updater.apply(feature_store, logger.drain())
```

### 11. 数据切分与负采样
```python
from recsys import train_validation_split, build_negative_samples

split = train_validation_split(interactions, validation_ratio=0.2)
negatives = build_negative_samples(split.train, [item.item_id for item in items], negative_per_positive=1)
```

## 模块速览
- types.py：核心数据结构与数据集定义
- models.py：召回/粗排/精排模型基类与示例实现
- pipeline.py：召回 → 排序 → 后处理的推荐主链路
- features.py：特征工程与特征仓库补全
- retrieval.py：向量索引与相似度检索
- policy.py：规则过滤、去重与多样性策略
- serving_support.py：缓存、在线指标与反馈闭环
- evaluation.py：离线评估与实验对比
- serving.py：在线 Serving 编排与批处理

## FAQ
- Q: 如何接入自研模型？
  A: 继承基类并替换 pipeline 中的模型实例即可。
- Q: 如何接入线上特征？
  A: 在 features.py 中扩展特征补全逻辑，或接入外部特征服务。
- Q: 如何替换召回与排序的真实模型？
  A: 将示例模型替换为你自己的模型打分实现。

## 扩展建议
- 接入在线特征与实时画像
- 替换粗排/精排为真实模型打分
- 增加模型热更新与灰度发布
- 打通日志埋点与可视化监控
