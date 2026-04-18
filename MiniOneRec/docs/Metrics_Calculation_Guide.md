# 参数计算介绍（HitRate / NDCG / AUC）

本文档详细说明以下两个脚本中的指标计算方式：

- `calc.py`（主评测链路：`evaluate.sh -> calc.py`）
- `RL_Reward/SASRec/test_reward_consistency.py`（SASRec reward 一致性评测）

---

## 1. `calc.py` 的指标计算

### 1.1 输入与评测对象

- 输入文件 `--path` 是 `final_result_*.json`，每条样本包含：
  - `output`：真实目标（正样本）
  - `predict`：模型生成的候选列表（已按推荐顺序排序）
- `calc.py` 默认评估 `topk_list = [1, 3, 5, 10, 20, 50]`，实际生效的是 `<= n_beam` 的部分。
- 每条样本只存在一个真实目标，因此是单正样本的 Top-K 评估。

### 1.2 关键中间量

对每条样本：

1. 从 `output` 取出真实目标 `target_item`。
2. 在 `predict` 中线性查找目标首次出现位置 `minID`（0-based）。
   - 若命中：`minID in [0, n_beam-1]`
   - 若未命中：`minID` 保持为大值（代码里初始化为 `1000000`）

其中：

- **排名（1-based）**：`rank = minID + 1`
- `n_beam = len(predict)`（通常和 `evaluate.py` 的 `num_beams` 一致）

### 1.3 HitRate@K（HR@K）

对每个 `K`：

- 若 `minID < K`（即 `rank <= K`），该样本在 `HR@K` 记 1；
- 否则记 0。

最终：

`HR@K = (命中样本数) / (总样本数)`

### 1.4 NDCG@K

对每个 `K`：

- 若 `rank <= K`，该样本贡献 `1 / log(rank + 1)`（代码使用自然对数）；
- 否则贡献 0。

最后再除以 `1/log(2)`，等价于标准写法：

`NDCG@K = mean( 1 / log2(rank + 1) )`（未命中样本贡献 0）

### 1.5 AUC_pos_vs_neg（当前 `calc.py` 版本）

- 构造正负样本对
  - 正样本（1个）：该条样本真实目标 target_item（来自 output），在 predict 中的位置 minID。
  - 负样本（n_beam-1个）：同一条 predict 列表里除正样本位置外的所有候选。
  - 配对方式：正样本分别和每个负样本形成一对，共 n_beam-1 对。

- 每对如何打分
  - 如果正样本排位更靠前（pos_rank_idx < neg_rank_idx），该对记 1.0；
  - 否则记 0.0（代码里 == 分支理论上不会触发，因为同一索引已 continue 掉）。
单样本 AUC = 这些 pair 分数的平均；全局 AUC = 所有样本单样本 AUC 的平均。


`calc.py` 中没有候选 raw score，只有排序结果，因此使用**基于排名的 pairwise AUC**：

- 将真实目标视为正样本，其余候选视为负样本；
- 对每个负样本做一次比较：
  - 若正样本排在负样本前，记 1.0；
  - 若并列，记 0.5（当前排名列表中实际不会出现并列索引）；
  - 否则记 0.0。
- 单样本 AUC 为上述比较的平均值。

若正样本排名为 `rank`（1-based），候选总数 `n_beam`，则单样本 AUC 可写为：

`AUC_i = (n_beam - rank) / (n_beam - 1)`（命中时）

若未命中（目标不在 `predict`），该样本 `AUC_i = 0`。

最终：

`AUC_pos_vs_neg = mean(AUC_i)`

> 直观理解：AUC 表示“随机取一个负样本时，正样本排在它前面的概率”。

### 1.6 其他输出（非三大指标）

- `CC`：预测结果中不在 `item_dict` 的条目计数（用于观察无效 item 生成问题）。

---

## 2. `test_reward_consistency.py` 的指标计算

### 2.1 输入与评测对象

- 读取测试集，构造 `EvalRecord(history_ids, target_id)`。
- 每条样本构造候选集：
  - 1 个正样本（真实 `target_id`）
  - `num_candidates - 1` 个随机负样本
- 用 `SASRec.forward_eval` 得到候选分数 `cand_scores`。
- 依据分数降序得到正样本排名 `rank`。

### 2.2 Hit@K

对每个 `K`：

- 若 `rank <= K`，则 `Hit@K` 的计数加 1。

最终：

`Hit@K = 命中数 / 样本数`

### 2.3 NDCG@K

对每个 `K`：

- 若 `rank <= K`，则累加 `1 / log2(rank + 1)`；
- 否则加 0。

最终：

`NDCG@K = 平均(1 / log2(rank + 1))`

### 2.4 AUC_pos_vs_neg（分数版 pairwise AUC）

脚本按“正样本分数 vs 负样本分数”逐对比较：

- `1.0`：`pos_score > neg_score`
- `0.5`：`pos_score == neg_score`
- `0.0`：`pos_score < neg_score`

单样本 AUC：

`AUC_i = mean_j [ I(pos>neg_j) + 0.5*I(pos=neg_j) ]`

整体 AUC：

`AUC_pos_vs_neg = mean_i(AUC_i)`

这是标准的二分类 AUC 在“1 正 + 多负”候选设定下的实现形式。

---

## 3. 两个脚本口径对齐与差异

### 3.1 一致点

- 都是“单正样本 + 若干负样本”的排序评估思想；
- HitRate 与 NDCG 的定义本质一致；
- AUC 都是 pairwise 概率语义：正样本优于负样本的平均概率。

### 3.2 差异点

1. **AUC 输入信息不同**
   - `test_reward_consistency.py`：有候选 raw score（分数级比较）
   - `calc.py`：只有排序结果（排名级比较）

2. **AUC 对“分数并列”的处理能力不同**
   - 分数版可真实处理 `pos == neg`（记 0.5）
   - 排名版通常不出现并列排名，实际等价于 0/1 比较

3. **候选集合来源不同**
   - `test_reward_consistency.py`：在线随机采样负样本
   - `calc.py`：使用生成模型输出的 `predict` 列表作为候选

### 3.3 如何理解“对齐”

- 如果只看指标定义语义，当前 `calc.py` 的 `AUC_pos_vs_neg` 已与 `test_reward_consistency.py` 对齐（都是 pairwise AUC）。
- 若想做到“完全同口径（分数级）”，需要在生成评估链路中保留每个候选的 raw score，再按分数版公式计算 AUC。

---

## 4. 简单示例（帮助验证）

设一条样本有 5 个候选，正样本排名为第 2 名（`rank=2`）：

- `Hit@1 = 0`，`Hit@3 = 1`
- `NDCG@3 = 1 / log2(2+1) = 1 / log2(3)`
- 排名版 AUC：正样本在 4 个负样本中赢了 3 个  
  `AUC_i = 3/4 = 0.75`

若使用分数版，且与上述排序一致且无并列，则得到同样的 `0.75`。

---

## 5. 代码定位

- `calc.py`
  - HR/NDCG 主循环：`minID` 查找与 `topk` 累计逻辑
  - AUC 逻辑：`AUC_pos_vs_neg` 的 rank-based pairwise 计算段
- `RL_Reward/SASRec/test_reward_consistency.py`
  - 候选打分：`score_candidates`
  - 排名计算：`compute_rank_desc`
  - AUC 逻辑：`pos_score` 与 `neg_scores` 的逐对比较平均

## 6. AUC 计算的必要性
正样本只有一个，有必要计算AUC吗
  有必要，但要看你的评估目标。

- 即使每条样本只有 1 个正样本，AUC 依然有定义。它本质是“这个正样本比分到的负样本更靠前的概率”（pairwise 胜率）。
- 在你当前场景里有意义：你是“1 正 + 多负候选”的排序任务，AUC 能反映整体排序质量，不只看 Top-K 截断。
- 和 HR/NDCG 互补：
  HR/NDCG 关注前 K（更贴近推荐使用场景）
  AUC 看全候选的相对排序（更平滑、阈值无关）

- 但也有局限：
  强依赖负样本构造（随机采样方式、数量会影响 AUC）。
  只有 1 正时，AUC 不是经典全局 ROC 曲线语义，更接近“正样本对负样本的平均胜率”。
  在你 calc.py 这种只有排名无分数的实现里，AUC 本质接近“归一化 rank 指标”，信息量会和 NDCG 有部分重叠。