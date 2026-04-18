# SASRec 预训练（用于 `rl.py` 的 reward 打分）

本目录提供一套可独立运行的 SASRec 预训练流程，目标是产出可被 `rl.py` 直接加载的 `state_dict`，用于 `reward_type="sasrec"` 时的 `cf_reward` 打分。

---

## 1. 为什么这里选用 SID 离散 ID，而不是 item embedding 向量

这里选择 **SID -> item_id 的离散 ID 训练**，理由如下：

1. `rl.py` 的 `cf_reward` 本质是：
   - 用 `item2id` 将 completion（SID 字符串）映射为整数 `pred_ids`
   - 用 `model.forward_eval(seq, len_seq)` 得到全量 item logits
   - 用 `torch.gather` 取出 completion 对应 item 的分数
2. 这个流程并不要求额外 item embedding 文件；只要求：
   - 训练和推理时 item 编号一致
   - 模型输出维度是 `item_num`
3. 因此最稳妥方案是与 `cf_reward` 完全同构：**历史 SID 序列 -> item_id 序列 -> SASRec 打分**。

---

## 2. 目录结构

- `model.py`：完整 SASRec 模型定义（含 MultiHeadAttention、FFN）
- `data_builder.py`：训练/验证样本构造、正负样本数据集
- `train_sasrec.py`：完整训练与评估流程
- `__init__.py`：模块导出

---

## 3. 训练数据构建思路（对齐 `rl.py:95-96`）

`rl.py` 中 reward 所依赖的数据来自 `SidDataset(train_file, ...)`，核心字段是：

- `history_item_sid`：用户历史 SID 列表（字符串形式）
- `item_sid`：下一条真实 SID（正样本）

本目录的数据构建逻辑如下：

1. 从 `info_file` 读取 SID 列表并构建 `sid2id`（与 `rl.py` 的 `item2id` 对齐）。
2. 对每一行样本：
   - 解析 `history_item_sid` 为列表，映射成 `seq_ids`
   - 取 `item_sid` 映射成 `target_id`（正样本）
   - 序列按 `max_seq_len` 截断并右侧补 pad（pad id = `item_num`）
3. 训练样本输出 `(seq, len_seq, target_item)`，在训练阶段做 listwise 候选集合构造。

### 3.1 详细构建流程（从原始文件到训练 batch）

下面按 `train_sasrec.py` 的真实执行顺序说明：

1. **读取映射字典（`load_sid_mappings`）**
   - 输入：`info_file`（格式：`sid \t title \t item_id`）
   - 输出：
     - `item_sids`: 按行顺序排列的 SID 列表
     - `sid2id`: SID 到连续整数 id 的映射
   - 约束：该映射同时决定模型输出维度 `item_num`，必须与 RL 阶段一致。

2. **构建序列样本（`build_training_records`）**
   - 输入：`train/valid/test` CSV + `sid2id`
   - 逐行处理：
     - 用 `ast.literal_eval` 解析 `history_item_sid`（字符串列表）
     - 将历史 SID 转成历史 id 序列 `seq_ids`
     - 将 `item_sid` 转成 `target_id`
     - 过滤规则：
       - `target_sid` 不在 `sid2id` -> 丢弃
       - 映射后 `seq_ids` 为空 -> 丢弃
     - 截断规则：`seq_ids = seq_ids[-max_seq_len:]`
   - 输出：`SequenceRecord(seq, len_seq, target)` 列表

3. **构建训练数据集（`SASRecTrainDataset`）**
   - 输入：`SequenceRecord` 列表
   - `__getitem__` 输出字段：
     - `seq`: 右补 pad 后的定长序列（pad id = `item_num`）
     - `len_seq`: 补 pad 前长度
     - `target_item`: 真实下一物品 id（CE 标签）
   - listwise 候选集策略在训练循环中完成：
     - `list_size = -1`：全量 item 做 softmax-CE
     - `list_size > 1`：每条样本构造 `1 个正样本 + (list_size-1) 个负样本` 的候选 list，再做 CE

4. **构建评估数据集（`SASRecEvalDataset`）**
   - 与训练集同样的序列处理（截断、右补 pad）
   - 输出 `seq/len_seq/target_item`，用于全量召回评估（HR/NDCG）

5. **进入 DataLoader**
   - 训练集：`shuffle=True`
   - 验证/测试：`shuffle=False`
   - 最终 batch 数据结构：
     - 训练：`{seq, len_seq, target_item}`
     - 评估：`{seq, len_seq, target_item}`

### 3.2 一个样本的构建示例

假设：

- `history_item_sid = ['<a_1><b_2><c_3>', '<a_4><b_5><c_6>']`
- `item_sid = '<a_7><b_8><c_9>'`
- `max_seq_len = 5`
- 其中 SID 映射后分别是 `[10, 23]`，目标是 `57`，`item_num = 3000`

则：

- 截断后 `seq_ids = [10, 23]`
- 右补 pad 后 `seq = [10, 23, 3000, 3000, 3000]`
- `len_seq = 2`
- `pos_item = 57`
这个样本会进入 listwise-CE 训练，作为类别标签 `target_item=57`。

---

## 4. 模型与奖励函数对齐关系

训练模型输出：

- `forward_eval(seq, len_seq) -> [batch_size, item_num]`

这与 `rl.py` 的打分方式严格兼容：

- `scores = gather(predictions, pred_ids)`

因此训练后的 `sasrec_state_dict.pth` 可以直接用于：

```python
model = SASRec(32, item_num, 10, 0.3, device)
model.load_state_dict(torch.load(cf_path))
```

> 注意：上面构造参数必须与训练参数一致（如 `hidden_size=32`, `max_seq_len=10`, `dropout=0.3`）。

---

## 5. 训练命令

在仓库根目录执行：

```bash
python RL_Reward/SASRec/train_sasrec.py \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --valid_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --test_file data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --output_dir RL_Reward/SASRec/output/industrial \
  --max_seq_len 10 \
  --hidden_size 32 \
  --dropout 0.3 \
  --list_size -1 \
  --batch_size 1024 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --early_stop 20 \
  --device cuda
```

### 5.1 模型训练流程（详细）

每个 epoch 的执行顺序如下：

1. **Train 阶段（`train_one_epoch`）**
   - 模型前向：`logits = model(seq, len_seq)`，输出 `[B, item_num]`
   - listwise 构造方式：
     - `list_size = -1`：直接对全量 `logits` 做 `CrossEntropyLoss(logits, target_item)`
     - `list_size > 1`：为每条样本构造候选 list（正样本 + 负样本），对 sampled logits 做 CE
   - 损失函数：`CrossEntropyLoss`
   - 反向传播 + `optimizer.step()`

2. **Valid/Test 阶段（`evaluate`）**
   - 用 `forward_eval(seq, len_seq)` 得到全量 item 排序分数
   - 对每个样本做 TopK 排名
   - 统计 `HR@K`、`NDCG@K`
   - 默认监控指标是最大 K 的 `NDCG@K`（例如 `NDCG@20`）

3. **最优模型保存**
   - 若验证监控指标提升：
     - 保存 `sasrec_state_dict.pth`
     - 刷新 `best_metrics.json`
   - 若持续 `early_stop` 轮不提升则提前停止

4. **训练结束产物**
   - `train_history.json`：完整 epoch 曲线
   - `train_config.json`：训练超参数和最优结果摘要

### 5.3 Hard Negative 采样机制（`list_size > 1` 时生效）

当 `list_size > 1` 使用 sampled listwise-CE 时，训练会为每条样本构造候选 list：

- 候选组成：`1 个正样本 + (list_size - 1) 个负样本`
- 正样本：当前样本的 `target_item`
- 负样本来源：由 `--neg_sampling_strategy` 决定


#### “样本”和“候选 item”的区别

train 样本数：29458
valid 样本数：3682
test 样本数：3683
item 总数：3106

- 在推荐训练里：
   29458 是训练样本条数（每条是一个 (history, target_item) 监督对）
   hard negatives 要选的是错误 item，来自 item 候选空间（这里约 3106 个 item）

#### 采样来源与流程

1. 先对当前样本前向得到全量 `logits`（长度为 `item_num=3106`）。
2. 通过 `topk(logits, hard_pool_size)` 构造 hard pool（高分候选池）。
3. 从 hard pool 中跳过正样本后，挑选 hard negatives。
   - 流程是：
      - 取一条训练样本（来自 29458 之一）
      - 模型对这条样本打出全量 item 分数（长度 item_num=3106）
      - 在这 3106 个 item 里挑高分但不是 target 的 item，作为 hard negatives
4. 若策略包含随机部分，则再从全量 item 空间补随机 negatives（排除正样本与已选负样本）。
5. 将候选 list 打乱顺序，再对 sampled logits 做 CE。

> 注意：hard negative 不是从“训练样本条数（例如 29458）”里选，而是从 item 候选空间（例如 `item_num=3106`）里选。

#### 参数说明

- `--neg_sampling_strategy hard|random|mixed`
  - `hard`：全部负样本来自 hard pool
  - `random`：全部负样本来自全量 item 随机采样
  - `mixed`：hard 和 random 混合
- `--hard_pool_size`
  - hard pool 大小，即从全量 logits 取前多少个高分 item 作为候选池
  - 数值越大越接近“在全量 logits 上挖 hardest negatives”
- `--hard_ratio`
  - 仅在 `mixed` 模式有效，表示负样本中 hard 部分占比
  - 例如 `hard_ratio=0.7` 表示约 70% hard + 30% random

#### 实用建议（本项目 `item_num` 约 3k）

- 稳健默认：`--neg_sampling_strategy mixed --hard_ratio 0.7 --hard_pool_size 1000~2000`
- 更激进：`--neg_sampling_strategy hard --hard_pool_size 1500~3000`
- 若训练震荡：降低 `hard_ratio`（如 0.5）或减小 `hard_pool_size`

### 5.2 训练完成后如何给 RL 使用

1. 确认训练时 `--max_seq_len`、`--hidden_size`、`--dropout`。
2. 在 `rl.py` 对应配置中设置：
   - `--reward_type sasrec`
   - `--cf_path <output_dir>/sasrec_state_dict.pth`
3. 确保 `rl.py` 使用的 `info_file` 与预训练完全同源，否则 SID->id 映射会错位，奖励分数失真。

---

## 6. 训练输出

`output_dir` 下会生成：

- `sasrec_state_dict.pth`：最佳验证指标对应权重（给 `rl.py --cf_path`）
- `best_metrics.json`：最优 epoch 的 val/test 指标
- `train_history.json`：每个 epoch 的训练日志
- `train_config.json`：本次训练参数快照

---

## 7. 在 RL 中接入

训练完成后，在 RL 训练脚本中：

1. `--reward_type sasrec`
2. `--cf_path RL_Reward/SASRec/output/industrial/sasrec_state_dict.pth`
3. 确保 RL 使用的 `info_file` 与 SASRec 训练时一致（SID->id 映射必须同源）

---

## 8. 常见问题

1. **为什么 completion 不在 item_name 时随机赋值？**
   - 这是 `cf_reward` 的既有容错逻辑。建议后续可改成固定低分策略，减少随机性。

2. **历史长度超过 `max_seq_len` 怎么办？**
   - 当前取最近 `max_seq_len` 条，和序列推荐常规设置一致。

3. **list_size 怎么设？**
   - `-1` 表示全量 list（标准 listwise-CE）；`32/64/128` 等表示 sampled listwise-CE，数值越大越接近全量训练。

---

## 9. Reward 一致性评测（新）

为了验证 `cf_reward` 对“同一 prompt 的多 completion”排序是否可靠，新增脚本：

- `eval_reward_consistency.py`

### 9.1 评测思路

对每条评估样本：

1. 使用真实 `history_item_sid` 构造一个候选 completion 集合：
   - 1 个正样本：真实 `item_sid`
   - `N-1` 个负样本：随机 SID（不等于正样本）
2. 用与 `cf_reward` 同口径的方式打分：
   - `history -> forward_eval -> gather(candidate_ids)`
3. 统计目标正样本在候选集中的排序质量：
   - `Hit@K` / `NDCG@K`（候选集内）
   - `MRR`
   - `AUC_pos_vs_neg`（正样本分数高于负样本分数的概率）

如果这些指标明显高于随机基线（例如 `Hit@1` 明显高于 `1/N`），说明 reward 对 completion 排序有实际区分能力。

### 9.2 使用命令

```bash
python RL_Reward/SASRec/eval_reward_consistency.py \
  --eval_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --cf_path /media/rxjqr/wy/MiniOneRec/RL_Reward/SASRec/output/industrial_and_scientific/sasrec_state_dict.pth \
  --num_candidates 16 \
  --eval_topk 1,3,5,10 \
  --output_path /media/rxjqr/wy/MiniOneRec/RL_Reward/SASRec/output/industrial_and_scientific/reward_consistency.json \
  --device cuda
```

### 9.3 关键指标解释

- `random_hit_at_1_baseline = 1 / num_candidates`
  - 例如 `num_candidates=16` 时随机基线是 `0.0625`。
- `Hit@1`
  - 正样本是否排在候选集第一名。
- `MRR`
  - 越高越好，能反映正样本平均排名。
- `AUC_pos_vs_neg`
  - 越接近 1 越好；0.5 约等于随机。
