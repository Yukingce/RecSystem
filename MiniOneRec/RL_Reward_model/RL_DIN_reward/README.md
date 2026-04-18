# RL_DIN_reward 使用说明

这个目录实现了：

- DIN 离线训练（基于现有 MiniOneRec CSV）
- DIN 奖励推理封装（供 `rl.py` 调用）
- RL 中 `reward_type=din` 的奖励接入

---

## 1. 使用的数据集

离线训练脚本直接复用你当前的 MiniOneRec 格式 CSV：

- `train/*.csv`
- `valid/*.csv`（可选，用于验证）

关键字段：

- `history_item_sid`：用户历史 SID 序列（变长）
- `item_sid`：下一条真实目标 SID

以当前仓库为例：

- `data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`

---

## 2. 目录文件说明

- `din_model.py`：DIN 模型定义（含 attention pooling）
- `data_utils.py`：SID 解析、词表构建、负采样、DataLoader collate
- `train_din.py`：DIN 离线训练入口
- `inference.py`：`DINRewardModel` 推理封装，供 RL 奖励调用

---

## 3. DIN 离线训练

在项目根目录运行：

```bash
python -m RL_DIN_reward.train_din \
  --train_file ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --valid_file ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir ./RL_DIN_reward/checkpoints/indsci \
  --epochs 10 \
  --batch_size 512 \
  --learning_rate 1e-3 \
  --max_seq_len 50 \
  --num_neg 2
```

输出文件：

- `din_reward_best.pt`：DIN checkpoint（包含模型参数、SID词表、模型配置）
- `din_train_args.json`：训练参数记录

---

## 4. 部署到 RL 训练

`rl.py` 已支持：

- `--reward_type din`
- `--din_ckpt_path <DIN checkpoint 路径>`

示例（只展示核心参数）：

```bash
python rl.py \
  --model_path ./sft_output/final_checkpoint \
  --train_file ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --reward_type din \
  --din_ckpt_path ./RL_DIN_reward/checkpoints/indsci/din_reward_best.pt \
  --output_dir ./rl_output_din
```

---

## 5. 变长 SID 序列如何处理

DIN 对 `history_item_sid` 使用如下策略：

- 读取原始变长序列
- 按 `max_seq_len` 截断（保留最近行为）
- batch 内动态 padding
- 使用 `history_mask` 屏蔽 padding 位，不参与 attention

因此你提到的“交互 SID 序列长度不一样”是原生支持的。

---

## 6. 奖励定义

RL 里对每个候选 completion：

1. 从 `prompt2history` 取历史 SID 序列；
2. 将 completion 作为候选 SID；
3. 送入 DIN 输出点击概率 `sigmoid(logit)`；
4. 该概率直接作为 reward。

这实现了“非规则奖励”，即不再依赖 exact match 或 NDCG 规则分值。
