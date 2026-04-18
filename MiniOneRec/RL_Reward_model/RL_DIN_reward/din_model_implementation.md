# DIN model implementation for reward function
_Exported on 4/17/2026 at 22:36:33 GMT+8 from Cursor (3.1.15)_

---

**User**

@rl.py:256-265 我想将这个奖励函数，换成推荐系统中的DIN模型，请问可以实现吗？不通过规则进行奖励。将用户的输入交互序列SID作为DIN的输入，但是这个交互的SID序列可能不一样长

---

**Cursor**

我先快速看一下 `rl.py` 里奖励函数的定义和调用方式，确认把 `reward_type` 替换成 DIN 需要接入哪些输入与输出，然后给你一个可落地的改造方案（不改代码，只给实现路径）。

可以实现，而且你现在这套框架**非常适合**把规则奖励替换成 DIN 打分奖励。

你当前的奖励函数接口是 `reward_func(prompts, completions, **kwargs)`，在训练时被统一调用，这一点在 `minionerec_trainer.py` 里已经支持自定义函数。你只需要新增一个 `din_reward`，然后在 `reward_type` 分支里接进去即可（思路上）。

```939:960:minionerec_trainer.py
rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
for i, (reward_func, reward_processing_class) in enumerate(
    zip(self.reward_funcs, self.reward_processing_classes)
):
    ...
    else:
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
```

---

你提到的“SID 序列长度不一样”也是 DIN 的常规场景，做法是：

- 把 `history SID` 序列转成 id 列表（变长）
- `pad` 到同一长度
- 同时构造 `mask`（或 `seq_len`）
- DIN 的注意力层只对有效位置计算（mask 掉 padding）

你现在已有类似变长处理雏形（在 `cf_reward` 中按 `len_seq` 补齐），DIN 只是在此基础上增加 attention pooling。

```237:251:rl.py
len_lis = []
history_ids = []
for his in history_list:
    his = [item2id[elm] for elm in his]
    len_lis.append(len(his))
    if len(his) < len_seq: 
        his = his + [item_num] * (len_seq - len(his))
    history_ids.append(his)

seq = torch.LongTensor(history_ids).to(device)
pred = torch.LongTensor(pred_ids).to(device)    

with torch.no_grad():
    predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
```

---

### 你可以按这个路径改（设计建议）

- 训练一个 DIN（离线先训好，冻结参数）
  - 输入：`history_sid_seq` + `candidate_sid`
  - 标签：点击/下一个 item（1/0）
- 在 RL 阶段加载 DIN 权重，定义 `din_reward(prompts, completions)`：
  - 从 `prompt2history` 取 `history_str`
  - `history_str.split("::")` 得到 SID 序列
  - `completion` 作为 candidate SID
  - 非法 SID 给小奖励或随机兜底
  - 输出 DIN 的 `sigmoid(logit)` 或 logit 作为 reward
- 在分支中增加：
  - `elif reward_type == "din": reward_fun = din_reward`

你当前 `SidDataset` 里 `history_str` 就是 SID 按 `::` 拼接，天然可直接喂给 DIN 前处理：

```362:379:data.py
def get_history(self, row):
    row['history_item_sid'] = eval(row['history_item_sid'])
    ...
    history_str = "::".join(row["history_item_sid"])
    ...
    return {"input": f"...",
            "output": target_item + "\n",
            "history_str": history_str,
            "dedup": target_item_sid == last_history_item_sid}
```

---

### 注意点（很关键）

- DIN 奖励是**可实现**的，但建议先冻结 DIN，避免 RL 同时训练 LLM+DIN 导致不稳定。
- 奖励尺度建议做裁剪或归一化（如 `[-1,1]` 或 z-score），否则会影响 GRPO 优化稳定性。
- 负样本问题：如果 DIN 训练时只用正样本，reward 区分度会弱，建议离线训练时加入 hard negatives。
- OOV SID（模型生成了不存在的 SID）要有明确定义：通常给 0 或较小负值。

---

如果你愿意，我下一步可以直接给你一版**最小改动的 `din_reward` 伪代码模板**（贴合你这个 `rl.py` 的变量命名和调用方式），你可以直接粘进去改。

---

**User**

帮我实现这个功能，包括完整的利用现有数据，进行DIN的离线训练。以及在RL部署，进行RL的奖励。DIN的相关代码放在一个新的文件夹RL_DIN_reward中。并提供一个文档，介绍如何离线训练，使用的数据集是什么，以及如何部署到RL训练中

---

**Cursor**

我会先梳理现有数据字段和 `rl.py` 的奖励接入点，然后新建 `RL_DIN_reward` 目录实现：DIN离线训练脚本、模型定义、数据处理与推理接口，最后把 `rl.py` 接入 `reward_type="din"` 并补一份使用文档。先快速扫描当前仓库里可复用的CF/SASRec相关实现。

我已经确认奖励接入点了，下一步会直接实现一套独立的 DIN 模块（数据读取/负采样/训练/推理），并在 `rl.py` 增加 `reward_type=din`。我先定位现有训练集CSV格式，确保离线训练脚本直接复用你当前数据。

我已经确认了数据格式（`history_item_sid` 是可解析的序列字符串，`item_sid` 是目标 SID），接下来开始落地代码：先在 `RL_DIN_reward` 新增 DIN 模型+训练脚本+推理封装，再改 `rl.py` 增加 `reward_type=din` 与 `din_ckpt_path` 参数，最后补一份中文文档说明离线训练与RL接入。

核心代码已完成，我现在做一次快速自检（语法编译 + 关键参数检索），确保 `DIN离线训练` 与 `RL接入` 这两条链路都能直接跑。

已完成你要的完整功能，包含：

- 新增 `RL_DIN_reward` 文件夹，实现 DIN 离线训练全流程
- 在 `rl.py` 中接入 `reward_type="din"` 奖励
- 提供中文文档说明数据集、训练、部署步骤

## 我做了哪些改动

### 1) 新增 DIN 模块目录 `RL_DIN_reward`

新增文件：

- `RL_DIN_reward/din_model.py`  
  - 实现 DIN 主体：`item embedding + attention pooling + prediction MLP`
  - 支持变长序列（通过 mask 屏蔽 padding）

- `RL_DIN_reward/data_utils.py`  
  - 解析 `history_item_sid`（支持 list 字符串等格式）
  - 从现有 CSV 构建 `sid2id` 词表
  - 负采样数据构建（正样本 + neg 样本）
  - `collate_fn` 动态 padding，适配不同长度 SID 序列

- `RL_DIN_reward/train_din.py`  
  - 离线训练入口（直接使用你当前 MiniOneRec 的 train/valid CSV）
  - 训练完成保存 `din_reward_best.pt`（包含模型参数+词表+配置）

- `RL_DIN_reward/inference.py`  
  - `DINRewardModel` 推理封装
  - 提供 `score_batch(history_sid_lists, candidate_sids)`，返回概率分数用于 RL reward

- `RL_DIN_reward/__init__.py`

- `RL_DIN_reward/run_din_offline.sh`  
  - 一键离线训练脚本模板

- `RL_DIN_reward/README.md`  
  - 中文使用文档（数据来源、训练、部署、奖励定义）

---

### 2) 在 `rl.py` 接入 DIN 奖励

主要改动：

- 新增导入：
  - `from RL_DIN_reward.inference import DINRewardModel`

- `train()` 新增参数：
  - `din_ckpt_path: str = ""`

- 增加 DIN 初始化逻辑：
  - 当 `reward_type == "din"` 时加载 `DINRewardModel.from_checkpoint(din_ckpt_path, device)`

- 新增 `din_reward(prompts, completions)`：
  - 从 `prompt2history` 拿历史 SID 序列
  - 将 completion 作为候选 SID
  - 调用 DIN 输出 `sigmoid` 概率作为 reward

- 增加 reward 分支：
  - `elif reward_type == "din": reward_fun = din_reward`

---

## 你现在怎么用

### A. 先离线训练 DIN

在项目根目录执行（示例）：

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

输出 checkpoint：

- `./RL_DIN_reward/checkpoints/indsci/din_reward_best.pt`

---

### B. 在 RL 训练中启用 DIN 奖励

运行 `rl.py` 时加入：

- `--reward_type din`
- `--din_ckpt_path ./RL_DIN_reward/checkpoints/indsci/din_reward_best.pt`

例如（只示核心参数）：

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

## 关于你最关心的“变长 SID 序列”

已处理，机制是：

- 每条样本保留最近 `max_seq_len`
- batch 内动态 padding
- attention 使用 `history_mask`，padding 位不参与计算

所以不同长度 SID 序列可以直接训练和推理。

---

如果你希望，我可以下一步再帮你把 `rl.sh` 也改成可直接切换 `reward_type=din` 的版本（加上 `DIN_CKPT_PATH` 变量和完整命令参数）。

