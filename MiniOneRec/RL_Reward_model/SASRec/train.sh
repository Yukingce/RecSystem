#!/bin/bash

python train_sasrec.py \
  --train_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --valid_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --test_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --output_dir output/industrial_and_scientific/Mixed_List_size_1024_hard_pool_2000 \
  --max_seq_len 10 \
  --hidden_size 32 \
  --dropout 0.45 \
  --list_size 1024 \
  --neg_sampling_strategy mixed \
  --hard_pool_size 2000 \
  --hard_ratio 0.1 \
  --batch_size 256 \
  --epochs 300 \
  --lr 5e-4 \
  --weight_decay 5e-5 \
  --early_stop 20 \
  --device cuda


# --neg_sampling_strategy：hard | random | mixed（默认 hard）
# --hard_pool_size：从当前 logits 的 TopK 中挖 hard negatives 的候选池大小（默认 1024）
# --hard_ratio：mixed 模式下 hard negative 占比（默认 0.8）

# sampled listwise (list_size > 0 且 < item_num) 的负采样逻辑改为：
# hard：全部负样本从当前 batch 的高分错误项（TopK logits）中选
# random：全部随机负样本
# mixed：一部分 hard + 一部分 random


# --list_size -1 全量 listwise
# 在你当前实现里，模型前向仍会先算全量 item_num logits，再 gather 到 list；
# 所以 list_size 主要影响“损失用到多少候选”和训练信号，不是线性节省前向计算。


# v1版本 候选是随机采样出来的，具体规则是：
        # 对每条样本先拿到真实标签 target_item（正样本）
        # 设 neg_count = list_size - 1
        # 在 [0, item_num-1] 里随机采 neg_count 个不重复负样本，且都满足 neg != target_item
        # 把候选拼成：[target_item] + negatives
        # 再随机打乱顺序（防止正样本总在固定位置）
        # 用这个候选列表去 gather 对应 logits，再做 CE
        # 也就是说，每条样本的候选 list 是：
        #         1 个正样本 + (list_size-1) 个随机负样本
        #         每个 step 都会重新随机，训练过程中会看到不同负样本组合

# v2版本 “hard negative”策略（比如优先采当前模型高分但错误的 item），通常比纯随机负样本更有效。



