#!/bin/bash

python test_reward_consistency.py \
  --test_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/minionerec_train_data/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --cf_path /media/rxjqr/wy/MiniOneRec/RL_Reward/SASRec/output/industrial_and_scientific/Mixed_List_size_1024/sasrec_state_dict.pth \
  --num_candidates 50 \
  --eval_topk 1,3,5,10,20,50 \
  --output_path output/industrial_and_scientific/Mixed_List_size_1024/reward_consistency.json \
  --device cuda