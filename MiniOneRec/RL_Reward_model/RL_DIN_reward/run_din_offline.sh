#!/bin/bash

set -e

TRAIN_FILE=./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/Industrial_and_Scientific_5_2016-10-2018-11.csv
VALID_FILE=./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv
OUTPUT_DIR=./RL_DIN_reward/checkpoints/indsci

python -m RL_DIN_reward.train_din \
  --train_file "${TRAIN_FILE}" \
  --valid_file "${VALID_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 10 \
  --batch_size 512 \
  --learning_rate 1e-3 \
  --max_seq_len 50 \
  --num_neg 2
