#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

# W&B credentials: only applied to this script's accelerate command.
# Please replace with your own values before running.
WANDB_API_KEY="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"
WANDB_ENTITY="yuking-sjtu"
BASE_MODEL_PATH="./sft_output/final_checkpoint"
CKPT_PATH=""
# true: resume optimizer/scheduler state as well; false: only load model weights from CKPT_PATH
RESUME_TRAINING_STATE="false"

if [ "${WANDB_API_KEY}" = "your_wandb_api_key" ] || [ "${WANDB_ENTITY}" = "your_wandb_entity" ]; then
    echo "Please set WANDB_API_KEY and WANDB_ENTITY in rl.sh before running."
    exit 1
fi

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/info/${category}*.txt)
    if [ -d "${CKPT_PATH}" ]; then
        model_path="${CKPT_PATH}"
        if [ "${RESUME_TRAINING_STATE}" = "true" ]; then
            resume_arg="--resume_from_checkpoint ${CKPT_PATH}"
            echo "Found checkpoint, resume model + training state from ${CKPT_PATH}"
        else
            resume_arg=""
            echo "Found checkpoint, load model weights only from ${CKPT_PATH}"
        fi
    else
        model_path="${BASE_MODEL_PATH}"
        resume_arg=""
        echo "Checkpoint not found, start from ${BASE_MODEL_PATH}"
    fi

    WANDB_API_KEY="${WANDB_API_KEY}" WANDB_ENTITY="${WANDB_ENTITY}" HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --multi_gpu \
                                    --num_processes 2 --main_process_port 29503 \
                                    rl.py \
                        --model_path "${model_path}" \
                        ${resume_arg} \
                        --train_batch_size 32 \
                        --eval_batch_size 64 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 4 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 8 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ./rl_output/ranking \
                        --wandb_project MiniOneRec-RL \
                        --wandb_entity ${WANDB_ENTITY} \
                        --wandb_run_name minionerec_rl_qwen3_1.7b_ranking_seed42 \
                        --sid_index_path /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.json \
                        --item_meta_path /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item.json \
                        --cf_path /media/rxjqr/wy/MiniOneRec/RL_Reward/SASRec/output/industrial_and_scientific/Mixed_List_size_1024/sasrec_state_dict.pth
done

# 1) 脚本流程
# export NCCL_IB_DISABLE=1：禁用 NCCL 的 IB/RoCE 通信（通常用于规避某些网络/驱动问题）。
# for category in "Industrial_and_Scientific"：当前只跑一个品类，但写成循环方便扩展多类目。
# 三个 ls 动态匹配数据路径：
# 训练集：train/${category}*.csv
# 验证集：valid/${category}*11.csv
# 信息文件：info/${category}*.txt
# 启动命令：
# HF_ENDPOINT=https://hf-mirror.com：临时为本次命令设置 Hugging Face 镜像。
# accelerate launch --config_file ./config/zero2_opt.yaml --num_processes 8：8 进程分布式训练（一般对应 8 张 GPU）。
# 训练脚本是 rl.py，后面传入一系列 RL 训练参数。
# 2) 关键训练参数解读
# 训练规模相关：
# --train_batch_size 64
# --eval_batch_size 128
# --gradient_accumulation_steps 2
# --num_train_epochs 2
# 生成/RL相关：
# --reward_type ranking
# --num_generations 16
# --beam_search True
# --temperature 1.0
# --sync_ref_model True
# --beta 1e-3
# 训练控制：
# --eval_step 0.0999（看起来像按比例触发评估）
# --test_during_training False
# --dynamic_sampling False
# --dapo False