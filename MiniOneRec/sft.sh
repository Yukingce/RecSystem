export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# W&B credentials: only applied to this script's torchrun command.
# Please replace with your own values before running.
WANDB_API_KEY="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"
WANDB_ENTITY="yuking-sjtu"

if [ "${WANDB_API_KEY}" = "your_wandb_api_key" ] || [ "${WANDB_ENTITY}" = "your_wandb_entity" ]; then
    echo "Please set WANDB_API_KEY and WANDB_ENTITY in sft.sh before running."
    exit 1
fi

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon18/Industrial_and_Scientific/minionerec_train_data/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    WANDB_API_KEY="${WANDB_API_KEY}" WANDB_ENTITY="${WANDB_ENTITY}" torchrun --nproc_per_node 2 \
            sft.py \
            --base_model Qwen/Qwen3-1.7B \
            --batch_size 64 \
            --micro_batch_size 2 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output \
            --wandb_project MiniOneRec-SFT \
            --wandb_run_name indsci_qwen7b_bs64_seed42 \
            --wandb_entity ${WANDB_ENTITY} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.json \
            --item_meta_path /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item.json \
            --freeze_LLM False
done


# batch_size：你期望的全局有效 batch size（逻辑上的每步训练样本数）。
# micro_batch_size：每张卡、每次前向/反向实际喂进去的小 batch，主要受显存限制。


########################################

# export NCCL_IB_DISABLE=1
# 禁用 NCCL 的 IB/RoCE 通信，通常用于机器网络环境对 IB 支持不好时避免卡住。

# for category in "Industrial_and_Scientific"; do ... done
# 目前只跑一个品类；注释里提到还可以换成 Office_Products 等。

# 自动找数据文件（第 4-7 行）
# 训练集：./data/Amazon/train/${category}*11.csv
# 验证集：./data/Amazon/valid/${category}*11.csv
# 测试集：./data/Amazon/test/${category}*11.csv
# 信息文件：./data/Amazon/info/${category}*.txt
# 用 ls + 通配符匹配，适合文件名包含版本后缀的情况。

# echo ...（第 8 行）
# 启动训练前把匹配到的路径打印出来，方便确认读的是不是你想要的数据。

# torchrun --nproc_per_node 8 sft.py ...（第 10 行开始）
# 用 8 个进程（通常对应 8 张 GPU）执行 sft.py。关键参数：
# --base_model your_model_path：底座模型路径（你需要改成真实路径）
# --batch_size 1024：全局 batch size
# --micro_batch_size 16：单卡/单步微批大小（配合梯度累积达到全局 batch）
# --train_file / --eval_file：训练和验证文件
# --output_dir output_dir/xxx：模型输出目录（建议改成含时间或类别名）
# --wandb_project / --wandb_run_name：W&B 记录配置
# --category：当前品类名，传给训练脚本用于条件逻辑/日志
# --train_from_scratch False：不是从零训练，而是基于 base_model 微调
# --seed 42：随机种子
# --sid_index_path / --item_meta_path：物品索引和元信息文件
# --freeze_LLM False：不冻结 LLM 参数（即参与训练）
