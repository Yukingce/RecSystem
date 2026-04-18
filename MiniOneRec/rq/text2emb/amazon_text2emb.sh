accelerate launch --num_processes 2 amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root /media/rxjqr/wy/MiniOneRec/data/Amazon18/Industrial_and_Scientific \
    --plm_name qwen \
    --plm_checkpoint Qwen/Qwen3-Embedding-4B

# --max_sent_len： 含义：每条文本 token 化后的最大长度上限。
# 超过 max_sent_len：截断（truncation=True）。
# 不足 max_sent_len：会 pad 到当前 batch 内的最长长度（padding=True）。
# 影响：
# 越大：保留更多语义细节，但注意力计算和激活显存显著增加。
# 越小：更省显存更快，但可能丢失长文本信息。