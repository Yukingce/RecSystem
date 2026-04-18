#!/bin/bash
#
# RQ-KMeans Constrained Training Script
#

# Default parameters
DATASET="Industrial_and_Scientific"
ROOT="../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy"  #"../data/Amazon18/$DATASET"
OUTPUT_ROOT="./output/Industrial_and_Scientific_rq_kmeans_constrained"
K=256
L=3
MAX_ITER=100
SEED=42

# K=256：每一层聚类的簇数（codebook size）。
# 即每层会把向量分到 256 个中心里。

# L=3：RQ（Residual Quantization）的层数。
# 你现在是 3 层逐级量化：第 1 层量化原向量残差，第 2 层量化剩余残差，第 3 层继续量化。

# MAX_ITER=100：每层 KMeans 的最大迭代次数。
# 单层聚类最多迭代 100 次，提前收敛会更早停止。


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --l) L="$2"; shift 2 ;;
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Dataset: $DATASET"
echo "K=$K, L=$L"
echo "Input root: $ROOT"
if [[ -n "$OUTPUT_ROOT" ]]; then
    echo "Output root: $OUTPUT_ROOT"
else
    echo "Output root: (default from Python logic)"
fi

PY_ARGS=(
    --dataset "$DATASET"
    --root "$ROOT"
    --k "$K"
    --l "$L"
    --max_iter "$MAX_ITER"
    --seed "$SEED"
    --verbose
)

if [[ -n "$OUTPUT_ROOT" ]]; then
    PY_ARGS+=(--output_root "$OUTPUT_ROOT")
fi

python rqkmeans_constrained.py "${PY_ARGS[@]}"
