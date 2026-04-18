#!/bin/bash

# 检查是否传入参数
if [ -z "$1" ]; then
    echo "Usage: source temp_switch_cuda.sh <cuda_version>"
    echo "Example: source temp_switch_cuda.sh 11.3"
    return 1
fi

# 定义目标 CUDA 版本
CUDA_VERSION=$1
CUDA_PATH=/usr/local/cuda-$CUDA_VERSION

# 检查目标 CUDA 版本是否存在
if [ ! -d $CUDA_PATH ]; then
    echo "Error: CUDA version $CUDA_VERSION is not installed at $CUDA_PATH"
    return 1
fi

# 切换 CUDA 版本
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
nvcc -V

# source  or .


