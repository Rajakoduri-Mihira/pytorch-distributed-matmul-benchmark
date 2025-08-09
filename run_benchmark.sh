#!/bin/bash

NUM_GPUS=${1:-1}
DTYPE=${2:-bfloat16}

echo "Starting distributed matrix multiplication benchmark with $NUM_GPUS GPU(s)"
echo "Data type: $DTYPE"
echo ""

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running in single GPU mode..."
    python3 matmul_benchmark.py --sizes 4096 8192 16384 --iterations 50 --warmup 10 --dtype $DTYPE
else
    echo "Running in distributed mode with $NUM_GPUS GPUs..."
    python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        matmul_benchmark.py \
        --sizes 4096 8192 16384 \
        --iterations 50 \
        --warmup 10 \
        --dtype $DTYPE
fi