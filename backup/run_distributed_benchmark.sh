#!/bin/bash

NUM_GPUS=${1:-2}
MODE=${2:-data_parallel}
DTYPE=${3:-bfloat16}

echo "Starting distributed matrix multiplication benchmark"
echo "  GPUs: $NUM_GPUS"
echo "  Mode: $MODE"
echo "  Data type: $DTYPE"
echo ""

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running in single GPU mode..."
    python3 matmul_distributed_benchmark.py \
        --sizes 4096 8192 16384 \
        --iterations 50 \
        --warmup 10 \
        --mode $MODE \
        --dtype $DTYPE
else
    echo "Running in distributed mode with $NUM_GPUS GPUs..."
    python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29501 \
        matmul_distributed_benchmark.py \
        --sizes 4096 8192 16384 \
        --iterations 50 \
        --warmup 10 \
        --mode $MODE \
        --dtype $DTYPE
fi