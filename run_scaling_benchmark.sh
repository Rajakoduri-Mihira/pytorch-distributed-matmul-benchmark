#!/bin/bash

NUM_GPUS=${1:-2}
MODE=${2:-independent}
DTYPE=${3:-bfloat16}

echo "Matrix Multiplication Scaling Benchmark"
echo "  GPUs: $NUM_GPUS"
echo "  Mode: $MODE (independent, batch_parallel, matrix_parallel)"
echo "  Data type: $DTYPE"
echo ""

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running in single GPU mode..."
    python3 matmul_scaling_benchmark.py \
        --sizes 4096 8192 16384 \
        --iterations 50 \
        --warmup 10 \
        --mode $MODE \
        --dtype $DTYPE
else
    echo "Running in distributed mode with $NUM_GPUS GPUs..."
    python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29503 \
        matmul_scaling_benchmark.py \
        --sizes 4096 8192 16384 \
        --iterations 50 \
        --warmup 10 \
        --mode $MODE \
        --dtype $DTYPE
fi