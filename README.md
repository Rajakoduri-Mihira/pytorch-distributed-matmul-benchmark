# PyTorch Distributed Matrix Multiplication Benchmark

## Files

### Main Benchmarks
- `matmul_benchmark.py` - Basic matrix multiplication benchmark
- `matmul_scaling_benchmark.py` - Advanced scaling benchmark with different parallelism modes
- `run_benchmark.sh` - Simple launcher for basic benchmark
- `run_scaling_benchmark.sh` - Launcher for scaling benchmark

### Backup Folder
Contains experimental versions with communication overlap testing.

## Quick Start

### 1. Basic Benchmark
```bash
# Single GPU
./run_benchmark.sh 1              # BFloat16 (default)
./run_benchmark.sh 1 float32      # Float32
./run_benchmark.sh 1 float16      # Float16

# Multi-GPU (independent, no communication)
./run_benchmark.sh 2              # 2 GPUs
```

### 2. Scaling Benchmark
```bash
# Independent (2x throughput with 2 GPUs)
./run_scaling_benchmark.sh 2 independent bfloat16

# Batch Parallel (ML training style)
./run_scaling_benchmark.sh 2 batch_parallel bfloat16

# Matrix Parallel (split one matrix)
./run_scaling_benchmark.sh 2 matrix_parallel bfloat16
```

## Results Summary (RTX 6000 Ada, 16k×16k matrices)

| Mode | GPUs | TFLOPS | Scaling Efficiency |
|------|------|--------|-------------------|
| Single GPU | 1 | ~140 | Baseline |
| Independent | 2 | ~294 | ~100% (perfect) |
| Batch Parallel | 2 | ~237 | ~85% |
| Matrix Parallel | 2 | ~141 | N/A (same work) |

## Key Insights

1. **BFloat16 is ~5x faster than Float32** on modern GPUs
2. **Independent scaling achieves near-perfect 2x speedup** with 2 GPUs
3. **Batch parallel** (realistic ML training) achieves ~85% scaling efficiency
4. **Matrix parallel** doesn't increase throughput (splits one operation)

## Matrix Sizes Tested
- 4k×4k (0.14 TFLOPS of work)
- 8k×8k (1.10 TFLOPS of work)
- 16k×16k (8.80 TFLOPS of work)