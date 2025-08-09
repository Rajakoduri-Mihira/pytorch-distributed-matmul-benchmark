import torch
import torch.distributed as dist
import torch.nn as nn
import time
import os
import argparse
from typing import List, Tuple
import enum

class ScalingMode(enum.Enum):
    INDEPENDENT = "independent"          # Each GPU does its own full matmul (2x throughput)
    BATCH_PARALLEL = "batch_parallel"    # Split batch dimension across GPUs (common in ML)
    MATRIX_PARALLEL = "matrix_parallel"  # Split one large matrix across GPUs

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size
    else:
        print("Running in single GPU mode")
        return 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def calculate_tflops(matrix_size: int, time_seconds: float, num_ops: int = 1) -> float:
    """Calculate TFLOPS. num_ops is number of matrix multiplications performed."""
    flops = 2.0 * (matrix_size ** 3) * num_ops
    tflops = (flops / time_seconds) / 1e12
    return tflops

def benchmark_independent(matrix_size: int, dtype: torch.dtype, device: str, rank: int,
                         num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float]:
    """Each GPU does its own independent matmul - true 2x throughput"""
    # Each GPU gets different random matrices
    torch.manual_seed(rank)  # Different seed per GPU
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup_iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_iterations):
        C = torch.matmul(A, B)
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    total_time = total_time_ms / 1000.0
    avg_time = total_time / num_iterations
    
    # Each GPU did one matmul per iteration
    tflops_per_gpu = calculate_tflops(matrix_size, avg_time, num_ops=1)
    
    return avg_time, tflops_per_gpu

def benchmark_batch_parallel(matrix_size: int, batch_size: int, dtype: torch.dtype, 
                            device: str, rank: int, world_size: int,
                            num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float]:
    """Batch parallel: Each GPU processes different batches (common in ML training)"""
    # Each GPU gets a portion of the total batch
    local_batch_size = batch_size // world_size
    
    # Create batched matrices - each GPU has different data
    torch.manual_seed(rank)
    A = torch.randn(local_batch_size, matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(local_batch_size, matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup_iterations):
        C = torch.bmm(A, B)  # Batched matrix multiply
        # In real training, we'd do allreduce on gradients here, not on C
        # But for benchmark, we'll measure the forward pass
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_iterations):
        C = torch.bmm(A, B)
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    total_time = total_time_ms / 1000.0
    avg_time = total_time / num_iterations
    
    # Each GPU did local_batch_size matmuls per iteration
    tflops_per_gpu = calculate_tflops(matrix_size, avg_time, num_ops=local_batch_size)
    
    return avg_time, tflops_per_gpu

def benchmark_matrix_parallel(matrix_size: int, dtype: torch.dtype, device: str, 
                             rank: int, world_size: int,
                             num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float]:
    """Matrix parallel: Split one large matrix multiplication across GPUs"""
    if world_size == 1:
        return benchmark_independent(matrix_size, dtype, device, rank, num_iterations, warmup_iterations)
    
    # For simplicity, split matrix B column-wise
    # A is replicated, B is split, result C needs to be gathered
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Each GPU gets a portion of B's columns
    cols_per_gpu = matrix_size // world_size
    start_col = rank * cols_per_gpu
    end_col = start_col + cols_per_gpu if rank < world_size - 1 else matrix_size
    
    B_local = torch.randn(matrix_size, end_col - start_col, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup_iterations):
        # Local computation
        C_local = torch.matmul(A, B_local)
        
        # In real scenario, we'd gather C_local from all GPUs
        if dist.is_initialized():
            # Allocate space for gather (simplified - in practice would optimize this)
            gathered = [torch.zeros_like(C_local) for _ in range(world_size)]
            dist.all_gather(gathered, C_local)
    
    torch.cuda.synchronize()
    
    # Benchmark
    compute_time = 0.0
    comm_time = 0.0
    
    for _ in range(num_iterations):
        # Time compute
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        C_local = torch.matmul(A, B_local)
        end_event.record()
        torch.cuda.synchronize()
        
        compute_time += start_event.elapsed_time(end_event)
        
        # Time communication
        if dist.is_initialized():
            gathered = [torch.zeros_like(C_local) for _ in range(world_size)]
            start_event.record()
            dist.all_gather(gathered, C_local)
            end_event.record()
            torch.cuda.synchronize()
            comm_time += start_event.elapsed_time(end_event)
    
    avg_compute_time = (compute_time / 1000.0) / num_iterations
    avg_comm_time = (comm_time / 1000.0) / num_iterations
    avg_total_time = avg_compute_time + avg_comm_time
    
    # Each GPU does 1/world_size of the total work
    # But together they complete one full matrix multiplication
    tflops_per_gpu = calculate_tflops(matrix_size, avg_total_time * world_size, num_ops=1)
    
    return avg_total_time, tflops_per_gpu

def run_benchmarks(rank: int, world_size: int, matrix_sizes: List[int], 
                   dtype: torch.dtype, mode: ScalingMode, 
                   num_iterations: int, warmup_iterations: int):
    device = f'cuda:{rank % torch.cuda.device_count()}'
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Matrix Multiplication Scaling Benchmark")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Mode: {mode.value}")
        print(f"  - Number of GPUs: {world_size}")
        print(f"  - Data type: {dtype}")
        print(f"  - Iterations per test: {num_iterations}")
        print(f"  - Warmup iterations: {warmup_iterations}")
        print(f"{'='*70}\n")
    
    for size in matrix_sizes:
        if rank == 0:
            bytes_per_element = 4 if dtype == torch.float32 else 2
            dtype_name = 'float32' if dtype == torch.float32 else ('float16' if dtype == torch.float16 else 'bfloat16')
            print(f"\nBenchmarking {size}x{size} matrix multiplication:")
            print(f"  - Memory per matrix: {size * size * bytes_per_element / (1024**3):.2f} GB ({dtype_name})")
            print(f"  - Mode: {mode.value}")
        
        try:
            if mode == ScalingMode.INDEPENDENT:
                avg_time, tflops_per_gpu = benchmark_independent(
                    size, dtype, device, rank, num_iterations, warmup_iterations
                )
                comm_time = 0.0
            elif mode == ScalingMode.BATCH_PARALLEL:
                batch_size = 4  # Total batch size across all GPUs
                avg_time, tflops_per_gpu = benchmark_batch_parallel(
                    size, batch_size, dtype, device, rank, world_size, 
                    num_iterations, warmup_iterations
                )
                comm_time = 0.0
            elif mode == ScalingMode.MATRIX_PARALLEL:
                avg_time, tflops_per_gpu = benchmark_matrix_parallel(
                    size, dtype, device, rank, world_size,
                    num_iterations, warmup_iterations
                )
                comm_time = 0.0  # Already included in avg_time
            
            # Gather results
            time_tensor = torch.tensor([avg_time], device=device)
            tflops_tensor = torch.tensor([tflops_per_gpu], device=device)
            
            if dist.is_initialized():
                dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
                # For TFLOPS, sum for independent, avg for others
                if mode == ScalingMode.INDEPENDENT:
                    dist.all_reduce(tflops_tensor, op=dist.ReduceOp.SUM)
                else:
                    dist.all_reduce(tflops_tensor, op=dist.ReduceOp.AVG)
            
            if rank == 0:
                print(f"\nResults for {size}x{size}:")
                print(f"  - Average time per operation: {time_tensor.item()*1000:.3f} ms")
                
                if mode == ScalingMode.INDEPENDENT:
                    print(f"  - TFLOPS per GPU: {tflops_per_gpu:.2f}")
                    print(f"  - Total system TFLOPS: {tflops_tensor.item():.2f}")
                    print(f"  - Scaling efficiency: {(tflops_tensor.item() / (tflops_per_gpu * world_size)) * 100:.1f}%")
                elif mode == ScalingMode.BATCH_PARALLEL:
                    print(f"  - TFLOPS per GPU: {tflops_per_gpu:.2f}")
                    total_tflops = tflops_per_gpu * world_size
                    print(f"  - Total system TFLOPS: {total_tflops:.2f}")
                    print(f"  - Processing {4} total batches across {world_size} GPU(s)")
                elif mode == ScalingMode.MATRIX_PARALLEL:
                    print(f"  - TFLOPS per GPU (portion): {tflops_per_gpu:.2f}")
                    print(f"  - Effective system TFLOPS: {tflops_tensor.item():.2f}")
                    print(f"  - Each GPU processes 1/{world_size} of the matrix")
                
                # Show actual work done
                if mode == ScalingMode.INDEPENDENT:
                    total_flops = 2.0 * size**3 * world_size  # Each GPU does full matmul
                elif mode == ScalingMode.BATCH_PARALLEL:
                    total_flops = 2.0 * size**3 * 4  # 4 batches total
                else:  # MATRIX_PARALLEL
                    total_flops = 2.0 * size**3  # One matmul split across GPUs
                
                actual_total_tflops = (total_flops / time_tensor.item()) / 1e12
                print(f"  - Actual TFLOPS (total FLOPs / time): {actual_total_tflops:.2f}")
                
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"\n  ERROR: Out of memory for {size}x{size} matrices")
        except Exception as e:
            if rank == 0:
                print(f"\n  ERROR: {str(e)}")
        
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()

def main():
    parser = argparse.ArgumentParser(description='Matrix Multiplication Scaling Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4096, 8192, 16384],
                        help='Matrix sizes to benchmark (default: 4096 8192 16384)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations per benchmark (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for matrices (default: bfloat16)')
    parser.add_argument('--mode', type=str, default='independent',
                        choices=['independent', 'batch_parallel', 'matrix_parallel'],
                        help='Scaling mode (default: independent)')
    
    args = parser.parse_args()
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    mode = ScalingMode(args.mode)
    
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"    SMs: {props.multi_processor_count}")
    
    try:
        run_benchmarks(rank, world_size, args.sizes, dtype, mode, args.iterations, args.warmup)
    finally:
        cleanup_distributed()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Benchmark completed!")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()