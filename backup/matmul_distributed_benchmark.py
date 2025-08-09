import torch
import torch.distributed as dist
import torch.nn as nn
import time
import os
import argparse
from typing import List, Tuple
import enum

class BenchmarkMode(enum.Enum):
    INDEPENDENT = "independent"  # Each GPU does its own matmul (no communication)
    DATA_PARALLEL = "data_parallel"  # Simulate data parallel training with allreduce
    MODEL_PARALLEL = "model_parallel"  # Split matrices across GPUs

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

def calculate_tflops(matrix_size: int, time_seconds: float) -> float:
    flops = 2.0 * (matrix_size ** 3)
    tflops = (flops / time_seconds) / 1e12
    return tflops

def benchmark_independent(matrix_size: int, dtype: torch.dtype, device: str, 
                         num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float, float]:
    """Each GPU does its own matmul independently (no communication)"""
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
    tflops = calculate_tflops(matrix_size, avg_time)
    
    return avg_time, tflops, 0.0  # No communication time

def benchmark_data_parallel(matrix_size: int, dtype: torch.dtype, device: str, rank: int,
                           num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float, float]:
    """Simulate data parallel: each GPU computes matmul then allreduce the result"""
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup_iterations):
        C = torch.matmul(A, B)
        if dist.is_initialized():
            dist.all_reduce(C, op=dist.ReduceOp.SUM)
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
        C = torch.matmul(A, B)
        end_event.record()
        torch.cuda.synchronize()
        
        compute_time += start_event.elapsed_time(end_event)
        
        # Time communication
        if dist.is_initialized():
            start_event.record()
            dist.all_reduce(C, op=dist.ReduceOp.SUM)
            end_event.record()
            torch.cuda.synchronize()
            comm_time += start_event.elapsed_time(end_event)
    
    avg_compute_time = (compute_time / 1000.0) / num_iterations
    avg_comm_time = (comm_time / 1000.0) / num_iterations
    avg_total_time = avg_compute_time + avg_comm_time
    
    tflops = calculate_tflops(matrix_size, avg_compute_time)  # TFLOPS based on compute only
    
    return avg_total_time, tflops, avg_comm_time

def benchmark_model_parallel(matrix_size: int, dtype: torch.dtype, device: str, 
                            rank: int, world_size: int,
                            num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float, float]:
    """Model parallel: split matrix B across GPUs, each GPU computes partial result"""
    if world_size == 1:
        return benchmark_independent(matrix_size, dtype, device, num_iterations, warmup_iterations)
    
    # Each GPU gets full A but only a slice of B
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Split B column-wise across GPUs
    cols_per_gpu = matrix_size // world_size
    start_col = rank * cols_per_gpu
    end_col = start_col + cols_per_gpu if rank < world_size - 1 else matrix_size
    
    B_local = torch.randn(matrix_size, end_col - start_col, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup_iterations):
        # Local matmul
        C_local = torch.matmul(A[:, start_col:end_col], B_local)
        
        # Gather results
        if dist.is_initialized():
            C_list = [torch.zeros_like(C_local) for _ in range(world_size)]
            dist.all_gather(C_list, C_local)
    
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
        C_local = torch.matmul(A[:, start_col:end_col], B_local)
        end_event.record()
        torch.cuda.synchronize()
        
        compute_time += start_event.elapsed_time(end_event)
        
        # Time communication
        if dist.is_initialized():
            C_list = [torch.zeros_like(C_local) for _ in range(world_size)]
            start_event.record()
            dist.all_gather(C_list, C_local)
            end_event.record()
            torch.cuda.synchronize()
            comm_time += start_event.elapsed_time(end_event)
    
    avg_compute_time = (compute_time / 1000.0) / num_iterations
    avg_comm_time = (comm_time / 1000.0) / num_iterations
    avg_total_time = avg_compute_time + avg_comm_time
    
    # Compute is reduced by world_size, but we're doing the full operation collectively
    tflops = calculate_tflops(matrix_size, avg_total_time) if world_size > 1 else calculate_tflops(matrix_size, avg_compute_time)
    
    return avg_total_time, tflops, avg_comm_time

def run_benchmarks(rank: int, world_size: int, matrix_sizes: List[int], 
                   dtype: torch.dtype, mode: BenchmarkMode, 
                   num_iterations: int, warmup_iterations: int):
    device = f'cuda:{rank % torch.cuda.device_count()}'
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Distributed Matrix Multiplication Benchmark")
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
            if rank == 0:
                print(f"  - Running warmup and benchmark...")
            
            if mode == BenchmarkMode.INDEPENDENT:
                avg_time, tflops, comm_time = benchmark_independent(
                    size, dtype, device, num_iterations, warmup_iterations
                )
            elif mode == BenchmarkMode.DATA_PARALLEL:
                avg_time, tflops, comm_time = benchmark_data_parallel(
                    size, dtype, device, rank, num_iterations, warmup_iterations
                )
            elif mode == BenchmarkMode.MODEL_PARALLEL:
                avg_time, tflops, comm_time = benchmark_model_parallel(
                    size, dtype, device, rank, world_size, num_iterations, warmup_iterations
                )
            
            # Gather results to rank 0
            time_tensor = torch.tensor([avg_time], device=device)
            tflops_tensor = torch.tensor([tflops], device=device)
            comm_tensor = torch.tensor([comm_time], device=device)
            
            if dist.is_initialized():
                if mode == BenchmarkMode.INDEPENDENT:
                    # For independent, sum TFLOPS (each GPU contributes)
                    dist.all_reduce(tflops_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
                else:
                    # For parallel modes, we want averages
                    dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(comm_tensor, op=dist.ReduceOp.AVG)
                    if mode == BenchmarkMode.DATA_PARALLEL:
                        dist.all_reduce(tflops_tensor, op=dist.ReduceOp.AVG)
            
            if rank == 0:
                print(f"\nResults for {size}x{size}:")
                print(f"  - Total time per operation: {time_tensor.item()*1000:.3f} ms")
                if comm_tensor.item() > 0:
                    compute_time = time_tensor.item() - comm_tensor.item()
                    print(f"  - Compute time: {compute_time*1000:.3f} ms")
                    print(f"  - Communication time: {comm_tensor.item()*1000:.3f} ms")
                    print(f"  - Communication overhead: {(comm_tensor.item()/time_tensor.item())*100:.1f}%")
                
                if mode == BenchmarkMode.INDEPENDENT:
                    print(f"  - TFLOPS per GPU: {tflops:.2f}")
                    print(f"  - Total TFLOPS (all GPUs): {tflops_tensor.item():.2f}")
                else:
                    print(f"  - Effective TFLOPS: {tflops_tensor.item():.2f}")
                
                print(f"  - Required FLOPs per operation: {2.0 * size**3 / 1e12:.2f} TFLOPs")
                
                # Calculate scaling efficiency for parallel modes
                if world_size > 1 and mode != BenchmarkMode.INDEPENDENT:
                    ideal_speedup = world_size
                    if comm_tensor.item() > 0:
                        actual_speedup = 1.0 / (compute_time / (time_tensor.item() * world_size))
                        scaling_efficiency = (actual_speedup / ideal_speedup) * 100
                        print(f"  - Scaling efficiency: {scaling_efficiency:.1f}%")
                
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
    parser = argparse.ArgumentParser(description='Distributed PyTorch Matrix Multiplication Benchmark with Communication')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4096, 8192, 16384],
                        help='Matrix sizes to benchmark (default: 4096 8192 16384)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations per benchmark (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for matrices (default: bfloat16)')
    parser.add_argument('--mode', type=str, default='data_parallel',
                        choices=['independent', 'data_parallel', 'model_parallel'],
                        help='Benchmark mode (default: data_parallel)')
    
    args = parser.parse_args()
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    mode = BenchmarkMode(args.mode)
    
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