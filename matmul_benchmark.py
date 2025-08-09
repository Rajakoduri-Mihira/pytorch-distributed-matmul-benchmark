import torch
import torch.distributed as dist
import torch.nn as nn
import time
import os
import argparse
from typing import List, Tuple

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

def benchmark_matmul(matrix_size: int, dtype: torch.dtype, device: str, 
                     num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float]:
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    print(f"  Performing {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        C = torch.matmul(A, B)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    print(f"  Running {num_iterations} benchmark iterations...")
    
    # Use CUDA events for precise timing
    if device.startswith('cuda'):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        
        end_event.record()
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        total_time = total_time_ms / 1000.0  # Convert to seconds
    else:
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        end_time = time.perf_counter()
        total_time = end_time - start_time
    
    avg_time = total_time / num_iterations
    tflops = calculate_tflops(matrix_size, avg_time)
    
    return avg_time, tflops

def run_benchmarks(rank: int, world_size: int, matrix_sizes: List[int], 
                   dtype: torch.dtype, num_iterations: int, warmup_iterations: int):
    device = f'cuda:{rank % torch.cuda.device_count()}'
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Matrix Multiplication Benchmark")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Number of GPUs: {world_size}")
        print(f"  - Data type: {dtype}")
        print(f"  - Device: CUDA")
        print(f"  - Iterations per test: {num_iterations}")
        print(f"  - Warmup iterations: {warmup_iterations}")
        print(f"{'='*60}\n")
    
    for size in matrix_sizes:
        if rank == 0:
            bytes_per_element = 4 if dtype == torch.float32 else 2
            dtype_name = 'float32' if dtype == torch.float32 else ('float16' if dtype == torch.float16 else 'bfloat16')
            print(f"\nBenchmarking {size}x{size} matrix multiplication:")
            print(f"  - Memory per matrix: {size * size * bytes_per_element / (1024**3):.2f} GB ({dtype_name})")
            print(f"  - Total memory for A, B, C: {3 * size * size * bytes_per_element / (1024**3):.2f} GB")
        
        try:
            avg_time, tflops = benchmark_matmul(
                size, dtype, device, num_iterations, warmup_iterations
            )
            
            tflops_tensor = torch.tensor([tflops], device=device)
            time_tensor = torch.tensor([avg_time], device=device)
            
            if dist.is_initialized():
                dist.all_reduce(tflops_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
                total_tflops = tflops_tensor.item()
                avg_time_global = time_tensor.item()
            else:
                total_tflops = tflops
                avg_time_global = avg_time
            
            if rank == 0:
                print(f"\nResults for {size}x{size}:")
                print(f"  - Average time per multiplication: {avg_time_global*1000:.3f} ms")
                print(f"  - TFLOPS per GPU: {tflops:.2f}")
                print(f"  - Total TFLOPS (all GPUs): {total_tflops:.2f}")
                print(f"  - Required FLOPs per operation: {2.0 * size**3 / 1e12:.2f} TFLOPs")
                # RTX 6000 Ada has ~91.1 TFLOPS FP32 theoretical peak
                print(f"  - GPU Efficiency: {(tflops / 91.1) * 100:.1f}% of theoretical peak")
                
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
    parser = argparse.ArgumentParser(description='Distributed PyTorch Matrix Multiplication Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4096, 8192, 16384],
                        help='Matrix sizes to benchmark (default: 4096 8192 16384)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations per benchmark (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for matrices (default: bfloat16)')
    
    args = parser.parse_args()
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
                props = torch.cuda.get_device_properties(i)
                # Calculate theoretical TFLOPS for the GPU
                sm_count = props.multi_processor_count
                print(f"    SMs: {sm_count}")
    
    try:
        run_benchmarks(rank, world_size, args.sizes, dtype, args.iterations, args.warmup)
    finally:
        cleanup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Benchmark completed!")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()