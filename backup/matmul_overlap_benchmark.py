import torch
import torch.distributed as dist
import torch.nn as nn
import time
import os
import argparse
from typing import List, Tuple
import enum
from collections import deque

class BenchmarkMode(enum.Enum):
    NO_OVERLAP = "no_overlap"      # Sequential compute then communicate
    OVERLAP = "overlap"            # Overlapped compute and communicate
    PIPELINE = "pipeline"          # Pipeline multiple operations

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

def benchmark_no_overlap(matrix_size: int, dtype: torch.dtype, device: str, rank: int,
                         num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float, float]:
    """Traditional approach: compute, wait, communicate, wait, repeat"""
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup_iterations):
        C = torch.matmul(A, B)
        if dist.is_initialized():
            dist.all_reduce(C, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_iterations):
        # Compute
        C = torch.matmul(A, B)
        
        # Wait for compute to finish before starting communication
        torch.cuda.synchronize()
        
        # Communicate
        if dist.is_initialized():
            dist.all_reduce(C, op=dist.ReduceOp.SUM)
        
        # Wait for communication to finish
        torch.cuda.synchronize()
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    total_time = total_time_ms / 1000.0
    avg_time = total_time / num_iterations
    
    # Calculate compute-only time for TFLOPS
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    compute_start.record()
    for _ in range(10):
        C = torch.matmul(A, B)
    compute_end.record()
    torch.cuda.synchronize()
    
    compute_time = (compute_start.elapsed_time(compute_end) / 1000.0) / 10
    tflops = calculate_tflops(matrix_size, compute_time)
    
    return avg_time, tflops, 0.0

def benchmark_overlap(matrix_size: int, dtype: torch.dtype, device: str, rank: int,
                     num_iterations: int = 50, warmup_iterations: int = 10) -> Tuple[float, float, float]:
    """Overlapped approach: start next compute while previous communication is happening"""
    
    # Create two sets of matrices to alternate between
    A1 = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B1 = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    A2 = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    B2 = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
    
    # Create CUDA streams for overlapping
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()
    
    # Warmup
    for _ in range(warmup_iterations):
        with torch.cuda.stream(compute_stream):
            C1 = torch.matmul(A1, B1)
        compute_stream.synchronize()
        if dist.is_initialized():
            with torch.cuda.stream(comm_stream):
                dist.all_reduce(C1, op=dist.ReduceOp.SUM)
        comm_stream.synchronize()
    
    # Benchmark with overlap
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    # First computation (no overlap possible)
    with torch.cuda.stream(compute_stream):
        C1 = torch.matmul(A1, B1)
    
    # Main loop with overlap
    for i in range(num_iterations - 1):
        # Start communication of previous result
        if dist.is_initialized():
            compute_stream.wait_stream(comm_stream)  # Wait for previous comm to finish
            with torch.cuda.stream(comm_stream):
                if i % 2 == 0:
                    dist.all_reduce(C1, op=dist.ReduceOp.SUM, async_op=True)
                else:
                    dist.all_reduce(C2, op=dist.ReduceOp.SUM, async_op=True)
        
        # Start next computation while communication is happening
        with torch.cuda.stream(compute_stream):
            if i % 2 == 0:
                C2 = torch.matmul(A2, B2)
            else:
                C1 = torch.matmul(A1, B1)
    
    # Final communication
    if dist.is_initialized():
        compute_stream.synchronize()
        with torch.cuda.stream(comm_stream):
            if (num_iterations - 1) % 2 == 0:
                dist.all_reduce(C2, op=dist.ReduceOp.SUM)
            else:
                dist.all_reduce(C1, op=dist.ReduceOp.SUM)
    
    # Wait for all operations to complete
    compute_stream.synchronize()
    comm_stream.synchronize()
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    total_time = total_time_ms / 1000.0
    avg_time = total_time / num_iterations
    
    # Calculate compute-only time for TFLOPS
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    compute_start.record()
    for _ in range(10):
        C1 = torch.matmul(A1, B1)
    compute_end.record()
    torch.cuda.synchronize()
    
    compute_time = (compute_start.elapsed_time(compute_end) / 1000.0) / 10
    tflops = calculate_tflops(matrix_size, compute_time)
    
    return avg_time, tflops, 0.0

def benchmark_pipeline(matrix_size: int, dtype: torch.dtype, device: str, rank: int,
                      num_iterations: int = 50, warmup_iterations: int = 10,
                      pipeline_depth: int = 3) -> Tuple[float, float, float]:
    """Deep pipeline: multiple operations in flight simultaneously"""
    
    # Create multiple sets of matrices for pipelining
    matrices_A = [torch.randn(matrix_size, matrix_size, dtype=dtype, device=device) 
                  for _ in range(pipeline_depth)]
    matrices_B = [torch.randn(matrix_size, matrix_size, dtype=dtype, device=device) 
                  for _ in range(pipeline_depth)]
    results = [None] * pipeline_depth
    
    # Create CUDA streams
    compute_streams = [torch.cuda.Stream() for _ in range(pipeline_depth)]
    comm_handles = []
    
    # Warmup
    for _ in range(warmup_iterations):
        C = torch.matmul(matrices_A[0], matrices_B[0])
        if dist.is_initialized():
            dist.all_reduce(C, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    # Benchmark with deep pipeline
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    # Pipeline filling phase
    for stage in range(min(pipeline_depth, num_iterations)):
        with torch.cuda.stream(compute_streams[stage % pipeline_depth]):
            results[stage % pipeline_depth] = torch.matmul(
                matrices_A[stage % pipeline_depth], 
                matrices_B[stage % pipeline_depth]
            )
    
    # Main pipeline execution
    for i in range(num_iterations):
        stage_idx = i % pipeline_depth
        
        # Start communication for completed computation
        if i >= pipeline_depth:
            # Wait for any pending communication to complete
            if comm_handles and len(comm_handles) >= pipeline_depth:
                oldest_handle = comm_handles.pop(0)
                if oldest_handle is not None:
                    oldest_handle.wait()
        
        # Start async communication for current result
        if dist.is_initialized() and results[stage_idx] is not None:
            handle = dist.all_reduce(results[stage_idx], op=dist.ReduceOp.SUM, async_op=True)
            comm_handles.append(handle)
        else:
            comm_handles.append(None)
        
        # Start next computation (if within iteration limit)
        if i + pipeline_depth < num_iterations:
            next_idx = (i + pipeline_depth) % pipeline_depth
            with torch.cuda.stream(compute_streams[next_idx]):
                results[next_idx] = torch.matmul(
                    matrices_A[next_idx], 
                    matrices_B[next_idx]
                )
    
    # Pipeline draining phase - wait for remaining communications
    for handle in comm_handles:
        if handle is not None:
            handle.wait()
    
    # Synchronize all streams
    for stream in compute_streams:
        stream.synchronize()
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    total_time = total_time_ms / 1000.0
    avg_time = total_time / num_iterations
    
    # Calculate compute-only time for TFLOPS
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    compute_start.record()
    for _ in range(10):
        C = torch.matmul(matrices_A[0], matrices_B[0])
    compute_end.record()
    torch.cuda.synchronize()
    
    compute_time = (compute_start.elapsed_time(compute_end) / 1000.0) / 10
    tflops = calculate_tflops(matrix_size, compute_time)
    
    return avg_time, tflops, 0.0

def run_benchmarks(rank: int, world_size: int, matrix_sizes: List[int], 
                   dtype: torch.dtype, mode: BenchmarkMode, 
                   num_iterations: int, warmup_iterations: int):
    device = f'cuda:{rank % torch.cuda.device_count()}'
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Overlapped Communication/Computation Benchmark")
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
            print(f"  - Running warmup and benchmark...")
        
        try:
            if mode == BenchmarkMode.NO_OVERLAP:
                avg_time, tflops, _ = benchmark_no_overlap(
                    size, dtype, device, rank, num_iterations, warmup_iterations
                )
            elif mode == BenchmarkMode.OVERLAP:
                avg_time, tflops, _ = benchmark_overlap(
                    size, dtype, device, rank, num_iterations, warmup_iterations
                )
            elif mode == BenchmarkMode.PIPELINE:
                avg_time, tflops, _ = benchmark_pipeline(
                    size, dtype, device, rank, num_iterations, warmup_iterations
                )
            
            # Gather results
            time_tensor = torch.tensor([avg_time], device=device)
            tflops_tensor = torch.tensor([tflops], device=device)
            
            if dist.is_initialized():
                dist.all_reduce(time_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(tflops_tensor, op=dist.ReduceOp.AVG)
            
            if rank == 0:
                print(f"\nResults for {size}x{size}:")
                print(f"  - Average time per operation: {time_tensor.item()*1000:.3f} ms")
                
                # Simple, clear calculation: FLOPs / Time
                flops_per_operation = 2.0 * size**3
                actual_tflops = (flops_per_operation / time_tensor.item()) / 1e12
                
                print(f"  - Actual TFLOPS: {actual_tflops:.2f} (FLOPs/Time)")
                
                # For multi-GPU in data parallel, each GPU does the same work
                # So total system TFLOPS is the same as single GPU TFLOPS
                if world_size > 1:
                    print(f"  - Note: In data parallel, each GPU does full matrix multiply")
                    print(f"  - System is achieving {actual_tflops:.2f} TFLOPS effectively")
                
                print(f"  - Required FLOPs per operation: {flops_per_operation / 1e12:.2f} TFLOPs")
                
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
    parser = argparse.ArgumentParser(description='Overlapped Communication/Computation Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4096, 8192, 16384],
                        help='Matrix sizes to benchmark (default: 4096 8192 16384)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations per benchmark (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for matrices (default: bfloat16)')
    parser.add_argument('--mode', type=str, default='overlap',
                        choices=['no_overlap', 'overlap', 'pipeline'],
                        help='Benchmark mode (default: overlap)')
    
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
            
            # Check if GPUs are connected via NVLink
            if torch.cuda.device_count() > 1:
                print(f"\nChecking GPU interconnect...")
                try:
                    # This is a simple check - actual NVLink detection would need nvidia-ml-py
                    print(f"  Multi-GPU system detected. Use nvidia-smi topo -m for interconnect details.")
                except:
                    pass
    
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