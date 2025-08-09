#!/usr/bin/env python3
"""
Quick comparison script to run all benchmark modes and compare results
"""

import subprocess
import sys
import time

def run_benchmark(script, gpus, mode, dtype="bfloat16"):
    """Run a benchmark and capture output"""
    cmd = f"./{script} {gpus} {mode} {dtype}"
    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse results for 16k matrix
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if "16384x16384" in line:
            # Print next 10 lines which contain results
            for j in range(i, min(i+15, len(lines))):
                if "Results for" in lines[j] or "Average time" in lines[j] or "TFLOPS" in lines[j] or "overhead" in lines[j]:
                    print(lines[j])
    
    return result.stdout

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print("="*80)
    
    # Test 1: Original benchmark (independent)
    print("\n### TEST 1: Original benchmark - Independent (no communication)")
    run_benchmark("run_benchmark.sh", 2, "", "bfloat16")
    
    # Test 2: Distributed with communication (data_parallel)
    print("\n### TEST 2: Distributed - Data Parallel (with allreduce)")
    run_benchmark("run_distributed_benchmark.sh", 2, "data_parallel", "bfloat16")
    
    # Test 3: No overlap
    print("\n### TEST 3: Overlap Benchmark - No Overlap")
    run_benchmark("run_overlap_benchmark.sh", 2, "no_overlap", "bfloat16")
    
    # Test 4: With overlap
    print("\n### TEST 4: Overlap Benchmark - With Overlap")
    run_benchmark("run_overlap_benchmark.sh", 2, "overlap", "bfloat16")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
    Key Metrics to Compare:
    1. Independent (no communication) = baseline maximum throughput
    2. Data Parallel (with allreduce) = realistic distributed training
    3. No Overlap = sequential compute then communicate
    4. With Overlap = overlapped compute and communicate
    
    The overlap should show improvement over no_overlap, but both should
    be slower than independent due to communication overhead.
    """)

if __name__ == "__main__":
    main()