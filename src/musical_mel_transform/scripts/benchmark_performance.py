#!/usr/bin/env python3
"""Performance benchmark script for Musical Mel Transform.

This script measures the actual performance of different FFT implementations
and frame sizes to generate accurate performance data.
"""

import argparse
import statistics
import sys
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from musical_mel_transform import ConvFFT, MusicalMelTransform


def warm_up_gpu(iterations: int = 50) -> None:
    """Warm up GPU with dummy operations to ensure consistent timing."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dummy_tensor = torch.randn(1000, 1000, device=device)
        for _ in range(iterations):
            _ = torch.matmul(dummy_tensor, dummy_tensor)
        torch.cuda.synchronize()


def benchmark_raw_fft(
    frame_size: int, iterations: int = 200, warmup_iterations: int = 50
) -> Dict[str, float]:
    """Benchmark raw FFT implementations (ConvFFT vs torch.fft.rfft)."""
    print(f"\nBenchmarking raw FFT (frame_size={frame_size})...")

    # Setup
    batch_size = 1
    num_channels = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conv_fft = ConvFFT(frame_size=frame_size).to(device)

    torch_times = []
    conv_times = []

    # Warmup
    print("  Warming up...")
    for _ in range(warmup_iterations):
        signal = torch.randn(batch_size, num_channels, frame_size, device=device)

        # Torch FFT warmup
        _ = torch.fft.rfft(signal, n=frame_size, dim=-1)

        # Conv FFT warmup
        _ = conv_fft.transform(signal)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"  Running {iterations} iterations...")

    # Actual benchmarking
    for i in tqdm(range(iterations), desc="  Progress"):
        signal = torch.randn(batch_size, num_channels, frame_size, device=device)

        # Benchmark Torch FFT
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        torch_output = torch.fft.rfft(signal, n=frame_size, dim=-1)
        torch_mag = torch.abs(torch_output)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_times.append((time.perf_counter() - start_time) * 1000)

        # Benchmark Conv FFT
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        _, _, conv_mag, _ = conv_fft.transform(signal)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        conv_times.append((time.perf_counter() - start_time) * 1000)

        # Verify accuracy (only on first few iterations to avoid overhead)
        if i < 5:
            assert torch.allclose(
                torch_mag, conv_mag, atol=1e-4
            ), "FFT implementations don't match!"

    return {
        "torch_mean": statistics.mean(torch_times),
        "torch_std": statistics.stdev(torch_times) if len(torch_times) > 1 else 0,
        "conv_mean": statistics.mean(conv_times),
        "conv_std": statistics.stdev(conv_times) if len(conv_times) > 1 else 0,
        "torch_median": statistics.median(torch_times),
        "conv_median": statistics.median(conv_times),
    }


def benchmark_mel_transform(
    frame_size: int, iterations: int = 200, warmup_iterations: int = 50
) -> Dict[str, float]:
    """Benchmark full MusicalMelTransform implementations."""
    print(f"\nBenchmarking MusicalMelTransform (frame_size={frame_size})...")

    # Setup common parameters
    common_kwargs = {
        "sample_rate": 44100,
        "frame_size": frame_size,
        "interval": 1.0,
        "f_min": 80.0,
        "passthrough_cutoff_hz": 8000,
        "norm": True,
        "min_bins": 2,
        "adaptive": True,
        "passthrough_grouping_size": 3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_transform = MusicalMelTransform(use_conv_fft=False, **common_kwargs).to(
        device
    )
    conv_transform = MusicalMelTransform(use_conv_fft=True, **common_kwargs).to(device)

    torch_times = []
    conv_times = []

    # Warmup
    print("  Warming up...")
    for _ in range(warmup_iterations):
        signal = torch.randn(1, frame_size, device=device)

        with torch.no_grad():
            _ = torch_transform.forward_frame(signal)
            _ = conv_transform.forward_frame(signal)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"  Running {iterations} iterations...")

    # Actual benchmarking
    for i in tqdm(range(iterations), desc="  Progress"):
        signal = torch.randn(1, frame_size, device=device)

        # Benchmark Torch-based transform
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            torch_mel, torch_fft = torch_transform.forward_frame(signal)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_times.append((time.perf_counter() - start_time) * 1000)

        # Benchmark Conv-based transform
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            conv_mel, conv_fft = conv_transform.forward_frame(signal)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        conv_times.append((time.perf_counter() - start_time) * 1000)

        # Verify accuracy (only on first few iterations)
        if i < 5:
            assert torch.allclose(
                torch_mel, conv_mel, atol=1e-4
            ), "Mel outputs don't match!"
            assert torch.allclose(
                torch_fft, conv_fft, atol=1e-4
            ), "FFT outputs don't match!"

    return {
        "torch_mean": statistics.mean(torch_times),
        "torch_std": statistics.stdev(torch_times) if len(torch_times) > 1 else 0,
        "conv_mean": statistics.mean(conv_times),
        "conv_std": statistics.stdev(conv_times) if len(conv_times) > 1 else 0,
        "torch_median": statistics.median(torch_times),
        "conv_median": statistics.median(conv_times),
    }


def print_performance_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print performance results in a clean table format."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)

    # Raw FFT Results
    if any("fft" in key for key in results.keys()):
        print("\nRaw FFT Performance:")
        print("| Configuration | Time (ms) | Speedup |")
        print("|---------------|-----------|---------|")

        frame_sizes = sorted(
            [int(k.split("_")[-1]) for k in results.keys() if "fft" in k]
        )

        for frame_size in frame_sizes:
            if f"fft_{frame_size}" in results:
                data = results[f"fft_{frame_size}"]
                torch_mean = data["torch_mean"]
                conv_mean = data["conv_mean"]
                speedup = torch_mean / conv_mean  # How much faster torch is

                print(
                    f"| Torch FFT (frame_size={frame_size}) | ~{torch_mean:.2f} | 1.0x |"
                )
                print(
                    f"| Conv FFT (frame_size={frame_size}) | ~{conv_mean:.2f} | {1/speedup:.2f}x |"
                )

        print(f"\n*Torch FFT is {speedup:.1f}x faster than Conv FFT on average*")

    # Mel Transform Results
    if any("mel" in key for key in results.keys()):
        print("\nFull MusicalMelTransform Performance:")
        print("| Configuration | Time (ms) | Speedup |")
        print("|---------------|-----------|---------|")

        for frame_size in frame_sizes:
            if f"mel_{frame_size}" in results:
                data = results[f"mel_{frame_size}"]
                torch_mean = data["torch_mean"]
                conv_mean = data["conv_mean"]
                speedup = conv_mean / torch_mean

                print(
                    f"| Torch Transform (frame_size={frame_size}) | ~{torch_mean:.2f} | 1.0x |"
                )
                print(
                    f"| Conv Transform (frame_size={frame_size}) | ~{conv_mean:.2f} | {1/speedup:.2f}x |"
                )

        print(
            f"\n*Torch Transform is {speedup:.1f}x faster than Conv Transform on average*"
        )

    # Detailed statistics
    print("\n" + "=" * 50)
    print("DETAILED STATISTICS")
    print("=" * 50)

    for key, data in results.items():
        test_type = "FFT" if "fft" in key else "Mel Transform"
        frame_size = key.split("_")[-1]

        print(f"\n{test_type} (frame_size={frame_size}):")
        print(
            f"  Torch - Mean: {data['torch_mean']:.2f}ms, Std: {data['torch_std']:.2f}ms, Median: {data['torch_median']:.2f}ms"
        )
        print(
            f"  Conv  - Mean: {data['conv_mean']:.2f}ms, Std: {data['conv_std']:.2f}ms, Median: {data['conv_median']:.2f}ms"
        )
        print(
            f"  Speedup: {data['conv_mean']/data['torch_mean']:.2f}x (torch is faster)"
        )

    # System info
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Number of threads: {torch.get_num_threads()}")


def main():
    """Main function for running performance benchmarks."""
    parser = argparse.ArgumentParser(
        description="Musical Mel Transform Performance Benchmark"
    )
    parser.add_argument(
        "--frame-sizes",
        nargs="+",
        type=int,
        default=[1024, 2048],
        help="Frame sizes to benchmark (default: 1024 2048)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of iterations for timing (default: 200)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--test-type",
        choices=["fft", "mel", "all"],
        default="fft",
        help="Type of test to run (default: fft)",
    )
    parser.add_argument(
        "--no-gpu-warmup",
        action="store_true",
        help="Skip GPU warmup (for CPU-only testing)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer iterations (50 iterations, 10 warmup)",
    )

    args = parser.parse_args()

    # Adjust iterations for quick test
    if args.quick:
        args.iterations = 50
        args.warmup = 10

    print("Musical Mel Transform Performance Benchmark")
    print("=" * 50)
    print(f"Testing frame sizes: {args.frame_sizes}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")
    print(f"Test type: {args.test_type}")

    # GPU warmup
    if torch.cuda.is_available() and not args.no_gpu_warmup:
        print("\nWarming up GPU...")
        warm_up_gpu()

    results = {}

    # Run benchmarks
    for frame_size in args.frame_sizes:
        if args.test_type in ["fft", "all"]:
            results[f"fft_{frame_size}"] = benchmark_raw_fft(
                frame_size, args.iterations, args.warmup
            )

        if args.test_type in ["mel", "all"]:
            results[f"mel_{frame_size}"] = benchmark_mel_transform(
                frame_size, args.iterations, args.warmup
            )

    # Print results
    print_performance_table(results)

    # Save results to file
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"

    results_with_metadata = {
        "timestamp": timestamp,
        "args": vars(args),
        "system": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
            "num_threads": torch.get_num_threads(),
        },
        "results": results,
    }

    with open(filename, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"\nDetailed results saved to: {filename}")


if __name__ == "__main__":
    main()
