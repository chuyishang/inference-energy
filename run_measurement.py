#!/usr/bin/env python3
"""Automated energy measurement workflow for vLLM."""

import argparse
import asyncio
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def compute_average_power(log_path: Path) -> float:
    """Compute average power from a power log."""
    powers = []
    with log_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            powers.append(float(row["power_W"]))
    return sum(powers) / len(powers) if powers else 0.0


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(1)


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    # Replace slashes and other special chars with underscores
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', model_name)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('._')
    return sanitized


def main():
    parser = argparse.ArgumentParser(
        description="Automated vLLM energy measurement workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_measurement.py --endpoint http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct

This will:
  1. Measure idle baseline power (5 min)
  2. Run warmup load (3 min)
  3. Measure active power + run load test (10 min)
  4. Analyze and report results
        """,
    )
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint URL")
    parser.add_argument("--model", required=True, help="Model name for vLLM")
    parser.add_argument("--gpu", type=int, default=3, help="GPU device index (default: 3)")
    parser.add_argument(
        "--idle-duration", type=int, default=300, help="Idle baseline duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--warmup-duration", type=int, default=180, help="Warmup duration in seconds (default: 180)"
    )
    parser.add_argument(
        "--measurement-duration", type=int, default=600, help="Active measurement duration in seconds (default: 600)"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent clients (default: 8)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: results/<model>_<timestamp>)")
    parser.add_argument(
        "--skip-idle", action="store_true", help="Skip idle measurement and provide --idle-power manually"
    )
    parser.add_argument("--idle-power", type=float, help="Manually specified idle power in W (requires --skip-idle)")
    parser.add_argument("--model-size-gb", type=float, help="Model size in GB (for D4 calculation)")
    parser.add_argument(
        "--gpu-memory-bw-gbs", type=float, help="Theoretical GPU memory bandwidth in GB/s (for D4)"
    )
    parser.add_argument("--flops", type=float, help="Measured FLOPs from profiler (for M10/D3)")

    args = parser.parse_args()

    if args.skip_idle and args.idle_power is None:
        parser.error("--skip-idle requires --idle-power to be specified")

    # Create output directory based on model name if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = sanitize_model_name(args.model)
        args.output_dir = Path("results") / f"{model_name_safe}_{timestamp}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Measure idle baseline
    idle_power_w = args.idle_power
    if not args.skip_idle:
        print("\n" + "="*60)
        print("STEP 1: Measuring idle baseline power")
        print("="*60)
        print(f"Please ensure vLLM is NOT running on GPU {args.gpu}")
        input("Press Enter when ready to start idle measurement...")

        idle_log = args.output_dir / "idle.csv"
        run_command(
            [
                "inference-energy",
                "log-power",
                "--duration",
                str(args.idle_duration),
                "--interval",
                "0.1",
                "--device-index",
                str(args.gpu),
                "--output",
                str(idle_log),
            ],
            f"Measuring idle power for {args.idle_duration}s",
        )

        idle_power_w = compute_average_power(idle_log)
        print(f"\nIdle power: {idle_power_w:.2f} W")

    # Step 2: Warmup
    print("\n" + "="*60)
    print("STEP 2: Warmup phase")
    print("="*60)
    print(f"Please ensure vLLM is running at {args.endpoint}")
    input("Press Enter when vLLM is ready...")

    warmup_log = args.output_dir / "warmup_requests.csv"
    run_command(
        [
            "inference-energy",
            "load-test",
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
            "--random-prompts",
            "--duration",
            str(args.warmup_duration),
            "--concurrency",
            "4",
            "--output",
            str(warmup_log),
        ],
        f"Warming up for {args.warmup_duration}s",
    )

    print("\nWarmup complete. Starting actual measurement in 5 seconds...")
    time.sleep(5)

    # Step 3: Run measurement (power logging + load test)
    print("\n" + "="*60)
    print("STEP 3: Active measurement")
    print("="*60)

    active_log = args.output_dir / "active.csv"
    requests_log = args.output_dir / "requests.csv"

    # Add buffer time for power logging
    power_duration = args.measurement_duration + 30

    print(f"Starting power logging for {power_duration}s (includes buffer)...")
    power_proc = subprocess.Popen(
        [
            "inference-energy",
            "log-power",
            "--duration",
            str(power_duration),
            "--interval",
            "0.1",
            "--device-index",
            str(args.gpu),
            "--output",
            str(active_log),
        ]
    )

    # Wait a bit for power logging to start
    time.sleep(2)

    print(f"Starting load test for {args.measurement_duration}s with concurrency {args.concurrency}...")
    load_proc = subprocess.Popen(
        [
            "inference-energy",
            "load-test",
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
            "--random-prompts",
            "--duration",
            str(args.measurement_duration),
            "--concurrency",
            str(args.concurrency),
            "--output",
            str(requests_log),
        ]
    )

    # Wait for load test to complete
    load_proc.wait()
    print("Load test complete. Waiting for power logging to finish...")

    # Wait for power logging
    power_proc.wait()
    print("Power logging complete.")

    # Step 4: Analyze
    print("\n" + "="*60)
    print("STEP 4: Analysis")
    print("="*60)

    summary_path = args.output_dir / "summary.json"
    analyze_cmd = [
        "inference-energy",
        "analyze",
        "--power-log",
        str(active_log),
        "--requests-log",
        str(requests_log),
        "--idle-power",
        str(idle_power_w),
        "--comprehensive",
        "--output",
        str(summary_path),
    ]

    if args.model_size_gb:
        analyze_cmd.extend(["--model-size-gb", str(args.model_size_gb)])
    if args.gpu_memory_bw_gbs:
        analyze_cmd.extend(["--gpu-memory-bw-gbs", str(args.gpu_memory_bw_gbs)])
    if args.flops:
        analyze_cmd.extend(["--flops", str(args.flops)])

    run_command(analyze_cmd, "Analyzing results")

    # Print summary
    with summary_path.open() as f:
        summary = json.load(f)

    print("\n" + "="*60)
    print("MEASUREMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")

    # Display comprehensive metrics
    print("\n" + "="*60)
    print("PRIMARY MEASUREMENTS (M1-M10)")
    print("="*60)
    print(f"M1  Total energy:              {summary['M1_total_energy_J']:.2f} J ({summary['M1_total_energy_J']/3600000:.6f} kWh)")
    print(f"M2  Total tokens:              {summary['M2_total_tokens']:,}")
    print(f"M3  Total time:                {summary['M3_total_time_s']:.2f} s ({summary['M3_total_time_s']/60:.2f} min)")
    print(f"M4  Avg prefill time:          {summary['M4_avg_prefill_time_s']:.4f} s (estimated)")
    print(f"M5  Avg decode time/token:     {summary['M5_avg_decode_time_per_token_s']:.4f} s (estimated)")
    print(f"M6  Average power:             {summary['M6_avg_power_W']:.2f} W")
    print(f"M7  Peak power:                {summary['M7_peak_power_W']:.2f} W")
    print(f"M8  Avg GPU utilization:       {summary['M8_avg_gpu_util_percent']:.1f}%")
    print(f"M9  Avg memory utilization:    {summary['M9_avg_mem_util_percent']:.1f}%")
    m10_str = f"{summary['M10_flops_measured']:.2e}" if summary['M10_flops_measured'] else "N/A"
    print(f"M10 FLOPs measured:            {m10_str}")

    print("\n" + "="*60)
    print("DERIVED METRICS (D1-D4)")
    print("="*60)
    print(f"D1  Energy per token:          {summary['D1_energy_per_token_J']:.4f} J/token")
    print(f"                               {summary['D1_energy_per_token_J']*1000:.2f} J/1K tokens")
    print(f"D2  Throughput:                {summary['D2_throughput_tokens_per_s']:.2f} tokens/s")
    d3_str = f"{summary['D3_power_efficiency_flops_per_W']:.2e} FLOPs/W" if summary['D3_power_efficiency_flops_per_W'] else "N/A (need --flops)"
    print(f"D3  Power efficiency:          {d3_str}")
    d4_str = f"{summary['D4_memory_bandwidth_util_percent']:.1f}%" if summary['D4_memory_bandwidth_util_percent'] else "N/A (need --model-size-gb and --gpu-memory-bw-gbs)"
    print(f"D4  Memory bandwidth util:     {d4_str}")

    print("\n" + "="*60)
    print("ADDITIONAL CONTEXT")
    print("="*60)
    print(f"Idle power:                    {summary['idle_power_W']:.2f} W")
    print(f"Active energy:                 {summary['active_energy_J']:.2f} J ({summary['active_energy_J']/3600000:.6f} kWh)")
    print(f"Total requests:                {summary['total_requests']:,}")
    print(f"Successful requests:           {summary['successful_requests']:,}")
    print(f"Average latency:               {summary['avg_latency_s']:.3f} s")
    print(f"GPU memory total:              {summary['mem_total_GB']:.2f} GB")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
