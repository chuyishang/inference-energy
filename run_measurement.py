#!/usr/bin/env python3
"""Automated energy measurement workflow for vLLM."""

import argparse
import asyncio
import csv
import json
import subprocess
import sys
import time
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
    parser.add_argument("--output-dir", type=Path, default=Path("logs"), help="Output directory (default: logs)")
    parser.add_argument(
        "--skip-idle", action="store_true", help="Skip idle measurement and provide --idle-power manually"
    )
    parser.add_argument("--idle-power", type=float, help="Manually specified idle power in W (requires --skip-idle)")

    args = parser.parse_args()

    if args.skip_idle and args.idle_power is None:
        parser.error("--skip-idle requires --idle-power to be specified")

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
    run_command(
        [
            "inference-energy",
            "analyze",
            "--power-log",
            str(active_log),
            "--requests-log",
            str(requests_log),
            "--idle-power",
            str(idle_power_w),
            "--output",
            str(summary_path),
        ],
        "Analyzing results",
    )

    # Print summary
    with summary_path.open() as f:
        summary = json.load(f)

    print("\n" + "="*60)
    print("MEASUREMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"\nKey metrics:")
    print(f"  Idle power:                {summary['idle_power_W']:.2f} W")
    print(f"  Total energy:              {summary['total_energy_J']:.2f} J")
    print(f"  Active energy:             {summary['active_energy_J']:.2f} J")
    print(f"  Completion tokens:         {summary['total_completion_tokens']:,}")
    print(f"  Energy per token:          {summary['energy_per_completion_token_J']:.4f} J/token")
    print(f"  Energy per 1K tokens:      {summary['energy_per_completion_token_J'] * 1000:.2f} J/1K tokens")

    # Additional useful metrics
    energy_kwh = summary['active_energy_J'] / 3600000
    print(f"  Active energy (kWh):       {energy_kwh:.6f} kWh")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
