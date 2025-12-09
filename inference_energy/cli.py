"""Command-line helpers for logging power and sending load to vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from inference_energy.analysis import analyze_comprehensive, summarize
from inference_energy.load_generator import LoadGenConfig, run_load_test
from inference_energy.power_logging import log_gpu_power


def _read_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Prompt file is empty after stripping blank lines")
    return prompts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inference energy measurement tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    power_parser = subparsers.add_parser("log-power", help="Record GPU power via NVML")
    power_parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    power_parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds")
    power_parser.add_argument("--device-index", type=int, default=0, help="GPU index to log")
    power_parser.add_argument("--output", type=Path, required=True, help="CSV output path")

    load_parser = subparsers.add_parser("load-test", help="Send load to vLLM OpenAI endpoint")
    load_parser.add_argument("--endpoint", type=str, required=True, help="Base URL for the vLLM server")
    load_parser.add_argument("--model", type=str, required=True, help="Model name to request")
    prompt_group = load_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompts", type=Path, help="Path to prompts file (one prompt per line)")
    prompt_group.add_argument(
        "--random-prompts",
        action="store_true",
        help="Generate synthetic prompts with configurable length distribution",
    )
    load_parser.add_argument(
        "--prompt-mean-tokens",
        type=int,
        default=256,
        help="Mean prompt length for synthetic prompts (tokens, approximate)",
    )
    load_parser.add_argument(
        "--prompt-std-tokens",
        type=int,
        default=64,
        help="Stddev of prompt length for synthetic prompts (tokens, approximate)",
    )
    load_parser.add_argument(
        "--synthetic-vocab-size",
        type=int,
        default=5000,
        help="Vocabulary size to sample synthetic tokens from",
    )
    load_parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    load_parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent clients")
    load_parser.add_argument("--max-new-tokens", type=int, default=256, help="max_tokens value")
    load_parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    load_parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    load_parser.add_argument("--output", type=Path, required=True, help="CSV output path for request logs")

    analyze_parser = subparsers.add_parser("analyze", help="Integrate power and attribute energy to tokens")
    analyze_parser.add_argument("--power-log", type=Path, required=True, help="CSV file from log-power")
    analyze_parser.add_argument("--requests-log", type=Path, required=True, help="CSV file from load-test")
    analyze_parser.add_argument("--idle-power", type=float, default=0.0, help="Idle power to subtract (W)")
    analyze_parser.add_argument("--output", type=Path, help="Optional JSON output file for the summary")
    analyze_parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate comprehensive metrics (M1-M10, D1-D4)",
    )
    analyze_parser.add_argument("--model-size-gb", type=float, help="Model size in GB (for D4 calculation)")
    analyze_parser.add_argument(
        "--gpu-memory-bw-gbs", type=float, help="Theoretical GPU memory bandwidth in GB/s (for D4)"
    )
    analyze_parser.add_argument("--flops", type=float, help="Measured FLOPs from profiler (for M10/D3)")

    args = parser.parse_args(argv)

    if args.command == "log-power":
        log_gpu_power(
            duration_s=args.duration,
            interval_s=args.interval,
            output_path=args.output,
            device_index=args.device_index,
        )
        return 0

    if args.command == "load-test":
        prompts = None
        if args.prompts:
            prompts = _read_prompts(args.prompts)
        cfg = LoadGenConfig(
            endpoint=args.endpoint,
            model=args.model,
            prompts=prompts,
            duration_s=args.duration,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            request_timeout_s=args.timeout,
            synthetic_prompts=args.random_prompts,
            prompt_mean_tokens=args.prompt_mean_tokens,
            prompt_std_tokens=args.prompt_std_tokens,
            synthetic_vocab_size=args.synthetic_vocab_size,
        )
        asyncio.run(run_load_test(cfg, output_path=args.output))
        return 0

    if args.command == "analyze":
        if args.comprehensive:
            # Use comprehensive metrics
            model_size_bytes = int(args.model_size_gb * 1024**3) if args.model_size_gb else None
            gpu_memory_bw = args.gpu_memory_bw_gbs * 1024**3 if args.gpu_memory_bw_gbs else None

            metrics = analyze_comprehensive(
                power_log=args.power_log,
                requests_log=args.requests_log,
                idle_power_w=args.idle_power,
                model_size_bytes=model_size_bytes,
                gpu_memory_bw_bytes_per_s=gpu_memory_bw,
                flops_measured=args.flops,
            )

            summary = {
                # Primary measurements (M1-M10)
                "M1_total_energy_J": metrics.m1_total_energy_j,
                "M2_total_tokens": metrics.m2_total_tokens,
                "M3_total_time_s": metrics.m3_total_time_s,
                "M4_avg_prefill_time_s": metrics.m4_avg_prefill_time_s,
                "M5_avg_decode_time_per_token_s": metrics.m5_avg_decode_time_per_token_s,
                "M6_avg_power_W": metrics.m6_avg_power_w,
                "M7_peak_power_W": metrics.m7_peak_power_w,
                "M8_avg_gpu_util_percent": metrics.m8_avg_gpu_util_percent,
                "M9_avg_mem_util_percent": metrics.m9_avg_mem_util_percent,
                "M10_flops_measured": metrics.m10_flops_measured,
                # Derived metrics (D1-D4)
                "D1_energy_per_token_J": metrics.d1_energy_per_token_j,
                "D2_throughput_tokens_per_s": metrics.d2_throughput_tokens_per_s,
                "D3_power_efficiency_flops_per_W": metrics.d3_power_efficiency_flops_per_w,
                "D4_memory_bandwidth_util_percent": metrics.d4_memory_bandwidth_util_percent,
                # Additional context
                "idle_power_W": metrics.idle_power_w,
                "active_energy_J": metrics.active_energy_j,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "avg_latency_s": metrics.avg_latency_s,
                "mem_total_GB": metrics.mem_total_gb,
            }
        else:
            # Legacy analysis
            energy, tokens = summarize(
                power_log=args.power_log,
                requests_log=args.requests_log,
                idle_power_w=args.idle_power,
            )

            summary = {
                "duration_s": energy.duration_s,
                "idle_power_W": energy.idle_power_w,
                "total_energy_J": energy.total_energy_j,
                "active_energy_J": energy.active_energy_j,
                "total_completion_tokens": tokens.total_completion_tokens,
                "energy_per_completion_token_J": tokens.energy_per_completion_token_j,
            }

        print(json.dumps(summary, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2))
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
