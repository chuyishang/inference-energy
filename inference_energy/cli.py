"""Command-line helpers for logging power and sending load to vLLM."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

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
    load_parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts file (one prompt per line)")
    load_parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    load_parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent clients")
    load_parser.add_argument("--max-new-tokens", type=int, default=256, help="max_tokens value")
    load_parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    load_parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    load_parser.add_argument("--output", type=Path, required=True, help="CSV output path for request logs")

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
        )
        asyncio.run(run_load_test(cfg, output_path=args.output))
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
