"""Async load generator for vLLM's OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx


@dataclass
class LoadGenConfig:
    """Configuration for the load generator."""

    endpoint: str
    model: str
    prompts: Sequence[str] | None
    duration_s: float
    concurrency: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.0
    request_timeout_s: float = 30.0
    synthetic_prompts: bool = False
    prompt_mean_tokens: int = 256
    prompt_std_tokens: int = 64
    synthetic_vocab_size: int = 5000

    def validate(self) -> None:
        if not self.synthetic_prompts and not self.prompts:
            raise ValueError("Provide a prompts list or enable synthetic_prompts")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.synthetic_prompts:
            if self.prompt_mean_tokens <= 0 or self.prompt_std_tokens < 0:
                raise ValueError("prompt_mean_tokens must be >0 and prompt_std_tokens >=0 for synthetic prompts")
            if self.synthetic_vocab_size <= 0:
                raise ValueError("synthetic_vocab_size must be >0")


@dataclass
class RequestLog:
    """A record for a single request."""

    timestamp_s: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    status_code: int
    error: str | None


async def _issue_request(client: httpx.AsyncClient, cfg: LoadGenConfig, prompt: str) -> RequestLog:
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "stream": False,
    }

    started = time.time()
    try:
        resp = await client.post("/v1/chat/completions", json=payload)
        latency = time.time() - started
        prompt_tokens = resp.json().get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = resp.json().get("usage", {}).get("completion_tokens", 0)
        return RequestLog(
            timestamp_s=started,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=resp.status_code,
            error=None if resp.is_success else resp.text,
        )
    except Exception as exc:
        latency = time.time() - started
        return RequestLog(
            timestamp_s=started,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            status_code=0,
            error=str(exc),
        )


def _sample_prompt(cfg: LoadGenConfig) -> str:
    if cfg.prompts:
        return random.choice(cfg.prompts)

    length = max(1, int(random.gauss(cfg.prompt_mean_tokens, cfg.prompt_std_tokens)))
    tokens = [f"tok{random.randint(0, cfg.synthetic_vocab_size)}" for _ in range(length)]
    return " ".join(tokens)


async def _worker(
    worker_id: int,
    cfg: LoadGenConfig,
    end_time: float,
    client: httpx.AsyncClient,
    writer: csv.writer,
) -> None:
    del worker_id  # unused but kept for future per-worker logging
    while time.time() < end_time:
        prompt = _sample_prompt(cfg)
        record = await _issue_request(client, cfg, prompt)
        writer.writerow(
            [
                record.timestamp_s,
                record.latency_s,
                record.prompt_tokens,
                record.completion_tokens,
                record.status_code,
                record.error or "",
            ]
        )


async def run_load_test(cfg: LoadGenConfig, output_path: Path) -> None:
    """Run a timed load test and write request logs to CSV."""
    cfg.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + cfg.duration_s
    async with httpx.AsyncClient(base_url=cfg.endpoint, timeout=cfg.request_timeout_s) as client:
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp_s", "latency_s", "prompt_tokens", "completion_tokens", "status_code", "error"]
            )

            async with asyncio.TaskGroup() as tg:
                for idx in range(cfg.concurrency):
                    tg.create_task(_worker(idx, cfg, deadline, client, writer))
