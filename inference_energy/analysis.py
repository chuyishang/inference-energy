"""Utilities to integrate power logs and attribute energy to requests/tokens."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PowerStats:
    """Statistics extracted from power log."""

    avg_power_w: float
    peak_power_w: float
    avg_gpu_util_percent: float
    avg_mem_util_percent: float
    mem_total_bytes: int


@dataclass
class RequestStats:
    """Statistics extracted from request log."""

    total_requests: int
    successful_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_s: float
    first_timestamp_s: float
    last_timestamp_s: float


@dataclass
class ComprehensiveMetrics:
    """All measured and derived metrics for inference energy analysis."""

    # === Primary Measurements (M1-M10) ===
    m1_total_energy_j: float  # Total energy consumed (Joules)
    m2_total_tokens: int  # Total number of tokens generated
    m3_total_time_s: float  # Total wall-clock time (seconds)
    m4_avg_prefill_time_s: float  # Average prefill time per request (NOTE: approximated from latency)
    m5_avg_decode_time_per_token_s: float  # Average decode time per token (NOTE: approximated)
    m6_avg_power_w: float  # Average power draw (Watts)
    m7_peak_power_w: float  # Peak power draw (Watts)
    m8_avg_gpu_util_percent: float  # Average GPU utilization (%)
    m9_avg_mem_util_percent: float  # Average memory utilization (%)
    m10_flops_measured: float | None  # Actual FLOPs achieved (requires profiler, None if unavailable)

    # === Derived Metrics (D1-D4) ===
    d1_energy_per_token_j: float  # E_total / N_tokens [Joules/token]
    d2_throughput_tokens_per_s: float  # N_tokens / T_total [tokens/second]
    d3_power_efficiency_flops_per_w: float | None  # FLOPs_measured / P_avg [FLOPs/Watt]
    d4_memory_bandwidth_util_percent: float | None  # (Model_size / M5) / GPU_memory_BW × 100 [%]

    # === Additional Context ===
    idle_power_w: float  # Idle baseline power
    active_energy_j: float  # Total energy minus idle baseline
    total_requests: int  # Total requests sent
    successful_requests: int  # Requests with status 200
    avg_latency_s: float  # Average request latency
    mem_total_gb: float  # Total GPU memory in GB


def _read_power_stats(path: Path) -> tuple[list[tuple[float, float]], PowerStats]:
    """Read power log and compute statistics."""
    timestamps = []
    powers = []
    gpu_utils = []
    mem_used = []
    mem_total = None

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp_s"]))
            powers.append(float(row["power_W"]))
            gpu_utils.append(float(row["gpu_util_percent"]))
            mem_used.append(float(row["mem_used_bytes"]))
            if mem_total is None:
                mem_total = int(row["mem_total_bytes"])

    if len(powers) < 2:
        raise ValueError("Need at least two power samples to integrate energy")

    samples = list(zip(timestamps, powers))
    samples.sort(key=lambda x: x[0])

    stats = PowerStats(
        avg_power_w=sum(powers) / len(powers),
        peak_power_w=max(powers),
        avg_gpu_util_percent=sum(gpu_utils) / len(gpu_utils),
        avg_mem_util_percent=100.0 * sum(mem_used) / len(mem_used) / mem_total if mem_total else 0.0,
        mem_total_bytes=mem_total or 0,
    )

    return samples, stats


def _integrate_energy(power_samples: list[tuple[float, float]]) -> float:
    """Integrate power samples to energy using trapezoidal rule."""
    total_energy_j = 0.0
    for (t0, p0), (t1, p1) in zip(power_samples, power_samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total_energy_j += 0.5 * (p0 + p1) * dt
    return total_energy_j


def _read_request_stats(path: Path) -> RequestStats:
    """Read request log and compute statistics."""
    latencies = []
    prompt_tokens = []
    completion_tokens = []
    timestamps = []
    successful = 0
    total = 0

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            status = int(row["status_code"])
            if status == 200:
                successful += 1
                latencies.append(float(row["latency_s"]))
                prompt_tokens.append(int(row["prompt_tokens"]))
                completion_tokens.append(int(row["completion_tokens"]))
            timestamps.append(float(row["timestamp_s"]))

    if not timestamps:
        raise ValueError("Request log is empty")

    return RequestStats(
        total_requests=total,
        successful_requests=successful,
        total_prompt_tokens=sum(prompt_tokens),
        total_completion_tokens=sum(completion_tokens),
        avg_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
        first_timestamp_s=min(timestamps),
        last_timestamp_s=max(timestamps),
    )


def analyze_comprehensive(
    power_log: Path,
    requests_log: Path,
    idle_power_w: float = 0.0,
    model_size_bytes: int | None = None,
    gpu_memory_bw_bytes_per_s: float | None = None,
    flops_measured: float | None = None,
) -> ComprehensiveMetrics:
    """Comprehensive analysis computing all M1-M10 and D1-D4 metrics.

    Args:
        power_log: Path to power CSV log
        requests_log: Path to requests CSV log
        idle_power_w: Idle baseline power in Watts
        model_size_bytes: Model size in bytes (for D4 calculation)
        gpu_memory_bw_bytes_per_s: Theoretical GPU memory bandwidth in bytes/s (for D4)
        flops_measured: Measured FLOPs from profiler (optional, for M10/D3)

    Returns:
        ComprehensiveMetrics with all measurements and derived values
    """
    # Read and analyze power log
    power_samples, power_stats = _read_power_stats(power_log)
    total_energy_j = _integrate_energy(power_samples)
    duration_s = power_samples[-1][0] - power_samples[0][0]
    active_energy_j = max(0.0, total_energy_j - idle_power_w * duration_s)

    # Read and analyze request log
    request_stats = _read_request_stats(requests_log)

    # === Primary Measurements ===
    m1 = total_energy_j
    m2 = request_stats.total_completion_tokens
    m3 = duration_s

    # M4 & M5: Prefill/decode time approximation
    # Since vLLM OpenAI API doesn't separate these, we estimate:
    # - Prefill: roughly proportional to prompt length
    # - Decode: roughly latency - prefill time, divided by output tokens
    # This is a rough approximation; for exact values, use vLLM metrics endpoint
    avg_prompt_tokens = request_stats.total_prompt_tokens / request_stats.successful_requests if request_stats.successful_requests else 0
    avg_completion_tokens = m2 / request_stats.successful_requests if request_stats.successful_requests else 0
    avg_latency = request_stats.avg_latency_s

    # Rough estimate: prefill takes ~10-30% of total latency for typical workloads
    # This is a placeholder - ideally get from vLLM metrics
    estimated_prefill_time = avg_latency * 0.2 if avg_latency > 0 else 0.0
    estimated_decode_time_total = avg_latency - estimated_prefill_time

    m4 = estimated_prefill_time
    m5 = estimated_decode_time_total / avg_completion_tokens if avg_completion_tokens > 0 else 0.0

    m6 = power_stats.avg_power_w
    m7 = power_stats.peak_power_w
    m8 = power_stats.avg_gpu_util_percent
    m9 = power_stats.avg_mem_util_percent
    m10 = flops_measured

    # === Derived Metrics ===
    d1 = m1 / m2 if m2 > 0 else 0.0
    d2 = m2 / m3 if m3 > 0 else 0.0
    d3 = m10 / m6 if m10 is not None and m6 > 0 else None

    # D4: Memory bandwidth utilization
    # For each token generated, you load all model weights once
    # Bandwidth required = Model_size / Time_per_token = Model_size / M5
    d4 = None
    if model_size_bytes and gpu_memory_bw_bytes_per_s and m5 > 0:
        bandwidth_required = model_size_bytes / m5
        d4 = 100.0 * bandwidth_required / gpu_memory_bw_bytes_per_s

    return ComprehensiveMetrics(
        # Primary measurements
        m1_total_energy_j=m1,
        m2_total_tokens=m2,
        m3_total_time_s=m3,
        m4_avg_prefill_time_s=m4,
        m5_avg_decode_time_per_token_s=m5,
        m6_avg_power_w=m6,
        m7_peak_power_w=m7,
        m8_avg_gpu_util_percent=m8,
        m9_avg_mem_util_percent=m9,
        m10_flops_measured=m10,
        # Derived metrics
        d1_energy_per_token_j=d1,
        d2_throughput_tokens_per_s=d2,
        d3_power_efficiency_flops_per_w=d3,
        d4_memory_bandwidth_util_percent=d4,
        # Additional context
        idle_power_w=idle_power_w,
        active_energy_j=active_energy_j,
        total_requests=request_stats.total_requests,
        successful_requests=request_stats.successful_requests,
        avg_latency_s=avg_latency,
        mem_total_gb=power_stats.mem_total_bytes / (1024**3),
    )


# Backward compatibility functions
@dataclass
class EnergyBreakdown:
    total_energy_j: float
    active_energy_j: float
    duration_s: float
    idle_power_w: float


@dataclass
class TokenAttribution:
    total_completion_tokens: int
    energy_per_completion_token_j: float
    energy_per_request: list[float]


def integrate_energy(power_log: Path, idle_power_w: float = 0.0) -> EnergyBreakdown:
    """Integrate power samples to energy using trapezoidal rule (legacy API)."""
    samples, _ = _read_power_stats(power_log)
    total_energy_j = _integrate_energy(samples)
    duration_s = samples[-1][0] - samples[0][0]
    active_energy_j = max(0.0, total_energy_j - idle_power_w * duration_s)
    return EnergyBreakdown(
        total_energy_j=total_energy_j,
        active_energy_j=active_energy_j,
        duration_s=duration_s,
        idle_power_w=idle_power_w,
    )


def attribute_energy(requests_log: Path, active_energy_j: float) -> TokenAttribution:
    """Allocate active energy to requests proportional to completion tokens (legacy API)."""
    request_stats = _read_request_stats(requests_log)
    total_tokens = request_stats.total_completion_tokens

    if total_tokens <= 0:
        raise ValueError("No completion tokens recorded; cannot attribute energy")

    energy_per_token = active_energy_j / total_tokens

    # Re-read to get per-request token counts
    completion_tokens = []
    with requests_log.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["status_code"]) == 200:
                completion_tokens.append(int(row["completion_tokens"]))

    energy_per_request = [energy_per_token * tokens for tokens in completion_tokens]

    return TokenAttribution(
        total_completion_tokens=total_tokens,
        energy_per_completion_token_j=energy_per_token,
        energy_per_request=energy_per_request,
    )


def summarize(
    power_log: Path,
    requests_log: Path,
    idle_power_w: float = 0.0,
) -> tuple[EnergyBreakdown, TokenAttribution]:
    """Convenience helper to compute both energy integration and attribution (legacy API)."""
    energy = integrate_energy(power_log=power_log, idle_power_w=idle_power_w)
    tokens = attribute_energy(requests_log=requests_log, active_energy_j=energy.active_energy_j)
    return energy, tokens
