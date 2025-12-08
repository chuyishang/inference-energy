"""Utilities to integrate power logs and attribute energy to requests/tokens."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


def _read_power(path: Path) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_s"])
            p = float(row["power_W"])
            samples.append((t, p))
    if len(samples) < 2:
        raise ValueError("Need at least two power samples to integrate energy")
    samples.sort(key=lambda x: x[0])
    return samples


def integrate_energy(power_log: Path, idle_power_w: float = 0.0) -> EnergyBreakdown:
    """Integrate power samples to energy using trapezoidal rule."""
    samples = _read_power(power_log)
    total_energy_j = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total_energy_j += 0.5 * (p0 + p1) * dt

    duration_s = samples[-1][0] - samples[0][0]
    active_energy_j = total_energy_j - idle_power_w * duration_s
    active_energy_j = max(active_energy_j, 0.0)
    return EnergyBreakdown(
        total_energy_j=total_energy_j,
        active_energy_j=active_energy_j,
        duration_s=duration_s,
        idle_power_w=idle_power_w,
    )


def attribute_energy(
    requests_log: Path,
    active_energy_j: float,
) -> TokenAttribution:
    """Allocate active energy to requests proportional to completion tokens."""
    completion_tokens: list[int] = []
    with requests_log.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            status_code = int(row["status_code"])
            if status_code != 200:
                continue
            completion_tokens.append(int(row["completion_tokens"]))

    total_tokens = sum(completion_tokens)
    if total_tokens <= 0:
        raise ValueError("No completion tokens recorded; cannot attribute energy")

    energy_per_token = active_energy_j / total_tokens
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
    """Convenience helper to compute both energy integration and attribution."""
    energy = integrate_energy(power_log=power_log, idle_power_w=idle_power_w)
    tokens = attribute_energy(requests_log=requests_log, active_energy_j=energy.active_energy_j)
    return energy, tokens
