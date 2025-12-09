"""NVML-backed GPU power logging helpers."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PowerSample:
    """A single power measurement."""

    timestamp_s: float
    power_w: float
    gpu_util_percent: int
    mem_used_bytes: int
    mem_total_bytes: int


class GpuPowerLogger:
    """Logs GPU power and utilization using NVML."""

    def __init__(self, device_index: int = 0) -> None:
        try:
            import pynvml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("pynvml is required for GPU power logging") from exc

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_index = device_index

    def sample(self) -> PowerSample:
        """Capture a single power/utilization sample."""
        now = time.time()
        power_w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        return PowerSample(
            timestamp_s=now,
            power_w=power_w,
            gpu_util_percent=util.gpu,
            mem_used_bytes=mem.used,
            mem_total_bytes=mem.total,
        )

    def record(
        self,
        duration_s: float,
        interval_s: float,
        output_path: Path,
    ) -> Iterable[PowerSample]:
        """Record power samples for a fixed duration.

        Returns the samples as they are written to disk.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples: list[PowerSample] = []
        end_time = time.time() + duration_s

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "power_W", "gpu_util_percent", "mem_used_bytes", "mem_total_bytes"])

            while time.time() < end_time:
                sample = self.sample()
                samples.append(sample)
                writer.writerow(
                    [sample.timestamp_s, sample.power_w, sample.gpu_util_percent, sample.mem_used_bytes, sample.mem_total_bytes]
                )
                time.sleep(interval_s)

        return samples

    def close(self) -> None:
        """Release NVML resources."""
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            # Best-effort cleanup; NVML shutdown failures should not crash callers.
            pass


def log_gpu_power(
    duration_s: float,
    interval_s: float,
    output_path: Path,
    device_index: int = 0,
) -> Iterable[PowerSample]:
    """Convenience wrapper to log power without manually managing the logger."""
    logger = GpuPowerLogger(device_index=device_index)
    try:
        return logger.record(duration_s=duration_s, interval_s=interval_s, output_path=output_path)
    finally:
        logger.close()
