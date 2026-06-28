"""GPU power monitoring via pynvml."""

from __future__ import annotations

import threading

import pynvml
from loguru import logger


class GpuPowerMonitor:
    """Sample total GPU power draw on a background thread via pynvml.

    Use as a context manager; read .watts_measured after exit for the
    average total watts (summed across all GPUs) during the run.
    Returns None if pynvml is unavailable or no samples were collected.
    """

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []
        self.watts_measured: float | None = None

    def __enter__(self) -> "GpuPowerMonitor":
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            logger.warning("pynvml init failed, power capture disabled: {}", exc)
            return self
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-power-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
        if self._samples:
            self.watts_measured = round(sum(self._samples) / len(self._samples), 1)
            logger.info(
                "measured avg GPU power: {:.1f} W ({} samples)",
                self.watts_measured,
                len(self._samples),
            )
        return False

    def _run(self) -> None:
        try:
            n = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        except pynvml.NVMLError as exc:
            logger.warning("gpu power monitor failed to enumerate devices: {}", exc)
            return
        while not self._stop.is_set():
            try:
                total_w = sum(
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
                    for h in handles
                )
                self._samples.append(total_w)
            except pynvml.NVMLError as exc:
                logger.warning("gpu power sample error: {}", exc)
            self._stop.wait(self._poll_interval_s)
