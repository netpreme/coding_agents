"""Async replay engine — worker dispatch, per-request timing, live display."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TextIO

import httpx
from loguru import logger
from tqdm import tqdm

from coding_agents.serving import (
    GpuPowerMonitor,
    MetricsScraper,
    VllmConfig,
    VllmServer,
    close_gpu_handles,
    open_gpu_handles,
    read_gpu_stats,
)

from .synthetic import build_turn_request


@dataclass(frozen=True)
class ReplayStats:
    """Aggregate outcomes returned after a completed replay run."""

    concurrency: int
    duration_s: float
    turns_completed: int
    turns_failed: int
    sessions_completed: int
    started_at: float
    ended_at: float

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


@dataclass
class LiveStats:
    """Counters shared between workers and the display loop.

    Updated by:
      - async workers  → turns_completed, turns_failed, sessions_completed,
                         ttft_sum, ttft_count
      - MetricsScraper callback (main.py) → osl_total, server
    All writes happen from a single OS thread (asyncio event loop or the
    MetricsScraper background thread), so no explicit locking is needed for
    display purposes.
    """

    turns_completed: int = 0
    turns_failed: int = 0
    sessions_completed: int = 0
    ttft_sum: float = 0.0  # running sum of client-side TTFT values (ms)
    ttft_count: int = 0  # number of TTFT samples
    osl_total: float = (
        0.0  # cumulative output tokens (updated by MetricsScraper callback)
    )
    server: dict = field(default_factory=dict)  # latest Prometheus row


class ReplayWriter:
    """Writes turn events to a JSONL file, serialising concurrent writes."""

    def __init__(self, events_path: Path) -> None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: TextIO = events_path.open("w")
        self._lock = asyncio.Lock()

    async def write_event(self, row: dict) -> None:
        async with self._lock:
            self._events.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._events.flush()

    def close(self) -> None:
        self._events.close()


def format_live_metrics(counter: LiveStats, elapsed_s: float, gpu_handles: list) -> str:
    """Format the live metrics bar description from current counter state."""
    parts = []
    s = counter.server
    if s.get("n") is not None:
        parts.append(f"n={s['n']}")
    isl = s.get("isl")
    isl_new = s.get("isl_new")
    if isl is not None:
        parts.append(f"isl={isl:.0f}")
    if s.get("osl") is not None:
        parts.append(f"osl={s['osl']:.0f}")
    if isl_new is not None:
        parts.append(f"isl_new={isl_new:.0f}")
    if isl and isl_new is not None and isl > 0:
        parts.append(f"prefix_hit={(isl - isl_new) / isl * 100:.1f}%")
    if s.get("prefill_ms") is not None:
        parts.append(f"prefill={s['prefill_ms']:.0f}ms")
    if s.get("decode_ms") is not None:
        parts.append(f"decode={s['decode_ms']:.0f}ms")
    if s.get("queue_ms") is not None:
        parts.append(f"queue={s['queue_ms']:.1f}ms")
    if s.get("itl_ms") is not None:
        parts.append(f"itl={s['itl_ms']:.1f}ms/tok")
    if s.get("e2e_ms") is not None:
        parts.append(f"e2e={s['e2e_ms']:.0f}ms")
    if s.get("kv_cache_usage_pct_peak") is not None:
        parts.append(f"kv_util={s['kv_cache_usage_pct_peak']:.1f}%")
    if counter.ttft_count > 0:
        parts.append(f"ttft={counter.ttft_sum / counter.ttft_count:.0f}ms")
    if elapsed_s > 0 and counter.osl_total > 0:
        parts.append(f"tok/s={counter.osl_total / elapsed_s:.0f}")
    gpu_util, gpu_mem = read_gpu_stats(gpu_handles)
    if gpu_util:
        parts.append(gpu_util)
    if gpu_mem:
        parts.append(gpu_mem)
    return ("  " + "  ".join(parts)) if parts else "  (waiting for first completions…)"


def replay_traces(
    sessions: list[dict],
    config: VllmConfig,
    save_dir: Path,
    phase_dir: Path,
    phase_id: str,
    concurrency: int,
    duration_s: float,
) -> tuple[ReplayStats, float | None]:
    """Start vLLM, replay captured traces with concurrent agents, return results.

    Manages the full lifecycle: vLLM server, Prometheus scraper, GPU power
    monitor, and the async worker pool. Always measures GPU power via pynvml.
    """
    live_stats = LiveStats()
    writer = ReplayWriter(events_path=phase_dir / "turn_events.jsonl")

    def _update_live_metrics(row: dict) -> None:
        live_stats.server.update(row)
        live_stats.osl_total += row.get("osl", 0.0) * row.get("n", 1)

    power_monitor = GpuPowerMonitor()
    try:
        with (
            VllmServer(config) as server,
            MetricsScraper(
                url=server.url,
                save_dir=save_dir,
                instance_id=phase_id,
                fn=_update_live_metrics,
            ),
            power_monitor,
        ):
            stats = asyncio.run(
                run_concurrent_replay(
                    sessions=sessions,
                    base_url=server.url,
                    model=server.model,
                    concurrency=concurrency,
                    duration_s=duration_s,
                    writer=writer,
                    live=live_stats,
                )
            )
    finally:
        writer.close()

    return stats, power_monitor.watts_measured


async def run_concurrent_replay(
    sessions: list[dict],
    base_url: str,
    model: str,
    concurrency: int,
    duration_s: float,
    writer: ReplayWriter | None = None,
    live: LiveStats | None = None,
) -> ReplayStats:
    """Dispatch ``concurrency`` virtual agents to replay sessions for ``duration_s`` seconds.

    Args:
        sessions:    Pre-loaded sessions to replay (cycled as needed).
        base_url:    vLLM server base URL.
        model:       Model name sent in each request.
        concurrency: Number of concurrent virtual agents.
        duration_s:  How long to run in seconds.
        writer:      Optional writer for per-turn event rows.
        live:        Optional shared counter for live display.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if not sessions:
        raise ValueError("at least one session is required")

    counter = live if live is not None else LiveStats()
    started_at = time.time()
    deadline = time.monotonic() + duration_s
    logger.info(
        "replay start: {} sessions, concurrency={}, duration={:.0f}s",
        len(sessions),
        concurrency,
        duration_s,
    )

    gpu_handles = open_gpu_handles()
    progress_bar = tqdm(
        total=int(duration_s),
        desc="replay",
        unit="s",
        dynamic_ncols=True,
        file=sys.stderr,
        position=0,
    )
    metrics_bar = tqdm(
        bar_format="{desc}",
        desc="  (waiting for first completions…)",
        dynamic_ncols=True,
        file=sys.stderr,
        position=1,
    )
    run_start_time = time.monotonic()

    try:
        refresh_task = asyncio.create_task(
            _update_progress_display(
                progress_bar=progress_bar,
                metrics_bar=metrics_bar,
                counter=counter,
                deadline=deadline,
                run_start_time=run_start_time,
                gpu_handles=gpu_handles,
            )
        )
        try:
            async with httpx.AsyncClient(
                base_url=base_url.rstrip("/"), timeout=None
            ) as client:
                worker_results = await asyncio.gather(
                    *[
                        _run_virtual_agent(
                            sessions=sessions,
                            client=client,
                            model=model,
                            deadline=deadline,
                            worker_id=worker_id,
                            writer=writer,
                            counter=counter,
                        )
                        for worker_id in range(concurrency)
                    ]
                )
        finally:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
    finally:
        progress_bar.close()
        metrics_bar.close()
        if gpu_handles:
            close_gpu_handles()

    stats = ReplayStats(
        concurrency=concurrency,
        duration_s=duration_s,
        turns_completed=sum(result.turns_completed for result in worker_results),
        turns_failed=sum(result.turns_failed for result in worker_results),
        sessions_completed=sum(result.sessions_completed for result in worker_results),
        started_at=started_at,
        ended_at=time.time(),
    )
    logger.info("replay done: {} turns completed", stats.turns_completed)
    return stats


class _AgentResult(NamedTuple):
    turns_completed: int
    turns_failed: int
    sessions_completed: int


async def _replay_session_turns(
    session: dict,
    client: httpx.AsyncClient,
    model: str,
    deadline: float,
    worker_id: int,
    cycle: int,
    writer: ReplayWriter | None,
    counter: LiveStats,
) -> _AgentResult:
    """Send each turn of a session to vLLM in order, stopping at the deadline."""
    turns_completed = 0
    turns_failed = 0

    for turn_idx, turn in enumerate(session["turns"]):
        if time.monotonic() >= deadline:
            break

        if turn["tool_exec_ms"] is not None:
            await asyncio.sleep(turn["tool_exec_ms"] / 1000)

        request_body = build_turn_request(
            session=session, turn_idx=turn_idx, model=model
        )
        started_at = time.time()
        first_byte_at: float | None = None
        status_code: int | None = None
        error: str | None = None

        try:
            async with client.stream(
                "POST", "/v1/messages", json=request_body
            ) as response:
                status_code = response.status_code
                async for chunk in response.aiter_bytes():
                    if first_byte_at is None and chunk:
                        first_byte_at = time.time()
            turns_completed += 1
            counter.turns_completed += 1
            if first_byte_at is not None:
                counter.ttft_sum += (first_byte_at - started_at) * 1000
                counter.ttft_count += 1
        except httpx.HTTPError as exc:
            error = repr(exc)
            turns_failed += 1
            counter.turns_failed += 1
            logger.warning(
                "session {} turn {} failed: {!r}", session["instance_id"], turn_idx, exc
            )

        ended_at = time.time()

        if writer is not None:
            await writer.write_event(
                {
                    "worker_id": worker_id,
                    "cycle": cycle,
                    "instance_id": session["instance_id"],
                    "turn": turn_idx,
                    **turn,
                    "started_at": round(started_at, 3),
                    "first_byte_at": round(first_byte_at, 3) if first_byte_at else None,
                    "ended_at": round(ended_at, 3),
                    "ttft_ms": (
                        round((first_byte_at - started_at) * 1000, 2)
                        if first_byte_at
                        else None
                    ),
                    "duration_ms": round((ended_at - started_at) * 1000, 2),
                    "status_code": status_code,
                    "error": error,
                }
            )

    sessions_completed = 1 if turns_completed + turns_failed > 0 else 0
    counter.sessions_completed += sessions_completed
    return _AgentResult(turns_completed, turns_failed, sessions_completed)


async def _run_virtual_agent(
    sessions: list[dict],
    client: httpx.AsyncClient,
    model: str,
    deadline: float,
    worker_id: int,
    writer: ReplayWriter | None,
    counter: LiveStats,
) -> _AgentResult:
    """Cycle through sessions as a single virtual agent until the deadline."""
    turns_completed = 0
    turns_failed = 0
    sessions_completed = 0
    session_idx = worker_id
    while time.monotonic() < deadline:
        session = sessions[session_idx % len(sessions)]
        cycle = session_idx // len(sessions)
        session_idx += 1
        result = await _replay_session_turns(
            session=session,
            client=client,
            model=model,
            deadline=deadline,
            worker_id=worker_id,
            cycle=cycle,
            writer=writer,
            counter=counter,
        )
        turns_completed += result.turns_completed
        turns_failed += result.turns_failed
        sessions_completed += result.sessions_completed
    return _AgentResult(turns_completed, turns_failed, sessions_completed)


async def _update_progress_display(
    progress_bar: tqdm,
    metrics_bar: tqdm,
    counter: LiveStats,
    deadline: float,
    run_start_time: float,
    gpu_handles: list,
) -> None:
    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        progress_bar.update(1)
        progress_bar.set_postfix(
            turns=counter.turns_completed,
            sessions=counter.sessions_completed,
            failed=counter.turns_failed,
            refresh=True,
        )
        elapsed = time.monotonic() - run_start_time
        metrics_bar.set_description(
            format_live_metrics(
                counter=counter, elapsed_s=elapsed, gpu_handles=gpu_handles
            )
        )
