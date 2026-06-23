"""vLLM Prometheus metrics — scrape per turn, parse, derive."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

from coding_agents.trace_collection.pipeline.utils.config import (
    JsonlWriter,
    instance_dir,
)

# ── Prometheus metric names ────────────────────────────────────────────────

PREFILL_SUM = "vllm:request_prefill_time_seconds_sum"
DECODE_SUM = "vllm:request_decode_time_seconds_sum"
QUEUE_SUM = "vllm:request_queue_time_seconds_sum"
TPOT_SUM = "vllm:request_time_per_output_token_seconds_sum"
E2E_SUM = "vllm:e2e_request_latency_seconds_sum"
PROMPT_TOKENS_SUM = "vllm:request_prompt_tokens_sum"
GEN_TOKENS_SUM = "vllm:request_generation_tokens_sum"
PREFILL_KV_COMPUTED_SUM = "vllm:request_prefill_kv_computed_tokens_sum"
FINISH_REASON = "vllm:request_success_total"
KV_USAGE_PCT = "vllm:kv_cache_usage_perc"
PREFIX_CACHE_HITS = "vllm:prefix_cache_hits_total"
EXTERNAL_PREFIX_CACHE_HITS = "vllm:external_prefix_cache_hits_total"
REQUEST_COUNT = "vllm:request_prompt_tokens_count"
FINISHED_REASONS = ("stop", "length", "abort", "error", "repetition")


# ── Main class ─────────────────────────────────────────────────────────────


class MetricsScraper:
    """Scrape vLLM's Prometheus /metrics once per turn on a background thread.

    enabled=False (Anthropic backend): no-op context manager."""

    def __init__(
        self,
        url: str,
        save_dir: Path,
        instance_id: str,
        poll_interval_s: float = 0.1,
        enabled: bool = True,
        fn: Callable[[dict], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self.instance_id = instance_id
        self.out_dir = save_dir / "runs"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._poller = Poller(
            url=url,
            instance_id=instance_id,
            out_dir=self.out_dir,
            poll_interval_s=poll_interval_s,
            fn=fn,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MetricsScraper":
        if not self.enabled:
            return self
        (instance_dir(self.out_dir, self.instance_id) / "engine_metrics.jsonl").unlink(
            missing_ok=True
        )
        self._thread = threading.Thread(
            target=self._poller.run_blocking,
            args=(self._stop,),
            name="vllm-metrics-scraper",
            daemon=True,
        )
        self._thread.start()
        logger.info("started metrics scraper thread for {}", self.instance_id)
        return self

    def __exit__(self, *exc) -> bool:
        if not self.enabled:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("stopped metrics scraper thread for {}", self.instance_id)
        return False


# ── Supporting classes ─────────────────────────────────────────────────────


class Poller:
    """Poll vLLM /metrics and write one JSONL row per detected request."""

    def __init__(
        self,
        url: str,
        instance_id: str,
        out_dir: Path,
        poll_interval_s: float = 0.1,
        fn: Callable[[dict], None] | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._instance_id = instance_id
        self._out = JsonlWriter(out_dir, "engine_metrics.jsonl")
        self._poll_interval_s = poll_interval_s
        self._fn = fn

    def run_blocking(self, stop_event: threading.Event) -> None:
        asyncio.run(self.run(stop_event))

    async def fetch_metrics_snapshot(self, client: httpx.AsyncClient) -> "Snapshot":
        ts = time.time()  # stamp before the round-trip
        resp = await client.get(f"{self._url}/metrics", timeout=10.0)
        return Snapshot.from_metrics(parse_raw_response(resp.text), ts=ts)

    async def run(self, stop_event: threading.Event) -> None:
        async with httpx.AsyncClient() as client:
            baseline: Snapshot | None = None
            while baseline is None and not stop_event.is_set():
                try:
                    baseline = await self.fetch_metrics_snapshot(client)
                except httpx.HTTPError as exc:
                    logger.warning("baseline scrape error: {!r}", exc)
                    await asyncio.sleep(self._poll_interval_s)
            if baseline is not None:
                logger.info("baseline scrape: request_count={}", baseline.request_count)

            peak_kv_usage = 0.0
            while not stop_event.is_set():
                await asyncio.sleep(self._poll_interval_s)
                try:
                    latest = await self.fetch_metrics_snapshot(client)
                except httpx.HTTPError as exc:
                    logger.warning("scrape error: {!r}", exc)
                    baseline = None
                    peak_kv_usage = 0.0
                    continue
                if baseline is None:
                    baseline = latest
                    continue

                peak_kv_usage = max(peak_kv_usage, latest.kv_usage_pct)
                new_completions = latest.request_count - baseline.request_count
                if new_completions >= 1:
                    row = compute_turn_metrics(
                        before=baseline,
                        after=latest,
                        peak_kv_usage=peak_kv_usage,
                        n=new_completions,
                    )
                    self._out.write(instance_id=self._instance_id, row=row)
                    if self._fn is not None:
                        self._fn(row)
                    baseline = latest
                    peak_kv_usage = 0.0


@dataclass(frozen=True)
class Snapshot:
    request_count: int
    timing_seconds_sum: dict[str, float]
    prompt_tokens: int
    gen_tokens: int
    prefill_kv_computed: int
    finished_reason_counts: dict[str, int]
    kv_usage_pct: float
    prefix_cache_hits: int
    external_prefix_cache_hits: int
    ts: float

    @classmethod
    def from_metrics(
        cls, metrics: dict[str, float], ts: float | None = None
    ) -> "Snapshot":
        """Parse a raw Prometheus metrics dict into a Snapshot.

        ``ts`` should be captured *before* the HTTP fetch so the timestamp
        reflects when the scrape was initiated, not when parsing completed.
        Defaults to ``time.time()`` when not provided.
        """
        timings = {
            "prefill": extract_metric(metrics=metrics, name_prefix=PREFILL_SUM),
            "decode": extract_metric(metrics=metrics, name_prefix=DECODE_SUM),
            "queue": extract_metric(metrics=metrics, name_prefix=QUEUE_SUM),
            "tpot": extract_metric(metrics=metrics, name_prefix=TPOT_SUM),
            "e2e": extract_metric(metrics=metrics, name_prefix=E2E_SUM),
        }
        finished = {
            reason: int(
                extract_metric(
                    metrics=metrics,
                    name_prefix=FINISH_REASON,
                    label_substring=f'finished_reason="{reason}"',
                )
            )
            for reason in FINISHED_REASONS
        }
        return cls(
            request_count=int(
                extract_metric(metrics=metrics, name_prefix=REQUEST_COUNT)
            ),
            timing_seconds_sum=timings,
            prompt_tokens=int(
                extract_metric(metrics=metrics, name_prefix=PROMPT_TOKENS_SUM)
            ),
            gen_tokens=int(extract_metric(metrics=metrics, name_prefix=GEN_TOKENS_SUM)),
            prefill_kv_computed=int(
                extract_metric(metrics=metrics, name_prefix=PREFILL_KV_COMPUTED_SUM)
            ),
            finished_reason_counts=finished,
            kv_usage_pct=extract_metric(metrics=metrics, name_prefix=KV_USAGE_PCT),
            prefix_cache_hits=int(
                extract_metric(metrics=metrics, name_prefix=PREFIX_CACHE_HITS)
            ),
            external_prefix_cache_hits=int(
                extract_metric(metrics=metrics, name_prefix=EXTERNAL_PREFIX_CACHE_HITS)
            ),
            ts=ts if ts is not None else time.time(),
        )


def compute_turn_metrics(
    before: Snapshot,
    after: Snapshot,
    peak_kv_usage: float = 0.0,
    n: int = 1,
) -> dict:
    """Build one row of per-request average measurements from two consecutive snapshots.

    When n > 1 requests completed in the same tick, latency and token fields are
    divided by n so each row represents per-request averages rather than sums.
    """

    def delta_ms(name: str) -> float:
        return (
            after.timing_seconds_sum[name] - before.timing_seconds_sum[name]
        ) * 1000.0

    isl = (after.prompt_tokens - before.prompt_tokens) / n
    osl = (after.gen_tokens - before.gen_tokens) / n
    isl_new = (after.prefill_kv_computed - before.prefill_kv_computed) / n

    stop_reason = ""
    for reason in FINISHED_REASONS:
        if (
            after.finished_reason_counts.get(reason, 0)
            - before.finished_reason_counts.get(reason, 0)
        ) >= 1:
            stop_reason = reason
            break

    return {
        "ts": round(before.ts, 3),
        "n": n,
        "isl": round(isl, 1),
        "osl": round(osl, 1),
        "isl_new": round(isl_new, 1),
        "prefill_ms": round(delta_ms("prefill") / n, 2),
        "decode_ms": round(delta_ms("decode") / n, 2),
        "queue_ms": round(delta_ms("queue") / n, 2),
        "itl_ms": round(delta_ms("tpot") / n, 3) if osl > 0 else None,
        "e2e_ms": round(delta_ms("e2e") / n, 2),
        "stop_reason": stop_reason,
        "kv_cache_usage_pct_peak": round(peak_kv_usage * 100, 3),
        "prefix_cache_hits": after.prefix_cache_hits - before.prefix_cache_hits,
        "external_prefix_cache_hits": after.external_prefix_cache_hits
        - before.external_prefix_cache_hits,
    }


# ── Private ────────────────────────────────────────────────────────────────

_METRIC_LINE = re.compile(r"^(.+?)\s+([0-9eE.+\-]+|NaN|\+Inf|\-Inf)$")


def parse_raw_response(body: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        try:
            parsed[match.group(1)] = float(match.group(2))
        except ValueError:
            pass
    return parsed


def extract_metric(
    metrics: dict[str, float], name_prefix: str, label_substring: str = ""
) -> float:
    for name, value in metrics.items():
        if not name.startswith(name_prefix):
            continue
        if label_substring and label_substring not in name:
            continue
        return value
    return 0.0
