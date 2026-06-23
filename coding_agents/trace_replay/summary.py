"""Aggregate replay results into a summary JSON and CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .engine import ReplayStats


def compute_and_write_summary(
    phase_dir: Path,
    stats: ReplayStats,
    concurrency: int,
    duration_s: float,
    gpus: int,
    gpu_name: str | None,
    power_watts: float | None,
) -> dict:
    """Read per-turn output files and write summary.json, returning the summary dict.

    Latency percentiles (ttft_ms) come from ``turn_events.jsonl`` — the
    accurate client-side per-request measurements — not from the Prometheus
    rows, which blend multiple completions when concurrency is high.

    Throughput and server-internal metrics (itl, e2e, kv cache) come from
    ``engine_metrics.jsonl``.
    """
    metric_rows = _read_jsonl_rows(phase_dir / "engine_metrics.jsonl")
    event_rows = _read_jsonl_rows(phase_dir / "turn_events.jsonl")

    measurement_s = stats.elapsed_s or duration_s
    power_mw = (power_watts / 1_000_000) if power_watts else None

    throughput = _compute_throughput_stats(
        metric_rows=metric_rows,
        event_rows=event_rows,
        stats=stats,
        measurement_s=measurement_s,
    )
    interactivity = _compute_latency_stats(
        metric_rows=metric_rows, event_rows=event_rows
    )
    normalization = _compute_efficiency_ratios(
        throughput=throughput,
        concurrency=concurrency,
        gpus=gpus,
        power_mw=power_mw,
    )

    summary = {
        "config": {
            "concurrency": concurrency,
            "duration_s": duration_s,
            "gpus": gpus,
            "gpu_name": gpu_name,
            "power_watts": power_watts,
            "power_mw": power_mw,
        },
        "replay": asdict(stats),
        "counts": {
            "completed_turns": stats.turns_completed,
            "failed_turns": stats.turns_failed,
            "completed_sessions": stats.sessions_completed,
            "metric_rows": len(metric_rows),
            "event_rows": len(event_rows),
            "total_output_tokens": throughput["total_output_tokens"],
        },
        "throughput": throughput,
        "interactivity": interactivity,
        "normalization": normalization,
    }
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def save_config(
    save_dir: Path,
    trace_path: Path,
    capture_config: dict,
    cli_args: dict,
    duration_s: float,
    gpus: int,
    gpu_name: str | None,
) -> None:
    """Write run_config.json to save_dir with the replay parameters."""
    payload = {
        "trace_path": str(trace_path),
        "capture_model": capture_config.get("model"),
        "capture_serving_config": capture_config.get("serving_config"),
        "cli_args": {
            k: str(v) if isinstance(v, Path) else v for k, v in cli_args.items()
        },
        "concurrency": cli_args["concurrency"],
        "duration_s": duration_s,
        "gpus": gpus,
        "gpu_name": gpu_name,
    }
    (save_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_summary_csv(save_dir: Path, summaries: list[dict]) -> None:
    """Write a flat CSV with one row per run, suitable for comparing sweeps."""
    rows = [_flatten_for_csv(summary) for summary in summaries]
    if not rows:
        return
    path = save_dir / "summary.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compute_efficiency_ratios(
    throughput: dict,
    concurrency: int,
    gpus: int,
    power_mw: float | None,
) -> dict:
    tps = throughput["system_output_tokens_per_s"]
    turns_per_s = throughput["turns_per_s"]
    return {
        "agents_per_gpu": _safe_divide(numerator=concurrency, denominator=gpus),
        "output_tokens_per_s_per_gpu": _safe_divide(numerator=tps, denominator=gpus),
        "turns_per_s_per_gpu": _safe_divide(numerator=turns_per_s, denominator=gpus),
        "agents_per_mw": _safe_divide(numerator=concurrency, denominator=power_mw),
        "output_tokens_per_s_per_mw": _safe_divide(numerator=tps, denominator=power_mw),
        "turns_per_s_per_mw": _safe_divide(numerator=turns_per_s, denominator=power_mw),
    }


def _compute_latency_stats(metric_rows: list[dict], event_rows: list[dict]) -> dict:
    # ttft: use per-request client-side values from turn_events.jsonl
    ttft_ms = [row["ttft_ms"] for row in event_rows if row.get("ttft_ms") is not None]

    # server-side latency breakdown from Prometheus (per-request averages)
    itl_ms = [row["itl_ms"] for row in metric_rows if row.get("itl_ms") is not None]
    e2e_ms = [row["e2e_ms"] for row in metric_rows if row.get("e2e_ms") is not None]
    output_speeds = [
        _to_float(row["osl"]) / (_to_float(row["decode_ms"]) / 1000)
        for row in metric_rows
        if _to_float(row.get("osl")) > 0 and _to_float(row.get("decode_ms")) > 0
    ]
    return {
        "ttft_ms": _compute_distribution_stats(ttft_ms),
        "itl_ms": _compute_distribution_stats(itl_ms),
        "e2e_ms": _compute_distribution_stats(e2e_ms),
        "output_tokens_per_s": _compute_distribution_stats(output_speeds),
    }


def _compute_throughput_stats(
    metric_rows: list[dict],
    event_rows: list[dict],
    stats: ReplayStats,
    measurement_s: float,
) -> dict:
    # osl per row is already a per-request average; multiply by n to recover
    # the true token total for rows where n > 1 completions were blended.
    total_output_tokens = sum(
        _to_float(row.get("osl")) * _to_float(row.get("n") or 1) for row in metric_rows
    )
    system_output_tps = total_output_tokens / measurement_s if measurement_s else 0.0
    turns_per_s = stats.turns_completed / measurement_s if measurement_s else 0.0
    requests_per_s = len(event_rows) / measurement_s if measurement_s else 0.0
    return {
        "total_output_tokens": round(total_output_tokens),
        "system_output_tokens_per_s": round(system_output_tps, 3),
        "turns_per_s": round(turns_per_s, 3),
        "requests_per_s": round(requests_per_s, 3),
    }


def _compute_distribution_stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(mean(ordered), 3),
        "p25": round(_interpolate_percentile(ordered=ordered, p=25), 3),
        "p50": round(_interpolate_percentile(ordered=ordered, p=50), 3),
        "p90": round(_interpolate_percentile(ordered=ordered, p=90), 3),
        "p95": round(_interpolate_percentile(ordered=ordered, p=95), 3),
    }


def _flatten_for_csv(summary: dict) -> dict:
    return {
        "concurrency": summary["config"]["concurrency"],
        "duration_s": summary["config"]["duration_s"],
        "gpus": summary["config"]["gpus"],
        "gpu_name": summary["config"]["gpu_name"],
        "power_watts": summary["config"]["power_watts"],
        "completed_turns": summary["counts"]["completed_turns"],
        "failed_turns": summary["counts"]["failed_turns"],
        "system_output_tokens_per_s": summary["throughput"][
            "system_output_tokens_per_s"
        ],
        "turns_per_s": summary["throughput"]["turns_per_s"],
        "p50_ttft_ms": summary["interactivity"]["ttft_ms"].get("p50"),
        "p95_ttft_ms": summary["interactivity"]["ttft_ms"].get("p95"),
        "p50_itl_ms": summary["interactivity"]["itl_ms"].get("p50"),
        "p95_itl_ms": summary["interactivity"]["itl_ms"].get("p95"),
        "p25_output_tokens_per_s": summary["interactivity"]["output_tokens_per_s"].get(
            "p25"
        ),
        "agents_per_gpu": summary["normalization"]["agents_per_gpu"],
        "agents_per_mw": summary["normalization"]["agents_per_mw"],
        "output_tokens_per_s_per_mw": summary["normalization"][
            "output_tokens_per_s_per_mw"
        ],
    }


def _interpolate_percentile(ordered: list[float], p: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (1 - (rank - lo)) + ordered[hi] * (rank - lo)


def _read_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe_divide(numerator: float, denominator: float | int | None) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 3)


def _to_float(value) -> float:
    return float(value) if value is not None else 0.0
