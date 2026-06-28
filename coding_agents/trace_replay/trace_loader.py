"""Load captured trace sessions from a trace_collection results directory."""

from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger

from coding_agents.serving import VllmConfig


def build_server_config_from_capture(capture_config: dict) -> VllmConfig:
    """Build a VllmConfig from a prior capture run's run_config.json."""
    serving = capture_config.get("serving_config") or {}
    args = capture_config.get("args") or {}
    network = capture_config.get("network") or {}
    url = serving.get("url") or network.get("server_url") or "http://localhost:8000"
    return VllmConfig.resolve(
        url=url,
        model=serving.get("model") or capture_config.get("model"),
        tensor_parallel_size=serving.get("tp") or args.get("tensor_parallel_size"),
        max_model_len=serving.get("max_model_len") or args.get("max_model_len"),
        gpu_memory_utilization=serving.get("gpu_util")
        or args.get("gpu_memory_utilization"),
        tool_call_parser=serving.get("tool_call_parser")
        or args.get("tool_call_parser"),
    )


def load_and_shuffle_sessions(results_dir: Path, seed: int) -> list[dict]:
    """Load all sessions from results_dir and shuffle them with a fixed seed."""
    sessions = sorted(
        load_sessions(results_dir), key=lambda session: session["instance_id"]
    )
    random.Random(seed).shuffle(sessions)
    return sessions


def load_sessions(results_dir: Path) -> list[dict]:
    """Return all sessions found under ``results_dir/runs/``.

    Each session directory must contain both ``engine_metrics.jsonl``
    (token counts and latency from the vLLM Prometheus scraper) and
    ``turn_traces.jsonl`` (tool execution timing from the capture proxy).
    Directories missing either file are silently skipped.
    """
    results_dir = Path(results_dir)
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"no runs/ directory found in {results_dir}")

    sessions: list[dict] = []
    for instance_dir in sorted(runs_dir.iterdir()):
        session = _load_session_from_dir(instance_dir)
        if session is not None:
            sessions.append(session)

    if not sessions:
        raise ValueError(
            f"no valid sessions (engine_metrics.jsonl + turn_traces.jsonl) "
            f"found in {runs_dir}"
        )
    return sessions


def _load_session_from_dir(instance_dir: Path) -> dict | None:
    """Load one session directory, returning None if files are missing."""
    if not instance_dir.is_dir():
        return None

    engine_file = instance_dir / "engine_metrics.jsonl"
    traces_file = instance_dir / "turn_traces.jsonl"
    if not engine_file.exists() or not traces_file.exists():
        return None

    engine_rows = _read_jsonl_rows(engine_file)
    trace_rows = _read_jsonl_rows(traces_file)

    if len(engine_rows) != len(trace_rows):
        logger.warning(
            "instance {}: {} engine rows vs {} trace rows — using the shorter",
            instance_dir.name,
            len(engine_rows),
            len(trace_rows),
        )

    turns = [
        _parse_turn_from_metrics(engine_row=engine_row, trace_row=trace_row)
        for engine_row, trace_row in zip(engine_rows, trace_rows)
    ]

    if not turns:
        return None
    return {"instance_id": instance_dir.name, "turns": turns}


def _parse_turn_from_metrics(engine_row: dict, trace_row: dict) -> dict:
    """Construct a turn dict from one engine_metrics row and one turn_traces row.

    engine_metrics values may be fractional after per-request normalisation
    (when n>1 completions land in one Prometheus tick), so explicit rounding
    is applied before converting to int.
    """
    return {
        "isl": int(round(engine_row["isl"])),
        "isl_new": int(round(engine_row["isl_new"])),
        "osl": int(round(engine_row["osl"])),
        "tool_exec_ms": trace_row["tool_exec_ms"],
    }


def _read_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
