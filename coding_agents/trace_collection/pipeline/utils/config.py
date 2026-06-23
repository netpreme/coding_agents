"""Run output helpers."""

from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path

from loguru import logger

_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def instance_dir(runs_dir: Path, instance_id: str) -> Path:
    return runs_dir / _SAFE_FILE_RE.sub("_", instance_id)[:200]


class JsonlWriter:
    def __init__(self, runs_dir: Path, filename: str) -> None:
        self._dir = runs_dir
        self._filename = filename

    def write(self, instance_id: str, row: dict) -> None:
        if not instance_id:
            return
        problem_dir = instance_dir(self._dir, instance_id)
        problem_dir.mkdir(parents=True, exist_ok=True)
        with (problem_dir / self._filename).open("a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def save_config(
    save_dir: Path,
    runtime: dict,
    args: dict,
    dataset_name: str,
    pending_count: int,
    skipped_solved_count: int,
    run_start_ts: float,
) -> None:
    from coding_agents.serving.utils import (
        get_package_version,
        gpu_info,
        read_env_file,
    )
    from coding_agents.trace_collection.pipeline.harness import get_claude_version

    config_path = save_dir / "run_config.json"
    if config_path.exists():
        return
    config = {
        "stamp": save_dir.name,
        "run_start_ts": round(run_start_ts, 3),
        "command": " ".join(sys.argv),
        "backend": runtime["backend_name"],
        "model": runtime["model"],
        "args": args,
        "serving_config": dict(runtime["serving_config"]),
        "dotenv": _redact_env(read_env_file()),
        "dataset": {
            "name": dataset_name,
            "pending": pending_count,
            "skipped_solved": skipped_solved_count,
        },
        "network": {
            "server_url": runtime["server_url"],
            "server_port": runtime["server_port"],
            "proxy_url": runtime["proxy_url"],
            "proxy_port": runtime["proxy_port"],
            "agent_base_url": runtime["base_url"],
        },
        "versions": {
            "claude": get_claude_version(),
            "vllm": get_package_version("vllm"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "gpu": gpu_info(),
    }
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    logger.info("wrote config to {}", config_path)


def save_session_config(
    save_dir: Path,
    task: dict,
    runtime: dict,
    start_ts: float,
    end_ts: float,
    exit_code: int,
) -> None:
    iid = task["instance_id"]
    problem_dir = instance_dir(save_dir / "runs", iid)
    problem_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "instance_id": iid,
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "backend": runtime["backend_name"],
        "model": runtime["model"],
        "serving_config": dict(runtime["serving_config"]),
        "agent_base_url": runtime["base_url"],
        "start_ts": round(start_ts, 3),
        "end_ts": round(end_ts, 3),
        "exit_code": exit_code,
    }
    (problem_dir / "session.json").write_text(json.dumps(session, indent=2) + "\n")


def _redact_env(values: dict[str, str]) -> dict[str, str]:
    sensitive_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    return {
        key: "<redacted>" if any(m in key.upper() for m in sensitive_markers) else value
        for key, value in values.items()
    }
