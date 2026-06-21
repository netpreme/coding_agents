from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from loguru import logger

from coding_agents.trace_collection.pipeline.utils.config import instance_dir


def get_session_id_from_logs(stdout, sink: dict) -> None:
    for line in stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("session_id"):
            sink["id"] = event["session_id"]


def locate_transcript(session_id: str | None, repo_dir: Path) -> Path | None:
    projects_dir = Path.home() / ".claude" / "projects"
    if session_id:
        transcript_paths = list(projects_dir.glob(f"*/{session_id}.jsonl"))
        if transcript_paths:
            return transcript_paths[0]
    encoded_repo_dir = re.sub(r"[^A-Za-z0-9]", "-", str(repo_dir.resolve()))
    transcript_paths = sorted(
        (projects_dir / encoded_repo_dir).glob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return transcript_paths[-1] if transcript_paths else None


def copy_transcript(
    session_id: str | None, repo_dir: Path, runs_dir: Path, instance_id: str
) -> None:
    tpath = locate_transcript(session_id, repo_dir)
    if tpath is None:
        logger.warning(
            "{}: no claude transcript found — telemetry skipped", instance_id
        )
        return
    idir = instance_dir(runs_dir, instance_id)
    idir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(tpath, idir / "claude_transcript.jsonl")
    except OSError as exc:
        logger.warning("{}: transcript copy failed: {!r}", instance_id, exc)
