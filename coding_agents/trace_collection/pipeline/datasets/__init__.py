from __future__ import annotations

import json
from pathlib import Path

from coding_agents.trace_collection.pipeline.datasets import (
    swe_bench_pro,
    swe_bench_verified,
)

TASK_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")

DATASETS = {
    "pro": (swe_bench_pro.DATASET_ID, swe_bench_pro.load),
    "verified": (swe_bench_verified.DATASET_ID, swe_bench_verified.load),
}


def get_dataset(
    name: str,
    resume_path: Path | None = None,
    num_samples: int | None = None,
) -> tuple[list[dict], str, int]:
    dataset_id, load_fn = DATASETS[name]
    tasks = [_normalize_task(row) for row in load_fn()]
    solved_ids = _load_solved_ids(resume_path) if resume_path is not None else set()
    skipped_count = sum(task["instance_id"] in solved_ids for task in tasks)
    pending = [task for task in tasks if task["instance_id"] not in solved_ids]
    if num_samples is not None:
        pending = pending[:num_samples]
    return pending, dataset_id, skipped_count


def _load_solved_ids(resume_path: Path) -> set[str]:
    if not resume_path.exists():
        return set()

    solved: set[str] = set()
    for path in resume_path.glob("*/session.json"):
        try:
            session = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if session.get("exit_code") == 0 and session.get("instance_id"):
            solved.add(session["instance_id"])
    return solved


def _normalize_task(row: dict) -> dict:
    task = dict(row)
    for field in TASK_FIELDS:
        task[field] = str(task[field])
    return task


__all__ = ["get_dataset"]
