from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from loguru import logger

from coding_agents.trace_collection.pipeline.harness.env import create_claude_env
from coding_agents.trace_collection.pipeline.harness.transcript import (
    copy_transcript,
    get_session_id_from_logs,
)
from coding_agents.trace_collection.pipeline.utils import git_repo as git
from coding_agents.trace_collection.pipeline.utils.processes import (
    terminate_process_tree,
)

DEFAULT_TIMEOUT_S = 7200
TIMEOUT_EXIT_CODE = 124
_TERM_GRACE_S = 10

PROMPT = """You are working on a real software-engineering bug from SWE-bench. Solve it by editing files in this repository.

Repository: {repo}
Base commit: {base_commit}

# Problem statement
{problem_statement}

# Instructions
- The repo is checked out in your working directory at the base commit.
- Read relevant files, understand the bug, then make code edits to fix it.
- You may run shell commands to explore and validate.
- When you believe the fix is complete, summarize what you changed and stop.
"""


def run_task(
    task: dict,
    sandbox_dir: Path,
    model: str,
    base_url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    oauth: bool = False,
    runs_dir: Path | None = None,
) -> int:
    try:
        repo = git.clone(task, sandbox_dir)
        return run_session(
            task,
            repo,
            model=model,
            url=base_url,
            timeout_s=timeout_s,
            oauth=oauth,
            runs_dir=runs_dir,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.error("{}: clone/solve failed: {!r}", task["instance_id"], exc)
        return 1


def get_claude_version() -> str:
    try:
        return subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def run_session(
    problem: dict,
    repo_dir: Path,
    model: str,
    url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    oauth: bool = False,
    runs_dir: Path | None = None,
) -> int:
    prompt = PROMPT.format(
        repo=problem["repo"],
        base_commit=problem["base_commit"],
        problem_statement=problem["problem_statement"],
    )
    capture = runs_dir is not None
    proc = subprocess.Popen(
        [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model",
            model,
        ],
        cwd=str(repo_dir),
        env=create_claude_env(model=model, url=url, oauth=oauth),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    reader = None
    session_sink: dict = {}
    if capture:
        reader = threading.Thread(
            target=get_session_id_from_logs,
            args=(proc.stdout, session_sink),
            daemon=True,
        )
        reader.start()
    try:
        proc.wait(timeout=timeout_s)
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning(
            "{}: claude session exceeded {}s — killing process tree",
            problem["instance_id"],
            timeout_s,
        )
        terminate_process_tree(proc.pid, _TERM_GRACE_S)
        return TIMEOUT_EXIT_CODE
    except BaseException:
        terminate_process_tree(proc.pid, _TERM_GRACE_S)
        raise
    finally:
        if reader is not None:
            reader.join(timeout=10)
        if capture:
            copy_transcript(
                session_sink.get("id"),
                Path(repo_dir),
                Path(runs_dir),
                problem["instance_id"],
            )
