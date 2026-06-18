"""Per-problem execution environment.

``Sandbox`` is the throwaway workspace for one dataset problem: a fresh temp
dir under ``root`` where the repo is checked out and claude-cli runs, removed
on exit. It also records wall-clock ``.start_ts`` / ``.end_ts`` around the
``with`` body.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path


class Sandbox:
    """Throwaway workspace + wall-clock timer for one problem.

    Creates a fresh temp dir under `root` on enter (``.dir``) and removes it
    on exit; records ``.start_ts`` / ``.end_ts`` around the ``with`` body."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = root
        self.prefix = prefix
        self.dir: Path | None = None
        self.start_ts = 0.0
        self.end_ts = 0.0

    def __enter__(self) -> Sandbox:
        self.root.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.root))
        self.start_ts = time.time()
        return self

    def __exit__(self, *exc) -> bool:
        self.end_ts = time.time()
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
        return False
