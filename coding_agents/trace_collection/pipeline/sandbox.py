"""Per-problem execution workspace."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

_SANDBOX_ROOT = Path("/tmp/swe_sandboxes")


class Sandbox:
    """Throwaway workspace + wall-clock timer for one problem."""

    def __init__(self, save_dir: Path, prefix: str) -> None:
        self.root = _SANDBOX_ROOT / save_dir.name
        self.prefix = prefix
        self.dir: Path | None = None
        self.start_ts = 0.0
        self.end_ts = 0.0

    def __enter__(self) -> "Sandbox":
        self.root.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.root))
        self.start_ts = time.time()
        return self

    def __exit__(self, *exc) -> bool:
        self.end_ts = time.time()
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
        return False
