"""Run the per-problem reverse proxy."""

from __future__ import annotations

import threading
from pathlib import Path

import uvicorn
from loguru import logger
from coding_agents.trace_collection.pipeline.proxy.app import ProxyApp
from coding_agents.trace_collection.pipeline.utils.config import instance_dir
from coding_agents.trace_collection.pipeline.utils.http import check_server_initialized

# Loopback host the proxy binds to and claude-cli connects back through.
PROXY_HOST = "127.0.0.1"
# Seconds to wait for the proxy thread to come up.
PROXY_READY_TIMEOUT_S = 10.0


class Proxy:
    def __init__(
        self,
        save_dir: Path,
        instance_id: str,
        *,
        url: str,
        proxy_port: int = 8001,
        enabled: bool = True,
        capture: bool = False,
        upstream_health: bool = True,
    ) -> None:
        self.save_dir = save_dir
        self.instance_id = instance_id
        self.url = url
        self.proxy_port = proxy_port
        self.enabled = enabled
        self.capture = capture
        self.upstream_health = upstream_health
        self.out_dir = save_dir / "runs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = url
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Proxy:
        if not self.enabled:
            return self  # no proxy; claude talks to vLLM directly

        if self.capture:
            idir = instance_dir(self.out_dir, self.instance_id)
            (idir / "turn_traces.jsonl").unlink(missing_ok=True)
        app = ProxyApp(
            self.url,
            self.out_dir,
            self.instance_id,
            capture=self.capture,
        ).build()

        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=PROXY_HOST,
                port=self.proxy_port,
                log_level="warning",
                access_log=False,
            )
        )
        self._thread = threading.Thread(
            target=self._server.run, name="proxy", daemon=True
        )
        self._thread.start()

        proxy_url = f"http://{PROXY_HOST}:{self.proxy_port}"
        if self.upstream_health and not check_server_initialized(
            f"{proxy_url}/v1/models", PROXY_READY_TIMEOUT_S
        ):
            self.__exit__(None, None, None)
            raise RuntimeError(f"proxy did not start on {proxy_url}")
        self.base_url = proxy_url
        logger.info("proxy serving at {} → {}", proxy_url, self.url)
        return self

    def __exit__(self, *exc) -> bool:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        self._server = None
        return False
