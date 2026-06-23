from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from coding_agents.serving import MetricsScraper, VllmConfig, VllmServer
from coding_agents.trace_collection.pipeline.proxy import Proxy


class VllmBackend:
    name = "vllm"

    def __init__(
        self,
        server_url: str,
        proxy_port: int,
        model=None,
        tensor_parallel_size=None,
        max_model_len=None,
        gpu_memory_utilization=None,
        tool_call_parser=None,
    ) -> None:
        if tool_call_parser is None:
            raise ValueError("--tool-call is required with --backend vllm")
        self.proxy_port = proxy_port
        self._config = VllmConfig.resolve(
            url=server_url,
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tool_call_parser=tool_call_parser,
        )

    def config(self) -> dict:
        proxy_url = f"http://127.0.0.1:{self.proxy_port}"
        return {
            "backend_name": self.name,
            "model": self._config.model,
            "base_url": proxy_url,
            "server_url": self._config.url,
            "server_port": self._config.port,
            "proxy_url": proxy_url,
            "proxy_port": self.proxy_port,
            "serving_config": self._config.serving_config(),
        }

    def session(
        self,
        save_dir: Path,
        instance_id: str,
        capture: bool = False,
        fn: Callable[[dict], None] | None = None,
    ) -> "VllmSession":
        return VllmSession(
            config=self._config,
            proxy_port=self.proxy_port,
            save_dir=save_dir,
            instance_id=instance_id,
            capture=capture,
            fn=fn,
        )


class VllmSession:
    def __init__(
        self,
        config: VllmConfig,
        proxy_port: int,
        save_dir: Path,
        instance_id: str,
        capture: bool,
        fn: Callable[[dict], None] | None = None,
    ) -> None:
        self._config = config
        self._proxy_port = proxy_port
        self._save_dir = save_dir
        self._instance_id = instance_id
        self._capture = capture
        self._fn = fn
        self._stack: contextlib.ExitStack | None = None
        self._server: VllmServer | None = None
        self.model: str = ""
        self.base_url: str = ""
        self.oauth: bool = False
        self.transcript_runs_dir: Path | None = None

    def __enter__(self) -> "VllmSession":
        self._stack = contextlib.ExitStack()
        server = self._stack.enter_context(VllmServer(self._config))
        self._stack.enter_context(
            MetricsScraper(
                url=server.url,
                save_dir=self._save_dir,
                instance_id=self._instance_id,
                enabled=True,
                fn=self._fn,
            )
        )
        proxy = self._stack.enter_context(
            Proxy(
                save_dir=self._save_dir,
                instance_id=self._instance_id,
                url=server.url,
                proxy_port=self._proxy_port,
                enabled=True,
                capture=self._capture,
            )
        )
        self._server = server
        self.model = server.model
        self.base_url = proxy.base_url
        return self

    def __exit__(self, *exc) -> bool:
        if self._stack is None:
            return False
        return self._stack.__exit__(*exc)

    def runtime(self) -> dict:
        server = self._server
        proxy_url = f"http://127.0.0.1:{self._proxy_port}"
        if server is None:
            return {
                "backend_name": "vllm",
                "model": self._config.model,
                "base_url": proxy_url,
                "server_url": self._config.url,
                "server_port": self._config.port,
                "proxy_url": proxy_url,
                "proxy_port": self._proxy_port,
                "serving_config": self._config.serving_config(),
            }
        return {
            "backend_name": "vllm",
            "model": self.model,
            "base_url": self.base_url,
            "server_url": server.url,
            "server_port": server.port,
            "proxy_url": self.base_url,
            "proxy_port": self._proxy_port,
            "serving_config": server.serving_config(),
            "transcript_runs_dir": None,
        }
