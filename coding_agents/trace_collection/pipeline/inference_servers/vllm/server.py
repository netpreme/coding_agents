"""vLLM inference server — process lifecycle only."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import psutil
from loguru import logger

from coding_agents.trace_collection.pipeline.inference_servers.vllm.utils import (
    LOG,
    SERVER_SH,
    check_server_initialized,
    get_server_metadata,
    gpu_used_mib,
    read_env_file,
    read_tail,
)
from coding_agents.trace_collection.pipeline.utils.processes import (
    process_family,
    processes_with_env,
    terminate_processes,
    unique_processes,
)


class VllmServer:
    """vLLM process lifecycle context manager.

    Enter to start the server, exit to stop it.
    Accepts a pre-resolved VllmConfig so config and lifecycle are separate.
    """

    _ENV_PORT = "CODING_AGENTS_VLLM_PORT"
    _CLEAN_SHM_ENV = "CODING_AGENTS_CLEAN_ORPHANED_SHM"
    _GPU_RELEASE_TIMEOUT = 60.0
    _READY_TIMEOUT = 600.0
    _TERM_GRACE_S = 5.0

    def __init__(self, config: "VllmConfig") -> None:
        self.config = config
        self.url = config.url
        self.port = config.port
        self.model = config.model
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "VllmServer":
        logger.info("starting vllm at {}", self.url)
        self._stop(require_port_free=True)
        with LOG.open("w") as log_fh:
            self._proc = subprocess.Popen(
                ["bash", str(SERVER_SH)],
                env=self._env(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + self._READY_TIMEOUT
        while not check_server_initialized(f"{self.url}/v1/models", 2.0):
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm died on startup; tail of {LOG}:\n{read_tail()}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"vllm not ready after {self._READY_TIMEOUT:.0f}s (see {LOG})"
                )
        info = get_server_metadata(self.url)
        self.model = info["id"]
        if info["max_model_len"] is not None:
            self.config = VllmConfig(
                model=self.model,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_model_len=info["max_model_len"],
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                tool_call_parser=self.config.tool_call_parser,
                port=self.config.port,
                url=self.config.url,
            )
        logger.info(
            "vllm ready at {} — {}",
            self.url,
            " ".join(f"{k}={v}" for k, v in self.serving_config().items()),
        )
        return self

    def __exit__(self, *exc) -> bool:
        logger.info("stopping vllm at {}", self.url)
        self._stop(require_port_free=False)
        return False

    def serving_config(self) -> dict:
        sc = self.config.serving_config()
        sc["model"] = self.model  # may be updated from server after start
        return sc

    # -- Private ---------------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config.env_vars())
        env[self._ENV_PORT] = str(self.port)
        env["VLLM_PYTHON"] = sys.executable
        return env

    def _stop(self, require_port_free: bool) -> None:
        killed = self._kill_vllm()
        port_is_free = self._wait_port_free()
        self._wait_gpu_free()
        if killed:
            self._clean_shm()
        if require_port_free and not port_is_free:
            raise RuntimeError(
                f"port {self.port} is busy, but no process owned by this harness was found to stop"
            )

    def _kill_vllm(self) -> int:
        processes = self._server_processes()
        if not processes:
            return 0
        killed_pids = terminate_processes(processes, self._TERM_GRACE_S)
        logger.info("killed vllm pids {}", killed_pids)
        return len(killed_pids)

    def _server_processes(self) -> list[psutil.Process]:
        roots: list[psutil.Process] = []
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                roots.append(psutil.Process(self._proc.pid))
        roots.extend(processes_with_env(self._ENV_PORT, str(self.port)))
        return unique_processes(
            process for root in roots for process in process_family(root)
        )

    def _clean_shm(self) -> None:
        if os.environ.get(self._CLEAN_SHM_ENV) != "1":
            logger.info(
                "skipping broad /dev/shm cleanup; set {}=1 to remove orphaned psm_* segments after stopping vllm",
                self._CLEAN_SHM_ENV,
            )
            return
        removed = 0
        for seg in Path("/dev/shm").glob("psm_*"):
            try:
                seg.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info("cleaned {} orphaned /dev/shm vllm segment(s)", removed)

    def _wait_port_free(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("localhost", self.port)) != 0:
                    return True
            time.sleep(0.5)
        return False

    def _wait_gpu_free(self) -> None:
        deadline = time.monotonic() + self._GPU_RELEASE_TIMEOUT
        while gpu_used_mib() >= 1000 and time.monotonic() < deadline:
            time.sleep(2)


# ── Supporting ─────────────────────────────────────────────────────────────

_SERVER_DEFAULTS = {
    "MODEL_NAME": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "TENSOR_PARALLEL_SIZE": "1",
    "MAX_MODEL_LEN": "131072",
    "GPU_MEMORY_UTILIZATION": "0.9",
    "TOOL_CALL_PARSER": "qwen3_coder",
}


@dataclass(frozen=True)
class VllmConfig:
    """Resolved launch settings for server.sh."""

    model: str
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    tool_call_parser: str
    port: int
    url: str

    @classmethod
    def resolve(
        cls,
        url,
        model=None,
        tensor_parallel_size=None,
        max_model_len=None,
        gpu_memory_utilization=None,
        tool_call_parser=None,
    ) -> "VllmConfig":
        values = dict(_SERVER_DEFAULTS)
        values.update(read_env_file())
        for key in _SERVER_DEFAULTS:
            if key in os.environ:
                values[key] = os.environ[key]
        overrides = {
            "MODEL_NAME": model,
            "TENSOR_PARALLEL_SIZE": tensor_parallel_size,
            "MAX_MODEL_LEN": max_model_len,
            "GPU_MEMORY_UTILIZATION": gpu_memory_utilization,
            "TOOL_CALL_PARSER": tool_call_parser,
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = str(value)
        port = urlparse(url).port or 8000
        return cls(
            model=values["MODEL_NAME"],
            tensor_parallel_size=int(values["TENSOR_PARALLEL_SIZE"]),
            max_model_len=int(values["MAX_MODEL_LEN"]),
            gpu_memory_utilization=float(values["GPU_MEMORY_UTILIZATION"]),
            tool_call_parser=values["TOOL_CALL_PARSER"],
            port=port,
            url=url,
        )

    def serving_config(self) -> dict:
        return {
            "backend": "vllm",
            "model": self.model,
            "tp": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_util": self.gpu_memory_utilization,
            "tool_call_parser": self.tool_call_parser,
            "port": str(self.port),
            "url": self.url,
        }

    def env_vars(self) -> dict[str, str]:
        return {
            "MODEL_NAME": self.model,
            "TENSOR_PARALLEL_SIZE": str(self.tensor_parallel_size),
            "MAX_MODEL_LEN": str(self.max_model_len),
            "GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "TOOL_CALL_PARSER": self.tool_call_parser,
            "PORT": str(self.port),
        }
