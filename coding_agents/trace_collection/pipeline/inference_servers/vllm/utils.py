"""Stateless utils for the vLLM server: HTTP probes, GPU/NVML queries,
version/log/.env readers. No dependency on VllmServer."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pynvml

# coding_agents/ — inference_servers/vllm/ is four levels up from this file.
SERVER_SH = Path(__file__).resolve().parents[4] / "server.sh"
ENV_PATH = SERVER_SH.parent / ".env"
LOG = Path("/tmp/vllm_server.log")


from coding_agents.trace_collection.pipeline.utils.http import (
    check_server_initialized,
)  # noqa: F401


def get_server_metadata(url: str) -> dict:
    """Fetch served-model metadata from /v1/models."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=2.0) as response:
        data = json.loads(response.read())["data"][0]
    return {"id": data["id"], "max_model_len": data.get("max_model_len")}


@contextmanager
def nvml_context():
    """NVML init/shutdown guard; yields the pynvml module."""
    pynvml.nvmlInit()
    try:
        yield pynvml
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def gpu_used_mib() -> int:
    """GPU 0 memory in use (MiB), via NVML; 0 if unavailable."""
    try:
        with nvml_context() as nv:
            handle = nv.nvmlDeviceGetHandleByIndex(0)
            return nv.nvmlDeviceGetMemoryInfo(handle).used // (1024 * 1024)
    except pynvml.NVMLError:
        return 0


def gpu_info() -> dict:
    """GPU name / count / total-memory (MiB) of device 0; {} if unavailable."""
    try:
        with nvml_context() as nv:
            count = nv.nvmlDeviceGetCount()
            if not count:
                return {}
            name = nv.nvmlDeviceGetName(nv.nvmlDeviceGetHandleByIndex(0))
            if isinstance(name, bytes):
                name = name.decode()
            total = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(0)).total
            return {"name": name, "count": count, "memory_mib": total // (1024 * 1024)}
    except pynvml.NVMLError:
        return {}


def get_package_version(package: str) -> str:
    """Installed distribution version of `package`; '' if not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return ""


def read_tail(line_count: int = 40) -> str:
    """Last `line_count` lines of the vLLM server log."""
    try:
        return "\n".join(LOG.read_text().splitlines()[-line_count:])
    except OSError:
        return ""


def read_env_file() -> dict[str, str]:
    """Parse server.sh's .env (KEY=VALUE, ignoring comments)."""
    env_values: dict[str, str] = {}
    try:
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_values[key.strip()] = value.split("#", 1)[0].strip()
    except OSError:
        pass
    return env_values
