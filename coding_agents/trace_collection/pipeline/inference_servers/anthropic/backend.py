from __future__ import annotations

from pathlib import Path


class AnthropicBackend:
    name = "anthropic"
    oauth = True

    def __init__(self, anthropic_url: str, model: str | None = None) -> None:
        if model is None:
            raise ValueError("--model is required with --backend anthropic")
        self.model = model
        self.api_url = anthropic_url.rstrip("/")

    def config(self) -> dict:
        return {
            "backend_name": self.name,
            "model": self.model,
            "base_url": self.api_url,
            "server_url": self.api_url,
            "server_port": None,
            "proxy_url": None,
            "proxy_port": None,
            "oauth": True,
            "serving_config": {
                "backend": self.name,
                "model": self.model,
                "url": self.api_url,
            },
        }

    def session(
        self, save_dir: Path, instance_id: str, **_: object
    ) -> "AnthropicSession":
        return AnthropicSession(backend=self, save_dir=save_dir)


class AnthropicSession:
    def __init__(self, backend: AnthropicBackend, save_dir: Path) -> None:
        self._backend = backend
        self._save_dir = save_dir
        self.model = backend.model
        self.base_url = backend.api_url
        self.oauth = True
        self.transcript_runs_dir = save_dir / "runs"

    def __enter__(self) -> "AnthropicSession":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def runtime(self) -> dict:
        return {
            "backend_name": "anthropic",
            "model": self.model,
            "base_url": self.base_url,
            "server_url": self.base_url,
            "server_port": None,
            "proxy_url": None,
            "proxy_port": None,
            "serving_config": {
                "backend": "anthropic",
                "model": self.model,
                "url": self.base_url,
            },
            "oauth": True,
            "transcript_runs_dir": self._save_dir / "runs",
        }
