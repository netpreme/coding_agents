from __future__ import annotations

import os


def create_claude_env(model: str, url: str, *, oauth: bool = False) -> dict[str, str]:
    """Env for claude-cli."""
    env = os.environ.copy()
    if oauth:
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
    else:
        env["ANTHROPIC_BASE_URL"] = url
        env.setdefault("ANTHROPIC_API_KEY", "vllm-local")
        env["ANTHROPIC_MODEL"] = model
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["IS_SANDBOX"] = "1"
    env.setdefault("BASH_DEFAULT_TIMEOUT_MS", "300000")
    env.setdefault("BASH_MAX_TIMEOUT_MS", "600000")
    return env
