#!/usr/bin/env bash
# Installs the coding-agents harness and deps on a virual env

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

log() { printf '\n\033[1;36m[install]\033[0m %s\n' "$*"; }

# --- 1. uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found — installing"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv $(uv --version)"

# --- 2. Claude Code CLI ------------------------------------------------------
# Native installer (no Node/npm) — drops the binary in ~/.local/bin. Falls back
# to npm only if curl is somehow unavailable.
if ! command -v claude >/dev/null 2>&1; then
    if command -v curl >/dev/null 2>&1; then
        log "claude CLI not found — installing via the native installer"
        curl -fsSL https://claude.ai/install.sh | bash
        export PATH="$HOME/.local/bin:$PATH"
    elif command -v npm >/dev/null 2>&1; then
        log "claude CLI not found, curl unavailable — installing via npm"
        npm install -g @anthropic-ai/claude-code
    else
        echo "ERROR: 'claude' CLI missing and neither curl nor npm is available." >&2
        echo "Install one, then: curl -fsSL https://claude.ai/install.sh | bash" >&2
        exit 1
    fi
fi
log "claude $(claude --version 2>&1 | head -1)"

# --- 3. venv + vLLM + harness ------------------------------------------------
if [[ -d "$HERE/.venv" ]]; then
    log "reusing existing venv at .venv (re-syncing packages)"
else
    log "creating Python 3.12 venv at .venv"
    uv venv --python 3.12
fi
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"

# `pip install .` pulls everything from pyproject.toml — including vLLM, which
# must land in this venv: server.sh launches `<venv>/bin/vllm`, and if it's
# absent vLLM falls back to a system `vllm` on PATH (wrong env / wrong deps),
# which fails cryptically on startup. (Stock vLLM ships the Anthropic
# /v1/messages endpoint Claude Code talks to — no fork needed.)
#
# `uv pip` targets the active venv. A bare `pip` wouldn't: `uv venv` ships no
# pip, so `pip install .` would silently fall back to a system/user pip and
# install into the wrong environment.
log "installing the harness + deps (pip install .)"
uv pip install .

log "done — .venv is ready. Activate it with:"
log "  source .venv/bin/activate"
