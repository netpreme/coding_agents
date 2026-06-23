"""Build synthetic /v1/messages request bodies from captured token counts.

Each turn's request is constructed from:
  - A fixed shared system prompt (same text across ALL sessions and turns,
    so vLLM's prefix cache hits it after the first session warms it).
  - Deterministic per-session/per-turn fake messages whose lengths
    match the captured isl_new and osl values.

The conversation history grows consistently within a session:
  turn N request = system + [user_0, asst_0, ..., user_{N-1}, asst_{N-1}, user_N]
  where user_N has isl_new tokens (the new prefill for this turn)
  and asst_i has osl_i tokens (the prior turn's output length).

Prefix caching: vLLM caches blocks of token IDs. Because the prior-turn
content is reproduced identically each time (same seed → same text →
same characters → same token IDs), vLLM reuses KV blocks for the shared
prefix and only re-prefills the new suffix.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4

# Repetitions chosen to produce a system prompt long enough (~5 k tokens)
# that vLLM allocates several full KV-cache blocks for it, making cross-session
# prefix-cache hits meaningful.
_SYSTEM_PROMPT_REPETITIONS = 300
_SYSTEM_PROMPT = (
    "You are an AI assistant that helps software engineers solve coding tasks. "
    * _SYSTEM_PROMPT_REPETITIONS
)
_SYSTEM_TOKENS = len(_SYSTEM_PROMPT) // _CHARS_PER_TOKEN


def build_turn_request(session: dict, turn_idx: int, model: str) -> dict:
    """Return a /v1/messages request body for the given turn.

    Message history is rebuilt from scratch each call — O(turn_idx) messages —
    because each turn must include all prior assistant responses to correctly
    reproduce the prefix that vLLM will cache.
    """
    turn = session["turns"][turn_idx]
    messages = []

    for prior_idx in range(turn_idx):
        prior_turn = session["turns"][prior_idx]
        messages.append(
            {
                "role": "user",
                "content": _build_user_message(session=session, turn_idx=prior_idx),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": _generate_text_of_token_length(
                    label=f"{session['instance_id']}:a{prior_idx}",
                    n_tokens=prior_turn["osl"],
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": _build_user_message(session=session, turn_idx=turn_idx),
        }
    )

    return {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "messages": messages,
        "max_tokens": turn["osl"],
        "min_tokens": turn["osl"],
    }


def _build_user_message(session: dict, turn_idx: int) -> str:
    if turn_idx == 0:
        # Turn 0 has no cached prefix; the user message fills up to the
        # captured isl minus the system prompt.
        n_tokens = max(1, session["turns"][0]["isl"] - _SYSTEM_TOKENS)
    else:
        # Turn N: only the isl_new suffix is new prefill; the rest is cached.
        n_tokens = max(1, session["turns"][turn_idx]["isl_new"])
    return _generate_text_of_token_length(
        label=f"{session['instance_id']}:u{turn_idx}", n_tokens=n_tokens
    )


def _generate_text_of_token_length(label: str, n_tokens: int) -> str:
    """Deterministic text of approximately n_tokens tokens."""
    target_chars = max(1, n_tokens * _CHARS_PER_TOKEN)
    chunk = f"[{label}] the quick brown fox jumps over the lazy dog "
    repeats = (target_chars // len(chunk)) + 1
    return (chunk * repeats)[:target_chars]
