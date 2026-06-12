# Coding Agent Traces Collection 

Coding agents take turns to carry out tasks. To understand the token distribution and inference metrics, a set up using Claude code with Opus and open source/weight models are constructed to carry out agentic turn inference. 

For Opus, only the locally saved files from the harness were used for analysis.

Traces data for Opus (counts only) and gpt-oss-120b (counts and raw texts) available on [huggingface](https://huggingface.co/netpreme).

---

Coding agents take multiple turns to carry out a task from the input prompt. To analyze the token distribution two models were selected: Anthropic’s Opus and OpenAI’s gpt-oss-120B. The input sequence length (ISL), output sequence length (OSL) and the uncached, new input sequence length (ISL_new) were extracted from locally saved files or a proxy used as a middleman. The setup consists of using Claude-code as the harness, SWE-Bench Pro as the dataset. For open source models, vLLM is used as the inference server and also uses SWE-Bench Verified dataset. 

Each task is solved sequentially to capture the token distribution. 

Using vLLM, for each turn, the uncached tokens (prefill) and the newly generated tokens (decode) will have their KV cache computed and will be stored in blocks. In the subsequent turn, the matching KV blocks will be used. Non-matching tokens will go through prefill (ISL_new tokens), and decode will generate one token at a time (OSL tokens), repeating the cycle.

The OSL is the cumulative tokens generated in decode. ISL_new is the unique tokens without prefix cache hit (tool call result + partial OSL). ISL is the total input token (previous ISL + partial OSL + tool call results). Prefix cache hit is computed as (ISL - ISL_new) / ISL. Token counts are obtained from vLLM’s prometheus loggers, measured in per turn sensitivity. Opus has these metrics that are accessible in the local computer inside `~/.claude/projects/<sanitized-cwd>/<session_id>.jsonl`.

**Inference metrics — from vLLM's `/metrics` Prometheus endpoint**:

| field | unit | meaning |
|---|---|---|
| `isl` | tokens | input sequence length: prompt tokens fed to the model this turn |
| `osl` | tokens | output sequence length: generated tokens this turn |
| `isl_new` | tokens | uncached input tokens that actually went through prefill compute |
| `isl_cached` | tokens | input tokens reused from the prefix cache (`isl − isl_new`) |
| `cache_hit_rate` | 0-1 | `isl_cached / isl` |
| `ttft_ms` | ms | time-to-first-token (reconstructed as `queue + prefill`) |
| `prefill_ms` | ms | scheduler time spent prefilling this request |
| `decode_ms` | ms | scheduler time spent decoding this request |
| `itl_ms` | ms/tok | mean inter-token latency during decode |
| `queue_ms` | ms | scheduler queue wait before prefill  |
| `kv_cache_usage_pct` | % | peak GPU KV-cache utilization across the turn's in-flight polls |
| `stop_reason` | enum | `stop` / `length` / `abort` / `error` / `repetition` |

**Harness metrics** — derived in the analysis layer from the raw text trace (`vllm_traces.jsonl`):

| field | meaning |
|---|---|
| `agent` | `main` (claude's outer loop, ~27 k char system prompt) or `sub` (Task-tool sub-agent, ~3 k char system prompt) |
| `num_tool_defs` | number of tool schemas claude shipped in the request |
| `num_messages` | length of the `messages` array |
| `system_prompt_chars` | raw character count of the system prompt  |

**Orchestration metadata**:

| field | meaning |
|---|---|
| `instance_id` | SWE-bench Pro/Verified instance_id |
| `turn` | turn number within the problem |
| `ts` | wall-clock timestamp of the turn (orders turns) |
| `e2e_ms` | vLLM's end-to-end latency for the turn |
| `prefix_kv_tokens` | `isl + osl` of the previous turn (max possible cache reuse this turn) |
| `usable_prefix_kv_tokens` | `cache_hit_rate × prefix_kv_tokens` |
| `kv_cache_used_bytes` | `isl_cached × Bytes/tok` |


## How to run

```bash
# GPT-OSS 120B on local vLLM (2 GPUs), capturing raw per-turn text traces
python coding_agents/isl_osl/main.py --model openai/gpt-oss-120b --tool-call openai --tensor-parallel 2 --capture raw

# Claude Opus via Anthropic (OAuth subscription; no local vLLM)
python coding_agents/isl_osl/main.py --model opus --backend anthropic --capture raw
```

`main.py` starts a fresh vLLM server per problem. To swap the served model, pass the flags to `main.py`

Flags:

| flag | default | meaning |
|---|---|---|
| `--backend NAME`               | `vllm` | `vllm` serves locally and scrapes Prometheus; `anthropic` uses Claude OAuth and saves transcripts for later analysis |
| `--dataset NAME`               | `pro` | benchmark dataset: `pro` ([SWE-bench Pro public set](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)) or `verified` ([SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)) |
| `--capture [raw]`              | —     | run the proxy and tee per-turn raw text traces to `vllm_traces.jsonl` (vLLM backend only) |
| `--limit N`                    | all pending problems | run at most `N` pending problems |
| `--resume SAVE_DIR`            | —     | reuse an existing run directory and skip problems with `exit_code == 0` |
| `--model HF_ID`                | `server.sh` / `.env` | model id; required for `--backend anthropic` |
| `--tool-call NAME`             | — (required for `--backend vllm`) | vLLM's tool-call parser. Must match the model family (`qwen3_coder` for Qwen, `openai` for GPT-OSS, `hermes` / `mistral` / `llama3_json` for others) |
| `--tensor-parallel N`          | `1`   | vLLM `--tensor-parallel-size` (bump for multi-GPU) |
| `--max-model-len N`            | `131072` | vLLM `--max-model-len`; cap is the model's `max_position_embeddings` |
| `--gpu-memory-utilization F`   | `0.85` | vLLM `--gpu-memory-utilization` (0-1) |

For Anthropic/OAuth runs, `main.py` copies `claude_transcript.jsonl` files; a
separate analysis pass derives `vllm_metrics.jsonl` and `vllm_traces.jsonl` from
those transcripts.


## System design

For open source/weight models

```
   ┌──────────────────────┐
   │   SWE-bench problem  │   GitHub repo cloned locally @ base_commit
   └──────────┬───────────┘
              │ problem statement
              ▼
   ┌──────────────────────┐
   │     Claude Code      │   reads files, edits, runs tests, loops tool calls 
   └──────────┬───────────┘
              │ POSTs /v1/messages, one per turn
              ▼
   ┌──────────────────────┐
   │        proxy         │   Middleman to capture request IO contents
   └──────────┬───────────┘
              │ forwards unmodified
              ▼
   ┌──────────────────────┐
   │        vLLM          │ ────────► /metrics  (Prometheus)
   └──────────────────────┘               ▲
              │ response streamed         │ scraped every 100 ms,
              ▼                           │ capture per-turn metrics
   ┌──────────────────────┐         ┌─────┴────────────────────┐
   │      claude-cli      │         │     metrics watcher      │
   │  (next turn, repeat) │         └──────────────────────────┘
   └──────────────────────┘        
                                    
```
For Opus, a request is sent to Anthropic's backend - no vLLM or proxy.


## Note
After submitting a task to Claude code, it may send title-generation requests to vLLM before sending task related requests. These are independent to the task at hand, and have ~50 shared tokens with the system prompt, altering the actual workload prefix cache metrics. These are filtered using a proxy. 

## Per-run output (`results/<stamp>/`)

- `config.json` — overall run config (CLI args, serving config, resolved model, dataset name + counts, versions, GPU info)
- `telemetry/<id>/vllm_metrics.jsonl` — one row per assistant turn (vLLM metrics, or derived Anthropic usage)
- `telemetry/<id>/vllm_traces.jsonl` — per-turn raw text trace (isl/isl_new/osl as text); only with `--capture`
- `telemetry/<id>/claude_transcript.jsonl` — Anthropic/OAuth transcript source, when using that backend
- `telemetry/<id>/session_config.json` — per-problem config (instance_id, repo/commit, server, model, started_at, ended_at, exit_code, ...)



