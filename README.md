# Coding Agents Experiments/Applications

Coding agents take turns to carry out tasks. Using Claude Code as the harness, the repo contains code to:
* collect per-turn token counts, inference metrics, inter-turn CPU execution time (harness wall-clock time between inference calls), and tool call distributions from live coding agent sessions
* replay captured coding agent sessions from the above using an inference engine to benchmark throughput, latency, efficiency and power consumption 

## Modules

### [`trace_collection`](coding_agents/trace_collection/README.md)

Runs coding agents against a SWE Bench dataset and captures per-turn token counts (ISL, OSL, ISL_new), inter-turn tool execution time, and server-side inference metrics via a transparent proxy. Produces a results directory of captured sessions that can be used directly for analysis or replayed.

Engine-side metrics are collected by polling vLLM's Prometheus `/metrics` endpoint on a background thread (every 100 ms). Each completed request produces one row in `engine_metrics.jsonl`, computed as the delta between consecutive snapshots (per-request averages when multiple requests complete in the same tick).

| Field | Description | Method |
| --- | --- | --- |
| `ts` | Unix timestamp of the metrics window start | Stamped before the `/metrics` scrape |
| `n` | Number of requests that completed in this polling tick (fields below are per-request averages) | Delta of `vllm:request_prompt_tokens_count` |
| `isl` | Input sequence length — total prompt tokens for the request (includes the full conversation history, so it grows each turn) | Delta of `vllm:request_prompt_tokens_sum` |
| `osl` | Output sequence length — generated tokens | Delta of `vllm:request_generation_tokens_sum` |
| `isl_new` | Prompt tokens actually computed in prefill (prefix-cache misses) — typically the previous turn's output plus tool results | Delta of `vllm:request_prefill_kv_computed_tokens_sum` |
| `prefill_ms` | Time spent in prefill | Delta of `vllm:request_prefill_time_seconds_sum` |
| `decode_ms` | Time spent in decode | Delta of `vllm:request_decode_time_seconds_sum` |
| `queue_ms` | Time spent queued before scheduling | Delta of `vllm:request_queue_time_seconds_sum` |
| `itl_ms` | Inter-token latency (time per output token) | Delta of `vllm:request_time_per_output_token_seconds_sum` |
| `e2e_ms` | End-to-end request latency | Delta of `vllm:e2e_request_latency_seconds_sum` |
| `stop_reason` | Why generation finished (`stop`, `length`, `abort`, `error`, `repetition`) | Delta of `vllm:request_success_total` per finish-reason label |
| `kv_cache_usage_pct_peak` | Peak KV cache utilisation (%) observed during the request | Max of `vllm:kv_cache_usage_perc` across polls since the last completion |
| `prefix_cache_hits` | Prompt tokens served from the local prefix cache | Delta of `vllm:prefix_cache_hits_total` |
| `external_prefix_cache_hits` | Prompt tokens served from an external prefix cache (e.g. KV offload) | Delta of `vllm:external_prefix_cache_hits_total` |

When `--capture` mode is selected, the raw prompts are recorded by a transparent proxy between Claude Code and vLLM. One row is written to `turn_traces.jsonl` per API request.

| Field | Description | Method |
| --- | --- | --- |
| `request_time` | Unix timestamp of the request arriving at the proxy | Stamped when the proxy receives the request |
| `tool_exec_ms` | Approximated CPU/tool execution time — harness wall-clock time between the previous inference response and the current inference request | Time between the previous response completing and this request arriving at the proxy |
| `isl_text` | Full prompt text of the request | Request body, split into message chunks and joined |
| `isl_new_text` | New portion of the prompt — the suffix not shared with the previous turn's prompt | Common-prefix comparison against the previous turn's chunks |
| `osl_text` | Model's response text | Captured from the streamed response |

### [`trace_replay`](coding_agents/trace_replay/README.md)

Replays captured sessions against a fresh inference engine to measure production throughput and latency under realistic concurrent load — without running live agents. The primary metrics are agents per megawatt and agents per GPU, with supporting breakdowns of TTFT, ITL, prefill/decode time, prefix cache hit rate, and output token throughput.

## Installation

Install prerequisites and set up the Python virtual environment.

```bash
./install.sh
```

## Collected traces

[`trace_collection`](coding_agents/trace_collection/README.md) was used to record the ISL, OSL, ISL_new of Claude Opus and gpt-oss-120b solving SWE Bench Pro / Verified using Claude Code. The traces are uploaded to [Huggingface](https://huggingface.co/datasets/netpreme/coding_agent_traces) with analysis on the token count and the tool call distributions.
