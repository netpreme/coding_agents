# Coding Agent Trace Replay

To simulate coding agents in production, trace_replay replays captured sessions from `trace_collection` against an inference engine — recording TTFT, ITL, prefill time, decode time, KV cache occupancy, prefix cache hit rate, and output token throughput. For each turn, synthetic prompts matching the captured ISL, OSL, and ISL_new are sent to the engine, with inter-turn sleeps reproducing the harness tool execution time. This preserves the token distribution and cache reuse patterns of real coding agent sessions without running live agents.

N concurrent virtual agents draw from the pool of captured sessions and replay them until a time limit T is reached. When a session ends, the agent picks the next one from the pool, cycling as needed.

For each agent, per-turn metrics are written to `turn_events.jsonl` (TTFT, duration, status) and `engine_metrics.jsonl` (prefill time, decode time, ITL, queue time, KV cache usage, prefix cache hit rate). Once the run completes, aggregate metrics are computed across all agents: throughput (tokens/s, turns/s), latency distributions (P50/P90/P95 TTFT and ITL), interactivity, and efficiency ratios normalized per GPU and per megawatt.

---

## Captured metrics

**Interactivity** — per-request, client-side (no blending across concurrent requests):

| metric | unit | description |
|---|---|---|
| TTFT (P50 / P95) | ms | time from request sent to first output token |
| Output token speed (P25) | tok/s | decode throughput per request |
| ITL (P50 / P95) | ms/tok | inter-token latency during decode |
| E2E latency (P50 / P95) | ms | full round-trip from request to final token |

**Server-side breakdown** — averaged per completion from the Prometheus scraper:

| metric | unit | description |
|---|---|---|
| Prefill time | ms | time to process the input and fill the KV cache |
| Decode time | ms | time to generate all output tokens |
| Queue wait | ms | scheduler queue time before prefill begins |
| Prefix cache hit rate | % | fraction of input tokens served from KV cache (`(ISL − ISL_new) / ISL`) |
| KV cache occupancy | % | peak fraction of GPU KV cache in use across the tick |

**Throughput** — system-wide across all concurrent agents:

| metric | unit | description |
|---|---|---|
| Output tokens/s | tok/s | total output token rate across all agents |
| Turns/s | turns/s | completed inference requests per second |

**Efficiency** — normalized for hardware comparison:

| metric | unit | description |
|---|---|---|
| Agents/GPU | agents | concurrent agents per physical GPU |
| Tokens/s/GPU | tok/s | output throughput per GPU |
| Agents/MW | agents | concurrent agents per megawatt of accelerator power |
| Tokens/s/MW | tok/s | output throughput per megawatt |

---

## How it works

Traces captured by `trace_collection` record the ISL, OSL, ISL_new, and inter-turn tool execution time for every turn of every session. The replay engine reconstructs realistic request traffic by dispatching synthetic prompts at the original sequence lengths, sleeping between turns to reproduce the captured tool execution delay, and keeping a fixed number of virtual agents active in parallel for the full run duration. Prefix cache reuse is enabled — the shared system prompt and growing conversation history accumulate KV blocks that the engine reuses across turns, matching how real agents behave.

Metrics are collected from two sources:

**Client-side** — each of the N concurrent async workers independently tracks wall-clock time around its own request. No polling, no blending across requests.

**Server-side** — `MetricsScraper` polls the inference engine's Prometheus endpoint every 100 ms and diffs consecutive snapshots to capture KV cache occupancy, queue time, and prefill/decode breakdown. When multiple requests complete in one tick, per-request latency fields are divided by the completion count to produce per-request averages rather than sums.

**Power** — `GpuPowerMonitor` samples total GPU power draw via pynvml every second for the duration of the run and averages the readings. Used to compute efficiency metrics (tokens/s/MW, agents/MW).

---

**Client-side metrics** — written to `turn_events.jsonl`, one row per completed turn:

| field | unit | meaning |
|---|---|---|
| `worker_id` | int | which concurrent worker handled this turn |
| `cycle` | int | how many times this worker has wrapped around the session list |
| `instance_id` | str | SWE-bench instance identifier |
| `turn` | int | turn index within the session |
| `ts` | unix s | wall-clock timestamp when the request was sent |
| `isl` | tokens | input sequence length (from captured trace) |
| `osl` | tokens | output sequence length (from captured trace) |
| `isl_new` | tokens | uncached input tokens (from captured trace) |
| `tool_exec_ms` | ms | inter-turn sleep simulating tool execution |
| `started_at` | unix s | when the HTTP request was sent |
| `first_byte_at` | unix s | when the first response byte arrived |
| `ended_at` | unix s | when the full response was received |
| `ttft_ms` | ms | time-to-first-token: `first_byte_at − started_at` (per-request, no blending) |
| `duration_ms` | ms | total request round-trip: `ended_at − started_at` |
| `status_code` | int | HTTP status code |
| `error` | str | error repr if the request failed; null otherwise |

**Server-side metrics** — written to `engine_metrics.jsonl`, one row per detected completion from Prometheus:

| field | unit | meaning |
|---|---|---|
| `n` | int | number of requests that completed in this tick (>1 means values are averages) |
| `isl` | tokens | average input sequence length |
| `osl` | tokens | average output sequence length |
| `isl_new` | tokens | average uncached input tokens (went through prefill compute) |
| `prefill_ms` | ms | average prefill time |
| `decode_ms` | ms | average decode time |
| `queue_ms` | ms | average scheduler queue wait before prefill |
| `itl_ms` | ms/tok | average inter-token latency during decode |
| `e2e_ms` | ms | average end-to-end latency from the inference engine |
| `kv_cache_usage_pct_peak` | % | peak KV-cache block occupancy across the tick (shown as `kv_util` in live display) |
| `prefix_cache_hits` | tokens | prefix cache hits this tick |
| `stop_reason` | enum | `stop` / `length` / `abort` / `error` / `repetition` |

---

## How to run

```bash
python -m coding_agents.trace_replay.main \
    --trace_path coding_agents/trace_collection/results/<stamp> \
    --concurrency 8 \
    --duration 10
```

Flags:

| flag | default | meaning |
|---|---|---|
| `--trace_path PATH` | required | `results/<stamp>/` directory from a prior capture run; must contain `run_config.json` |
| `--concurrency N` | `8` | number of concurrent virtual agents to keep active |
| `--duration M` | `10` | replay duration in minutes |

The inference engine is configured automatically from the capture run's `run_config.json` (model, tensor-parallel size, max model length, tool-call parser).

---

## System design

```
   ┌──────────────────────────┐
   │  Captured trace sessions │   ISL / OSL / ISL_new / tool_exec_ms per turn
   └────────────┬─────────────┘
                │ N concurrent workers
                ▼
   ┌──────────────────────────┐
   │     Replay engine        │   async workers, each replaying one session at a time
   │                          │   sleeps tool_exec_ms between turns
   └────────────┬─────────────┘
                │ POST /v1/messages (streaming)
                ▼
   ┌──────────────────────────┐
   │   Inference engine       │ ────────► /metrics  (Prometheus)
   └──────────────────────────┘               ▲
                │ SSE stream                  │ scraped every 100 ms
                │ first byte → ttft_ms        │ KV cache, queue, prefill/decode times
                ▼                             │
   ┌──────────────────────────┐    ┌──────────┴───────────┐
   │   turn_events.jsonl      │    │  engine_metrics.jsonl │
   │   (client-side, per-req) │    │  (server-side, Prom.) │
   └──────────────────────────┘    └──────────────────────┘
```

---

## Per-run output (`results/replay_<stamp>/`)

- `run_config.json` — replay config (trace path, capture model, concurrency, duration, GPU name/count, measured power)
- `runs/<phase>/turn_events.jsonl` — client-side per-request metrics (ttft_ms, duration_ms, status); one row per turn, no blending across concurrent requests
- `runs/<phase>/engine_metrics.jsonl` — server-side Prometheus metrics per completion
- `runs/<phase>/summary.json` — aggregated throughput, latency percentiles, and efficiency normalizations
- `summary.csv` — flat CSV version of summary for easy comparison across runs
