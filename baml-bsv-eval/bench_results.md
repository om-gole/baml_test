# Benchmark Results: BAML vs Raw SDK

## Configuration

| Parameter | Value |
|---|---|
| Runs per email | 10 |
| Overhead baseline runs | 3 |
| Raw SDK model | `claude-haiku-4-5-20251001` |
| BAML model | `claude-sonnet-4-20250514` |
| Timestamp | 2026-05-16T16:50:23.573970+00:00 |

> **Note on models**: Raw SDK uses Haiku, BAML uses Sonnet.
> Sonnet is slower and more expensive per token — latency differences partially
> reflect model choice, not framework overhead alone.
> The overhead analysis section isolates framework cost.

## 1  Prompt Token Comparison

| Email | Raw total | > system | > user msg | > tool schema | BAML total | > rules+email | > ctx.output_format |
|---|---:|---:|---:|---:|---:|---:|---:|
| email_1 | **1659** | 90 | 600 | 969 | **912** | 789 | 123 |
| email_2 | **1599** | 90 | 540 | 969 | **852** | 729 | 123 |
| email_3 | **1653** | 90 | 594 | 969 | **906** | 783 | 123 |
| email_bad | **1084** | 90 | 25 | 969 | **337** | 214 | 123 |

> **Schema encoding overhead** (email_1 as representative):
> Raw SDK tool schema = 969 tokens · BAML ctx.output_format = 123 tokens · delta = -747 tokens on total prompt

## 2  End-to-End Latency (ms)

| Email | Raw mean | Raw p95¹ | BAML mean | BAML p95¹ |
|---|---:|---:|---:|---:|
| email_1 | 2845.3ms | 3293.5ms | 3823.0ms | 4111.2ms |
| email_2 | 3107.6ms | 3417.2ms | 4570.0ms | 4939.5ms |
| email_3 | 3303.1ms | 3864.5ms | 4189.4ms | 5273.7ms |
| email_bad | 1474.4ms | 5266.2ms | — | — |

¹ p95 = max for n=10.

## 3  Raw SDK: LLM vs Parse Time

| Email | LLM mean | LLM p95 | Parse mean | Parse p95 | Parse share |
|---|---:|---:|---:|---:|---:|
| email_1 | 2845.3ms | 3293.5ms | 0.0ms | 0.1ms | 0.00% |
| email_2 | 3107.6ms | 3417.2ms | 0.0ms | 0.1ms | 0.00% |
| email_3 | 3303.0ms | 3864.5ms | 0.0ms | 0.1ms | 0.00% |
| email_bad | 1474.4ms | 5266.1ms | 0.0ms | 0.1ms | 0.00% |

> Parse time = tool_use block lookup + Pydantic model_validate().
> Sub-millisecond; raw SDK parsing overhead is negligible.

## 4  BAML Framework Overhead

| Email | BAML total mean | Direct Sonnet mean | Est. overhead | Overhead % |
|---|---:|---:|---:|---:|
| email_1 | 3823.0ms | 3579.0ms | 244.0ms | 6.4% |
| email_2 | 4570.0ms | 4500.4ms | 69.6ms | 1.5% |
| email_3 | 4189.4ms | 4432.7ms | -243.3ms | -5.8% |
| email_bad | — | 1847.2ms | — | — |

> **Direct Sonnet**: raw `messages.create` with BAML's rendered prompt, no BAML wrapper.
> Overhead includes prompt rendering, request serialisation, response JSON parsing,
> and BAML's type coercion — everything except the LLM call itself.

## 5  Token Usage

| Email | Raw tokens_in | Raw tokens_out | BAML tokens_in | BAML tokens_out |
|---|---:|---:|---:|---:|
| email_1 | 1759 | 424.4 | 912 | 234.4 |
| email_2 | 1699 | 365.4 | 852 | 214 |
| email_3 | 1753 | 392.5 | 906 | 208.2 |
| email_bad | 1184 | 114 | — | — |

> BAML tokens_in = count_tokens on rendered prompt (exact, same per run).
> BAML tokens_out = count_tokens on result JSON (exact per run).
> Raw SDK tokens from response.usage (exact).

## 6  Failure Rate

| Email | Raw failures | BAML failures | Raw failure type | BAML failure type |
|---|---:|---:|---|---|
| email_1 | 0/10 | 0/10 | — | — |
| email_2 | 0/10 | 0/10 | — | — |
| email_3 | 0/10 | 0/10 | — | — |
| email_bad | 0/10 | 10/10 | — | BamlValidationError(message=Failed to parse LLM response: Fa |
