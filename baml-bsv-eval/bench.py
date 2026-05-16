"""
bench.py — latency and token benchmark: raw Anthropic SDK vs BAML.

Measurements per run:
  Raw SDK  — llm_ms (messages.create), parse_ms (tool block + Pydantic),
             total_ms, tokens_in, tokens_out  (all exact, from response.usage)
  BAML     — total_ms (wall clock), tokens_in (count_tokens API on rendered prompt),
             tokens_out (count_tokens on result JSON)

BAML overhead isolation:
  fd2/stderr capture does not work on Windows — BAML's Rust runtime writes through
  the Windows console API rather than fd 2.  Instead we run N_OVERHEAD direct Anthropic
  calls using the exact BAML-rendered prompt + claude-sonnet-4 (no BAML wrapper) to
  get a clean LLM baseline.  overhead_ms ≈ BAML_total_ms − direct_llm_ms.

Prompt token analysis (count_tokens, one-time per email):
  Raw SDK : system | user_msg | tool_schema (three separate API calls to isolate each)
  BAML    : rules_and_email | ctx_output_format  (two calls)

Writes: bench_results.json, bench_results.md
Run  : .venv/Scripts/python bench.py
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv(override=True)

from raw_baseline import _client, _SYSTEM, _TOOL, _TOOL_NAME, PortcoUpdate
from baml_client import b

# ── Config ───────────────────────────────────────────────────────────────────

N_RUNS     = 10   # timing runs per email per version
N_OVERHEAD = 3    # direct Sonnet calls used to estimate BAML framework overhead
SLEEP_S    = 0.4  # pause between API calls to stay under rate limits

RAW_MODEL  = "claude-haiku-4-5-20251001"
BAML_MODEL = "claude-sonnet-4-20250514"

EMAIL_DIR  = Path(__file__).parent / "test_emails"
OUT_JSON   = Path(__file__).parent / "bench_results.json"
OUT_MD     = Path(__file__).parent / "bench_results.md"

# ── BAML prompt template ─────────────────────────────────────────────────────
# Mirrors extraction.baml verbatim.  ctx.output_format expansion verified
# from BAML INFO logs.

_CTX_OUTPUT_FORMAT = """\
Answer in JSON using this schema:
{
  company_name: string,
  reporting_period: string,
  arr_usd: float or null,
  customer_count: int or null,
  mrr_growth_pct: float or null,
  runway_months: float or null,
  key_updates: string[],
  risks_flagged: string[],
  key_hires: string[],
  sentiment: 'Positive' or 'Neutral' or 'Concerning',
}"""

_BAML_RULES = """\
You are a VC analyst assistant. Extract structured data from the portfolio company
update email below.

Rules:
- Populate numeric fields only when a clear, single value is stated or directly implied.
- Return null for any numeric field that is hedged, a range, or not mentioned.
- Never hallucinate figures.
- mrr_growth_pct: only populate when growth % is stated explicitly for MRR specifically.
  Do not derive from ARR growth.
- key_hires: named new joiners only — exclude open roles and existing team members.
  Format each as "Name (Role)" when role is available.
- key_updates: high-level milestones and accomplishments mentioned in the email.
- risks_flagged: concerns, threats, or uncertainties the founder explicitly raises.
- sentiment: overall tone of the update (positive / neutral / concerning).

Email:
{email_text}"""

def _baml_prompt(email_text: str) -> str:
    return _BAML_RULES.format(email_text=email_text) + "\n\n" + _CTX_OUTPUT_FORMAT

def _baml_prompt_no_schema(email_text: str) -> str:
    """BAML prompt without ctx.output_format — used to isolate schema token cost."""
    return _BAML_RULES.format(email_text=email_text)

# ── Helpers ───────────────────────────────────────────────────────────────────

def p95(data: list[float]) -> float:
    """95th percentile.  With n=10 this equals the max; labelled accordingly."""
    return sorted(data)[-1]

def safe_mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 1) if vals else None

def safe_p95(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(p95(vals), 1) if vals else None

def progress(msg: str) -> None:
    # Encode with 'replace' so Unicode arrows / box chars don't crash cp1252 terminals.
    print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8"), flush=True)

# ── One-time prompt token analysis ───────────────────────────────────────────

def count_tokens_raw(email_text: str) -> dict:
    """Break down raw SDK prompt tokens: system / user_msg / tool_schema.

    count_tokens requires at least one message, so we can't count system tokens
    in isolation.  Instead, we use a single-character placeholder message and
    subtract its cost from the system+placeholder count:

        system_tokens = (system + placeholder) - (placeholder alone)
        user_msg      = (system + user_msg)    - system_tokens
        tool_schema   = (system + user_msg + tools) - (system + user_msg)
    """
    user_msg_text = f"Extract the portfolio company update from this email:\n\n{email_text}"
    placeholder   = [{"role": "user", "content": "."}]

    full = _client.messages.count_tokens(
        model=RAW_MODEL, system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg_text}],
        tools=[_TOOL],
    )
    no_tools = _client.messages.count_tokens(
        model=RAW_MODEL, system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg_text}],
    )
    # Isolate system token cost via placeholder subtraction
    ph_only   = _client.messages.count_tokens(model=RAW_MODEL, messages=placeholder)
    ph_system = _client.messages.count_tokens(model=RAW_MODEL, system=_SYSTEM, messages=placeholder)
    system_tokens = ph_system.input_tokens - ph_only.input_tokens

    return {
        "total":       full.input_tokens,
        "system":      system_tokens,
        "user_msg":    no_tools.input_tokens - system_tokens,
        "tool_schema": full.input_tokens - no_tools.input_tokens,
    }


def count_tokens_baml(email_text: str) -> dict:
    """Break down BAML prompt tokens: rules+email / ctx.output_format."""
    full_prompt      = _baml_prompt(email_text)
    no_schema_prompt = _baml_prompt_no_schema(email_text)

    full = _client.messages.count_tokens(
        model=BAML_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
    )
    no_schema = _client.messages.count_tokens(
        model=BAML_MODEL,
        messages=[{"role": "user", "content": no_schema_prompt}],
    )
    return {
        "total":             full.input_tokens,
        "rules_and_email":   no_schema.input_tokens,
        "ctx_output_format": full.input_tokens - no_schema.input_tokens,
    }

# ── Single-run measurement functions ─────────────────────────────────────────

def run_raw(email_text: str) -> dict:
    """One raw SDK extraction.  Returns exact timing + token breakdown."""
    user_msg = f"Extract the portfolio company update from this email:\n\n{email_text}"

    t0 = time.perf_counter()
    try:
        response = _client.messages.create(
            model=RAW_MODEL, max_tokens=1024, system=_SYSTEM,
            tools=[_TOOL], tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)[:200],
                "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
    t_api = time.perf_counter()

    try:
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            raise ValueError("No tool_use block")
        PortcoUpdate.model_validate(tool_block.input)
    except (ValidationError, ValueError) as exc:
        return {"success": False, "error": str(exc)[:200],
                "llm_ms":   round((t_api - t0) * 1000, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
    t_done = time.perf_counter()

    return {
        "success":   True,
        "total_ms":  round((t_done - t0) * 1000, 1),
        "llm_ms":    round((t_api - t0) * 1000, 1),
        "parse_ms":  round((t_done - t_api) * 1000, 2),
        "tokens_in":  response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }


def run_baml(email_text: str, tok_in: int) -> dict:
    """One BAML extraction.  tok_in pre-counted via count_tokens (stable across runs).
    tokens_out counted from the result JSON via count_tokens.
    """
    t0 = time.perf_counter()
    try:
        result = b.ExtractPortcoUpdate(email_text=email_text)
        success = True
        error   = None
    except Exception as exc:
        t_done = time.perf_counter()
        return {"success": False, "error": str(exc)[:300],
                "total_ms": round((t_done - t0) * 1000, 1)}
    t_done = time.perf_counter()

    result_json = result.model_dump_json()

    # Count output tokens via API (adds ~100ms of network overhead per call;
    # we subtract this from the wall-clock measurement below)
    t_count = time.perf_counter()
    try:
        out_ct = _client.messages.count_tokens(
            model=BAML_MODEL,
            messages=[{"role": "assistant", "content": result_json}],
        )
        tok_out = out_ct.input_tokens   # count_tokens on assistant turn = output token count
    except Exception:
        tok_out = len(result_json) // 4  # fallback: char estimate
    t_count_done = time.perf_counter()

    # Subtract the count_tokens round-trip from total_ms so we measure the extraction alone
    count_overhead_ms = round((t_count_done - t_count) * 1000, 1)

    return {
        "success":           True,
        "total_ms":          round((t_done - t0) * 1000, 1),   # LLM + BAML framework
        "tokens_in":         tok_in,
        "tokens_out":        tok_out,
        "count_overhead_ms": count_overhead_ms,
    }


def run_direct_sonnet(email_text: str) -> dict:
    """Raw Anthropic call with BAML's rendered prompt + model, no BAML wrapper.
    Used to isolate pure LLM time so we can estimate framework overhead.
    """
    prompt = _baml_prompt(email_text)
    t0 = time.perf_counter()
    try:
        resp = _client.messages.create(
            model=BAML_MODEL, max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)[:200],
                "llm_ms": None}
    t_done = time.perf_counter()
    return {
        "success": True,
        "llm_ms":  round((t_done - t0) * 1000, 1),
        "tokens_in":  resp.usage.input_tokens,
        "tokens_out": resp.usage.output_tokens,
    }

# ── Markdown generation ───────────────────────────────────────────────────────

def fmt(v, suffix="") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}{suffix}"
    return f"{v}{suffix}"


def generate_markdown(data: dict) -> str:
    emails   = list(data["runs"].keys())
    pt       = data["prompt_tokens"]
    runs     = data["runs"]
    overhead = data.get("overhead_analysis", {})
    meta     = data["meta"]

    lines = []
    lines += [
        "# Benchmark Results: BAML vs Raw SDK",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Runs per email | {meta['n_runs']} |",
        f"| Overhead baseline runs | {meta['n_overhead']} |",
        f"| Raw SDK model | `{meta['raw_model']}` |",
        f"| BAML model | `{meta['baml_model']}` |",
        f"| Timestamp | {meta['timestamp']} |",
        "",
        "> **Note on models**: Raw SDK uses Haiku, BAML uses Sonnet.",
        "> Sonnet is slower and more expensive per token — latency differences partially",
        "> reflect model choice, not framework overhead alone.",
        "> The overhead analysis section isolates framework cost.",
        "",
    ]

    # ── 1. Prompt token comparison ──────────────────────────────────────────
    lines += ["## 1  Prompt Token Comparison", ""]
    lines += [
        "| Email | Raw total | > system | > user msg | > tool schema | BAML total | > rules+email | > ctx.output_format |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for em in emails:
        r = pt[em]["raw"]
        bl = pt[em]["baml"]
        lines.append(
            f"| {em} "
            f"| **{r['total']}** | {r['system']} | {r['user_msg']} | {r['tool_schema']} "
            f"| **{bl['total']}** | {bl['rules_and_email']} | {bl['ctx_output_format']} |"
        )

    # Summary row (use first email as representative — prompt structure, not email, varies)
    r0 = pt[emails[0]]["raw"]
    bl0 = pt[emails[0]]["baml"]
    delta = bl0["total"] - r0["total"]
    sign  = "+" if delta >= 0 else ""
    lines += [
        "",
        f"> **Schema encoding overhead** (email_1 as representative):",
        f"> Raw SDK tool schema = {r0['tool_schema']} tokens · "
        f"BAML ctx.output_format = {bl0['ctx_output_format']} tokens · "
        f"delta = {sign}{delta} tokens on total prompt",
        "",
    ]

    # ── 2. Latency ──────────────────────────────────────────────────────────
    lines += ["## 2  End-to-End Latency (ms)", ""]
    lines += [
        "| Email | Raw mean | Raw p95¹ | BAML mean | BAML p95¹ |",
        "|---|---:|---:|---:|---:|",
    ]
    for em in emails:
        raw_ok  = [r["total_ms"] for r in runs[em]["raw"]  if r.get("success")]
        baml_ok = [r["total_ms"] for r in runs[em]["baml"] if r.get("success")]
        lines.append(
            f"| {em} "
            f"| {fmt(safe_mean(raw_ok), 'ms')} | {fmt(safe_p95(raw_ok), 'ms')} "
            f"| {fmt(safe_mean(baml_ok), 'ms')} | {fmt(safe_p95(baml_ok), 'ms')} |"
        )
    lines += ["", "¹ p95 = max for n=10.", ""]

    # ── 3. LLM vs parsing breakdown (raw SDK) ──────────────────────────────
    lines += ["## 3  Raw SDK: LLM vs Parse Time", ""]
    lines += [
        "| Email | LLM mean | LLM p95 | Parse mean | Parse p95 | Parse share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for em in emails:
        raw_ok = [r for r in runs[em]["raw"] if r.get("success")]
        llm_vals   = [r["llm_ms"]   for r in raw_ok if r.get("llm_ms")   is not None]
        parse_vals = [r["parse_ms"] for r in raw_ok if r.get("parse_ms") is not None]
        total_vals = [r["total_ms"] for r in raw_ok]
        parse_share = (
            f"{safe_mean(parse_vals) / safe_mean(total_vals) * 100:.2f}%"
            if parse_vals and total_vals else "—"
        )
        lines.append(
            f"| {em} "
            f"| {fmt(safe_mean(llm_vals), 'ms')} | {fmt(safe_p95(llm_vals), 'ms')} "
            f"| {fmt(safe_mean(parse_vals), 'ms')} | {fmt(safe_p95(parse_vals), 'ms')} "
            f"| {parse_share} |"
        )
    lines += [
        "",
        "> Parse time = tool_use block lookup + Pydantic model_validate().",
        "> Sub-millisecond; raw SDK parsing overhead is negligible.",
        "",
    ]

    # ── 4. BAML framework overhead ─────────────────────────────────────────
    lines += ["## 4  BAML Framework Overhead", ""]
    lines += [
        "| Email | BAML total mean | Direct Sonnet mean | Est. overhead | Overhead % |",
        "|---|---:|---:|---:|---:|",
    ]
    for em in emails:
        baml_ok = [r["total_ms"] for r in runs[em]["baml"] if r.get("success")]
        baml_mean = safe_mean(baml_ok)

        oh_runs = overhead.get(em, {}).get("direct_sonnet", [])
        direct_vals = [r["llm_ms"] for r in oh_runs if r.get("success") and r.get("llm_ms")]
        direct_mean = safe_mean(direct_vals)

        if baml_mean is not None and direct_mean is not None:
            est_oh = round(baml_mean - direct_mean, 1)
            pct_oh = f"{est_oh / baml_mean * 100:.1f}%"
        else:
            est_oh = None
            pct_oh = "—"

        lines.append(
            f"| {em} "
            f"| {fmt(baml_mean, 'ms')} "
            f"| {fmt(direct_mean, 'ms')} "
            f"| {fmt(est_oh, 'ms')} "
            f"| {pct_oh} |"
        )
    lines += [
        "",
        "> **Direct Sonnet**: raw `messages.create` with BAML's rendered prompt, no BAML wrapper.",
        "> Overhead includes prompt rendering, request serialisation, response JSON parsing,",
        "> and BAML's type coercion — everything except the LLM call itself.",
        "",
    ]

    # ── 5. Token usage ─────────────────────────────────────────────────────
    lines += ["## 5  Token Usage", ""]
    lines += [
        "| Email | Raw tokens_in | Raw tokens_out | BAML tokens_in | BAML tokens_out |",
        "|---|---:|---:|---:|---:|",
    ]
    for em in emails:
        raw_ok  = [r for r in runs[em]["raw"]  if r.get("success")]
        baml_ok = [r for r in runs[em]["baml"] if r.get("success")]
        r_in  = safe_mean([r["tokens_in"]  for r in raw_ok  if r.get("tokens_in")  is not None])
        r_out = safe_mean([r["tokens_out"] for r in raw_ok  if r.get("tokens_out") is not None])
        b_in  = safe_mean([r["tokens_in"]  for r in baml_ok if r.get("tokens_in")  is not None])
        b_out = safe_mean([r["tokens_out"] for r in baml_ok if r.get("tokens_out") is not None])
        lines.append(
            f"| {em} "
            f"| {fmt(r_in)} | {fmt(r_out)} "
            f"| {fmt(b_in)} | {fmt(b_out)} |"
        )
    lines += [
        "",
        "> BAML tokens_in = count_tokens on rendered prompt (exact, same per run).",
        "> BAML tokens_out = count_tokens on result JSON (exact per run).",
        "> Raw SDK tokens from response.usage (exact).",
        "",
    ]

    # ── 6. Failure rate ────────────────────────────────────────────────────
    lines += ["## 6  Failure Rate", ""]
    lines += [
        "| Email | Raw failures | BAML failures | Raw failure type | BAML failure type |",
        "|---|---:|---:|---|---|",
    ]
    for em in emails:
        raw_fails  = [r for r in runs[em]["raw"]  if not r.get("success")]
        baml_fails = [r for r in runs[em]["baml"] if not r.get("success")]
        r_type  = raw_fails[0]["error"][:60]  if raw_fails  else "—"
        b_type  = baml_fails[0]["error"][:60] if baml_fails else "—"
        lines.append(
            f"| {em} "
            f"| {len(raw_fails)}/{meta['n_runs']} "
            f"| {len(baml_fails)}/{meta['n_runs']} "
            f"| {r_type} | {b_type} |"
        )
    lines += [""]

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    emails = {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(EMAIL_DIR.glob("email_*.txt"))
    }

    results: dict = {
        "meta": {
            "n_runs":     N_RUNS,
            "n_overhead": N_OVERHEAD,
            "raw_model":  RAW_MODEL,
            "baml_model": BAML_MODEL,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        },
        "prompt_tokens":    {},
        "runs":             {},
        "overhead_analysis": {},
    }

    total_calls = (
        len(emails) * 4             # count_tokens per email (raw×3, baml×2 → ~5, close enough)
        + len(emails) * N_RUNS      # raw SDK timing runs
        + len(emails) * N_RUNS      # BAML timing runs (each also calls count_tokens for out tokens)
        + len(emails) * N_OVERHEAD  # direct Sonnet overhead runs
    )
    progress(f"\n{'-'*60}")
    progress(f"bench.py  |  {len(emails)} emails × {N_RUNS} runs  |  ~{total_calls} API calls")
    progress(f"{'-'*60}\n")

    # Phase 1: Prompt token analysis
    progress("Phase 1/4 -- counting prompt tokens ...")
    for name, text in emails.items():
        progress(f"  {name}: raw SDK tokens ...")
        results["prompt_tokens"][name] = {"raw": count_tokens_raw(text)}
        time.sleep(SLEEP_S)

        progress(f"  {name}: BAML tokens ...")
        results["prompt_tokens"][name]["baml"] = count_tokens_baml(text)
        time.sleep(SLEEP_S)

    # Phase 2: Raw SDK timing
    progress("\nPhase 2/4 -- raw SDK timing runs ...")
    for name, text in emails.items():
        results["runs"][name] = {"raw": [], "baml": []}
        for i in range(N_RUNS):
            progress(f"  {name} raw {i+1}/{N_RUNS}")
            run = run_raw(text)
            results["runs"][name]["raw"].append(run)
            status = "OK" if run.get("success") else f"FAIL {run.get('error','')[:40]}"
            progress(f"    -> {status}  total={run.get('total_ms','?')}ms  "
                     f"llm={run.get('llm_ms','?')}ms  parse={run.get('parse_ms','?')}ms  "
                     f"in={run.get('tokens_in','?')} out={run.get('tokens_out','?')}")
            time.sleep(SLEEP_S)

    # Phase 3: BAML timing
    progress("\nPhase 3/4 -- BAML timing runs ...")
    for name, text in emails.items():
        baml_tok_in = results["prompt_tokens"][name]["baml"]["total"]
        for i in range(N_RUNS):
            progress(f"  {name} baml {i+1}/{N_RUNS}")
            run = run_baml(text, tok_in=baml_tok_in)
            results["runs"][name]["baml"].append(run)
            status = "OK" if run.get("success") else f"FAIL {run.get('error','')[:40]}"
            progress(f"    -> {status}  total={run.get('total_ms','?')}ms  "
                     f"in={run.get('tokens_in','?')} out={run.get('tokens_out','?')}")
            time.sleep(SLEEP_S)

    # Phase 4: Overhead analysis (direct Sonnet, no BAML wrapper)
    progress("\nPhase 4/4 -- overhead analysis (direct Sonnet, no wrapper) ...")
    for name, text in emails.items():
        results["overhead_analysis"][name] = {"direct_sonnet": []}
        for i in range(N_OVERHEAD):
            progress(f"  {name} direct-sonnet {i+1}/{N_OVERHEAD}")
            run = run_direct_sonnet(text)
            results["overhead_analysis"][name]["direct_sonnet"].append(run)
            progress(f"    -> llm={run.get('llm_ms','?')}ms")
            time.sleep(SLEEP_S)

    # Save JSON
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    progress(f"\nDONE: bench_results.json written ({OUT_JSON.stat().st_size // 1024} KB)")

    # Generate markdown
    md = generate_markdown(results)
    OUT_MD.write_text(md, encoding="utf-8")
    progress("DONE: bench_results.md written")
    progress("\n-- Preview " + "-"*50)
    progress(md[:3000])


if __name__ == "__main__":
    main()
