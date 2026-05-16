"""
server.py — FastAPI backend for the BAML vs Raw SDK comparison demo.

Endpoints:
  GET  /emails          — returns test email file contents
  POST /extract/raw     — raw Anthropic SDK extraction, streamed as SSE
  POST /extract/baml    — BAML extraction, streamed as SSE

SSE event shapes:
  {"type": "prompt",  "content": {"system": str, "user": str, "tool": dict}}
  {"type": "prompt",  "content": str}          # BAML: unified rendered string
  {"type": "result",  "content": dict}
  {"type": "meta",    "latency_ms": int, "tokens_in": int, "tokens_out": int, "estimated": bool}
  {"type": "error",   "content": str}

Run with:
  uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# dotenv must run before any module that reads ANTHROPIC_API_KEY at import time
load_dotenv(override=True)

from pydantic import ValidationError                         # noqa: E402
from raw_baseline import _SYSTEM, _TOOL, _TOOL_NAME, _client, PortcoUpdate   # noqa: E402
from baml_client import b                                    # noqa: E402

app = FastAPI(title="BAML vs Raw SDK Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# BAML prompt template — mirrors extraction.baml exactly so the demo shows the
# real rendered prompt (with ctx.output_format expanded) rather than the source.
# ---------------------------------------------------------------------------

# This is what {{ ctx.output_format }} expands to for our PortcoUpdate schema.
# Verified from BAML INFO logs during actual runs.
_BAML_OUTPUT_FORMAT = """\
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

# Matches the prompt #"..."# block in extraction.baml verbatim.
_BAML_TEMPLATE = """\
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
{email_text}

{output_format}"""


def _render_baml_prompt(email_text: str) -> str:
    return _BAML_TEMPLATE.format(email_text=email_text, output_format=_BAML_OUTPUT_FORMAT)


# ---------------------------------------------------------------------------
# Blocking extraction functions (run in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _run_raw(email_text: str) -> tuple:
    """Returns (result_dict, tokens_in, tokens_out, latency_ms)."""
    t0 = time.perf_counter()
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": f"Extract the portfolio company update from this email:\n\n{email_text}"}],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    tool_block = next((blk for blk in response.content if blk.type == "tool_use"), None)
    if tool_block is None:
        raise ValueError(f"No tool_use block returned (stop_reason={response.stop_reason})")

    result = PortcoUpdate.model_validate(tool_block.input)
    return (
        json.loads(result.model_dump_json()),
        response.usage.input_tokens,
        response.usage.output_tokens,
        latency_ms,
    )


def _run_baml(email_text: str) -> tuple:
    """Returns (result_dict, latency_ms, tokens_in_est, tokens_out_est).
    Token counts are character-based estimates (labeled in the UI); BAML's Rust
    runtime writes token info to stderr directly, bypassing Python capture.
    """
    t0 = time.perf_counter()
    result = b.ExtractPortcoUpdate(email_text=email_text)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    result_dict = json.loads(result.model_dump_json())

    prompt = _render_baml_prompt(email_text)
    result_text = json.dumps(result_dict)
    # ~4 chars per token is a standard rough estimate for English + JSON
    tokens_in_est = len(prompt) // 4
    tokens_out_est = len(result_text) // 4

    return result_dict, latency_ms, tokens_in_est, tokens_out_est


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


_SSE_DONE = "data: [DONE]\n\n"

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/emails")
async def list_emails():
    """Return all test email texts keyed by stem (e.g. 'email_1')."""
    email_dir = Path(__file__).parent / "test_emails"
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(email_dir.glob("email_*.txt"))
    }


@app.post("/extract/raw")
async def extract_raw(request: Request):
    body = await request.json()
    email_text = (body.get("email_text") or "").strip()

    async def stream():
        # Emit prompt immediately — no API call needed to know what we'll send
        prompt_payload = {
            "system": _SYSTEM,
            "user": f"Extract the portfolio company update from this email:\n\n{email_text}",
            "tool": _TOOL,
        }
        yield _sse({"type": "prompt", "content": prompt_payload})

        try:
            result_dict, tok_in, tok_out, latency_ms = await asyncio.to_thread(_run_raw, email_text)
            yield _sse({"type": "result", "content": result_dict})
            yield _sse({"type": "meta", "latency_ms": latency_ms,
                        "tokens_in": tok_in, "tokens_out": tok_out, "estimated": False})
        except ValidationError as exc:
            yield _sse({"type": "error", "content": f"Pydantic validation error:\n{exc}"})
        except Exception as exc:
            yield _sse({"type": "error", "content": str(exc)})

        yield _SSE_DONE

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/extract/baml")
async def extract_baml(request: Request):
    body = await request.json()
    email_text = (body.get("email_text") or "").strip()

    async def stream():
        # Render the full BAML prompt (template + ctx.output_format expansion) upfront
        yield _sse({"type": "prompt", "content": _render_baml_prompt(email_text)})

        try:
            result_dict, latency_ms, tok_in, tok_out = await asyncio.to_thread(_run_baml, email_text)
            yield _sse({"type": "result", "content": result_dict})
            yield _sse({"type": "meta", "latency_ms": latency_ms,
                        "tokens_in": tok_in, "tokens_out": tok_out, "estimated": True})
        except Exception as exc:
            # Includes baml_py.BamlValidationError on schema mismatches (e.g. email_bad.txt)
            yield _sse({"type": "error", "content": str(exc)})

        yield _SSE_DONE

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
