"""
raw_baseline.py — portfolio update extractor using the raw Anthropic SDK + tool use.

Friction log (collected inline below):
  1. Tool schema must be built by hand (or massaged from Pydantic's output).
  2. Optional / nullable fields require explicit anyOf:[type, null] in JSON Schema.
  3. Response parsing: no .tool_calls shortcut — must walk response.content list.
  4. tool_block.input is already a dict, not a raw JSON string (easy to double-parse).
  5. Pydantic validation is entirely manual — no framework safety net.
  6. mrr_growth_pct is rarely stated explicitly; model infers or returns null.

# Schema iteration time (raw SDK): ~2 minutes
# (added key_hires: list[str] — touched Pydantic model, tool description, system prompt,
#  and this docstring; _anthropic_schema() picked up the field automatically via
#  model_json_schema(), so no schema wiring was needed manually — that part was free)
"""

import json
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(override=True)  # override=True in case ANTHROPIC_API_KEY is already set to '' in the OS env


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class PortcoUpdate(BaseModel):
    company_name: str
    reporting_period: str
    arr_usd: float | None = None
    customer_count: int | None = None
    mrr_growth_pct: float | None = None   # rarely stated explicitly — expect lots of nulls
    runway_months: float | None = None
    key_updates: list[str]
    risks_flagged: list[str]
    # default=[] so validation doesn't fail when no hires are mentioned in the email
    key_hires: list[str] = []
    sentiment: Literal["positive", "neutral", "concerning"]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

# FRICTION 1: Anthropic expects a JSON Schema dict under input_schema. Pydantic's
# model_json_schema() is close but emits a top-level "title" and may emit "$defs" for
# complex types. We strip those — they're not wrong, but Anthropic's schema validator
# can reject unknown top-level keys in some SDK versions.
def _anthropic_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)   # Anthropic doesn't need this
    schema.pop("$defs", None)   # no nested refs in this model, but defensive
    return schema

# FRICTION 2: Pydantic encodes `float | None` as anyOf:[{type:number},{type:null}].
# That's correct JSON Schema, but you have to trust Pydantic to emit it right and
# trust the model to honor it. There's no framework-level guarantee the model
# won't hallucinate a number for a field marked nullable.
_TOOL_NAME = "extract_portco_update"

_TOOL = {
    "name": _TOOL_NAME,
    "description": (
        "Extract structured portfolio company update fields from an investor update email. "
        "Populate numeric fields only when the email states or clearly implies a single value. "
        "Return null for any field that is ambiguous, missing, or expressed as a range. "
        "For key_hires, list each named hire as '<Name> (<role/title>)' when available; "
        "return an empty list if no specific hires are mentioned."
    ),
    "input_schema": _anthropic_schema(PortcoUpdate),
}

_SYSTEM = (
    "You are a VC analyst assistant. Extract structured data from portfolio company update emails. "
    "Be conservative on numeric fields: if a value is hedged, described as a range, or omitted, "
    "return null. Do not hallucinate figures. Sentiment should reflect the overall tone of the update. "
    "For key_hires, include only people explicitly named as new hires or recent joiners; "
    "do not include existing team members or open roles."
)

# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def extract_portco_update(email_text: str) -> PortcoUpdate:
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",  # fast + cheap for a structured-extraction baseline
        max_tokens=1024,
        system=_SYSTEM,
        tools=[_TOOL],
        # FRICTION 3: tool_choice syntax is Anthropic-specific — {"type":"tool","name":...}
        # forces a tool call, which is what we want. The alternative "auto" means the model
        # might just reply in text and skip the schema entirely, breaking downstream parsing.
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the portfolio company update from this email:\n\n"
                    + email_text
                ),
            }
        ],
    )

    # FRICTION 3 (continued): response.content is a list of content blocks (TextBlock,
    # ToolUseBlock, …). There's no .tool_calls shortcut like OpenAI's client exposes.
    # We have to find the right block type manually.
    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        # Shouldn't happen with tool_choice forced, but the SDK doesn't guarantee it.
        raise ValueError(
            f"No tool_use block in response. Stop reason: {response.stop_reason}. "
            f"Content: {response.content}"
        )

    # FRICTION 4: tool_block.input is already a parsed dict — the SDK deserializes the
    # JSON for you. If you call json.loads() on it you'll get a TypeError. Not obvious
    # from reading the API docs, which show raw JSON strings in the wire format.
    raw: dict = tool_block.input

    # FRICTION 5: validation is entirely our responsibility. The API happily returns
    # whatever the model produces; if sentiment comes back as "mixed" instead of a
    # Literal value, we only find out here. No auto-retry, no schema enforcement at
    # the transport layer.
    try:
        return PortcoUpdate.model_validate(raw)
    except ValidationError as exc:
        print(f"  [ValidationError] raw tool output:\n{json.dumps(raw, indent=2)}")
        raise


# ---------------------------------------------------------------------------
# Run against all three test emails
# ---------------------------------------------------------------------------

def main() -> None:
    email_dir = Path(__file__).parent / "test_emails"
    emails = sorted(email_dir.glob("email_*.txt"))

    if not emails:
        print("No test emails found in test_emails/")
        return

    for path in emails:
        print(f"\n{'=' * 62}")
        print(f"  {path.name}")
        print("=" * 62)
        email_text = path.read_text(encoding="utf-8")

        try:
            result = extract_portco_update(email_text)
            # FRICTION 6: mrr_growth_pct will be null for most emails — the field is
            # rarely stated explicitly in founder updates. The model has to either derive
            # it from ARR figures (which is a different metric) or admit it's missing.
            # A well-prompted BAML function could add a @describe annotation to handle
            # this ambiguity; here we just surface the null.
            print(result.model_dump_json(indent=2))
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
