# Eval Notes: Raw Anthropic SDK vs BAML
## Portfolio Company Update Extraction — BSV Eval

---

## Setup

| | Raw SDK (`raw_baseline.py`) | BAML (`baml_extraction.py`) |
|---|---|---|
| Model | claude-haiku-4-5-20251001 | claude-sonnet-4-20250514 |
| Extraction mechanism | Tool use (manual JSON Schema) | `{{ ctx.output_format }}` (auto-rendered) |
| Schema enforcement | Pydantic, called manually | BAML runtime, enforced before caller sees result |
| Output type | `PortcoUpdate` (hand-defined Pydantic) | `PortcoUpdate` (generated Pydantic) |

Note: the model difference (Haiku vs Sonnet) means output quality isn't a fair apples-to-apples
comparison on the real emails. The degenerate input test is more informative because the
difference there is entirely about schema enforcement behavior, not model reasoning.

---

## Degenerate Input Test: `email_bad.txt`

Input: `"hey lmk when u can chat"`

### Raw SDK result

```json
{
  "company_name": "<UNKNOWN>",
  "reporting_period": "<UNKNOWN>",
  "arr_usd": null,
  "customer_count": null,
  "mrr_growth_pct": null,
  "runway_months": null,
  "key_updates": [],
  "risks_flagged": [],
  "key_hires": [],
  "sentiment": "neutral"
}
```

**Outcome: validation passed. Silent failure.**

The model, forced by `tool_choice={"type":"tool","name":"..."}` to fill every field, invented
`"<UNKNOWN>"` placeholder strings for `company_name` and `reporting_period`. These satisfy the
`string` type constraint in the JSON Schema tool definition. Pydantic accepted them without
complaint. The caller gets a fully-formed `PortcoUpdate` object with no indication that the
input was garbage.

This is a silent data quality failure. A downstream system writing these records to a database
or dashboard would ingest `company_name: "<UNKNOWN>"` without any alarm.

### BAML result

**Outcome: `BamlValidationError` raised. Loud failure.**

The model returned `null` for `company_name` and `reporting_period` — honest, since there is
genuinely no company or period to extract. BAML's runtime enforced the schema before returning
control to Python: `company_name: string` (non-nullable) cannot be null, so it threw:

```
Failed to parse field company_name: Expected string, got null
Failed to parse field reporting_period: Expected string, got null
```

The error message names the exact failing fields and their expected vs. actual types. The raw
response is included in the exception for debugging.

### Comparison

| Behavior | Raw SDK | BAML |
|---|---|---|
| Model response | `"<UNKNOWN>"` (hallucinated placeholder) | `null` (honest) |
| Validation | Passed | Failed (`BamlValidationError`) |
| Failure mode | Silent — garbage data returned as valid | Loud — exception with field-level detail |
| Caller awareness | None (must inspect output values manually) | Immediate (exception is raised) |
| Debuggability | Must add your own field-level checks | Error includes raw LLM output + prompt |

**Neither result is "correct" in a product sense.** Both surface a schema design question:
should `company_name` and `reporting_period` be nullable (`string?`) to handle non-portco
inputs gracefully? Or should there be an upstream routing check (e.g., a `ClassifyEmail`
function that rejects non-update emails before reaching `ExtractPortcoUpdate`)?

The BAML failure is arguably more useful for catching this during development. The raw SDK
failure would pass silently through a test suite that only checks for exceptions.

---

## Schema Iteration Observations

### Adding `key_hires: list[str]`

| Step | Raw SDK | BAML |
|---|---|---|
| Model field | 1 line in `PortcoUpdate` | 1 line in `.baml` class |
| Tool description | Manual prose update | Not needed — `{{ ctx.output_format }}` renders it |
| System prompt | Manual prose update | Not needed |
| Docstring / other | Updated module docstring | Updated `.baml` file comment |
| Regenerate step | None | `baml-cli generate` (~0.8 seconds) |
| Total surfaces touched | 4 | 1 |
| Risk of drift | High — tool description, system prompt, and Pydantic model can silently diverge | Low — single source of truth in `.baml` |

### Where raw SDK friction accumulates

1. **Manual JSON Schema wiring.** `_anthropic_schema()` derives the schema from Pydantic, which
   helps, but the adapter is hand-written and strips keys (`title`, `$defs`) that may or may not
   matter depending on SDK version.

2. **Two separate prose surfaces for field semantics.** The tool `description` and the system
   prompt both carry field-level guidance. They can drift. There's no lint or compile check.

3. **`tool_block.input` is a pre-parsed dict.** Not obvious from docs; easy to double-parse.

4. **No transport-layer validation.** Pydantic is the only guard and must be called manually.
   A model returning `sentiment: "mixed"` only fails at runtime, on that specific call.

5. **Silent degenerate-input failures.** The tool-call forcing mechanism causes the model to
   invent placeholder strings rather than return null, so garbage inputs produce garbage-but-valid
   output.

### Where BAML friction accumulates

1. **Enum values must be PascalCase.** `Sentiment.positive` is a compile error; must use
   `Sentiment.Positive`. Downstream consumers expecting lowercase need a mapping layer.

2. **`baml-cli generate` is a required step.** Forgetting to regenerate after a `.baml` edit
   causes a confusing mismatch between the schema the model sees and the types Python uses.

3. **`load_dotenv(override=True)` ordering.** BAML's Rust runtime reads env vars through
   Python's `os.environ`. If the OS already has `ANTHROPIC_API_KEY=''` set, `load_dotenv()`
   without `override=True` silently leaves it empty. The BAML warning is informative, but the
   fix (override) is non-obvious.

4. **Required string fields fail loudly on degenerate input.** Whether this is friction or a
   feature depends on the use case. For a production pipeline, it's a feature. For a scratchpad
   that wants partial results, it's friction.

---

## Extraction Quality Observations (real emails)

### Email 1 (Narrate AI — clean, structured)

Both approaches extracted the same core values. No meaningful difference at this input quality.
Notable: both correctly returned `mrr_growth_pct: null` despite ARR growth being explicitly
stated (50% QoQ) — the prompt instruction to not derive MRR growth from ARR held.

### Email 2 (FieldLogic — conversational, implicit numbers)

Both returned `arr_usd: null` for "comfortably into seven figures" — good conservatism.
Both extracted `customer_count: 50` from "just over 50." Both correctly identified
James Kowalski (Head of Sales) and excluded the unnamed Tesla engineers from `key_hires`.
`runway_months: null` for "a couple years" — correct, it's a hedge.

### Email 3 (Helix Tax — missing data, hedged language)

Both returned all numeric fields as null. `customer_count: null` for "somewhere between 20
and 35" — correct (range, not a single value). Both extracted Leila Vasquez (Head of Product)
and correctly excluded the VP of Sales still being searched for.
`sentiment: "Concerning"` / `"Concerning"` — strong agreement on a genuinely ambiguous email.

---

## Open Questions for Next Steps

- Should `company_name` and `reporting_period` be `string?` to handle non-portco inputs,
  or should a separate `ClassifyEmail` guard function run first?
- The raw SDK uses Haiku and BAML uses Sonnet — a controlled model comparison would require
  both to use the same model.
- Neither approach has retry logic for the degenerate case. A production pipeline would want
  a fallback: detect the validation failure and either skip the record or route it to a
  human review queue.
- BAML's `BamlValidationError` on the degenerate input is the right signal for a circuit
  breaker. The raw SDK's silent pass is the wrong signal for any downstream consumer that
  trusts the output schema.
