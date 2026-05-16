// extraction.baml — Portfolio company update extractor for BSV eval

// Schema iteration time (BAML): ~0 seconds for the field itself + 0.8s for `baml-cli generate`
// key_hires was already present when the schema was first defined to match raw_baseline.py.
// If added fresh: 1 line in the class block + optional /// docstring = done. No tool description,
// no system prompt, no separate validation logic to update — ctx.output_format renders it automatically.
// Compare: raw SDK took ~2 minutes across 4 separate prose surfaces (Pydantic model, tool
// description string, system prompt string, module docstring) with no compile-time link between them.

enum Sentiment {
  Positive
  Neutral
  Concerning
}

class PortcoUpdate {
  company_name string
  reporting_period string

  /// Annual recurring revenue in USD. Null if not mentioned or expressed as a range.
  arr_usd float?

  /// Total paying customer count. Null if ambiguous or not stated.
  customer_count int?

  /// MRR growth as a percentage. Null if not explicitly stated (common — do not derive from ARR).
  mrr_growth_pct float?

  /// Months of runway remaining at current burn. Null if hedged or not mentioned.
  runway_months float?

  key_updates string[]
  risks_flagged string[]

  /// Named new hires and recent joiners only. Format: "Name (Role)". Empty list if none mentioned.
  key_hires string[]

  sentiment Sentiment
}

function ExtractPortcoUpdate(email_text: string) -> PortcoUpdate {
  client CustomSonnet4
  prompt #"
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
    {{ email_text }}

    {{ ctx.output_format }}
  "#
}
