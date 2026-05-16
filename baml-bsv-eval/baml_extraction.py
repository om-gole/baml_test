"""
baml_extraction.py — same extraction task as raw_baseline.py, using the BAML generated client.

Output format is identical (model_dump_json indent=2) so results are directly comparable.

One cosmetic diff to expect: sentiment values are PascalCase here ("Positive", "Neutral",
"Concerning") because BAML enums must start with an uppercase letter. raw_baseline.py uses
lowercase Literals ("positive", …). The underlying semantics are the same.
"""

from pathlib import Path

from dotenv import load_dotenv

# override=True: Windows may already have ANTHROPIC_API_KEY='' in the environment;
# without override, load_dotenv silently skips keys that are already set (even to empty).
load_dotenv(override=True)

from baml_client import b  # noqa: E402  (import after dotenv is intentional)


def extract_portco_update(email_text: str):
    # Single line. No schema wiring, no tool block parsing, no manual Pydantic validation.
    # BAML handles: prompt rendering, tool-call forcing, response parsing, and type coercion.
    return b.ExtractPortcoUpdate(email_text=email_text)


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
            print(result.model_dump_json(indent=2))
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
