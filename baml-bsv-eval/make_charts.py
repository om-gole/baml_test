"""
make_charts.py — Generate evaluation visuals from bench_results.json.

Produces:
  charts/fig1_token_cost.png   — schema encoding token breakdown (bar chart)
  charts/fig2_latency.png      — end-to-end latency mean ± p95 (bar chart)
  charts/fig3_overhead.png     — BAML framework overhead (bar chart)
  charts/fig4_failure.png      — failure rate on degenerate input (annotated)
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = json.loads(Path("bench_results.json").read_text())
OUT  = Path("charts")
OUT.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
RAW_C   = "#4A90D9"   # blue  — raw SDK
BAML_C  = "#E67E22"  # orange — BAML
GREY    = "#95A5A6"
DARK    = "#2C3E50"

plt.rcParams.update({
    "font.family":   "monospace",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
})

EMAIL_LABELS = ["Email 1\n(Structured)", "Email 2\n(Conversational)",
                "Email 3\n(Hedged)", "email_bad\n(Degenerate)"]
EMAIL_KEYS   = ["email_1", "email_2", "email_3", "email_bad"]


# ── Fig 1: Token cost breakdown ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("Fig 1 — Prompt Token Cost: Raw SDK vs BAML", fontsize=14, fontweight="bold", y=1.01)

pt = DATA["prompt_tokens"]

# Left panel: stacked bar — raw SDK
ax = axes[0]
systems = [pt[k]["raw"]["system"]      for k in EMAIL_KEYS]
users   = [pt[k]["raw"]["user_msg"]    for k in EMAIL_KEYS]
tools   = [pt[k]["raw"]["tool_schema"] for k in EMAIL_KEYS]
totals  = [pt[k]["raw"]["total"]       for k in EMAIL_KEYS]
x = np.arange(len(EMAIL_KEYS))
w = 0.5
b1 = ax.bar(x, systems, w, label="System prompt", color="#2980B9")
b2 = ax.bar(x, users,   w, bottom=systems, label="User message", color="#5DADE2")
b3 = ax.bar(x, tools,   w, bottom=[s+u for s,u in zip(systems,users)],
            label="Tool schema (JSON Schema)", color="#F39C12")
for i, (xp, tot) in enumerate(zip(x, totals)):
    ax.text(xp, tot + 15, str(tot), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(EMAIL_LABELS, fontsize=8)
ax.set_ylabel("Input tokens"); ax.set_title("Raw SDK (Anthropic tool_use)")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 2100)
ax.axhline(969, color="#F39C12", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(3.4, 980, "969 tok\ntool schema", fontsize=7, color="#E67E22", va="bottom")

# Right panel: stacked bar — BAML
ax = axes[1]
rules   = [pt[k]["baml"]["rules_and_email"]   for k in EMAIL_KEYS]
ctx     = [pt[k]["baml"]["ctx_output_format"]  for k in EMAIL_KEYS]
totals2 = [pt[k]["baml"]["total"]              for k in EMAIL_KEYS]
b4 = ax.bar(x, rules, w, label="Rules + email body", color="#27AE60")
b5 = ax.bar(x, ctx,   w, bottom=rules, label="ctx.output_format (schema)", color="#F1C40F")
for i, (xp, tot) in enumerate(zip(x, totals2)):
    ax.text(xp, tot + 15, str(tot), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(EMAIL_LABELS, fontsize=8)
ax.set_ylabel("Input tokens"); ax.set_title("BAML (ctx.output_format injection)")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 2100)
ax.axhline(123, color="#F1C40F", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(3.4, 134, "123 tok\nctx.output_format", fontsize=7, color="#B7950B", va="bottom")

fig.tight_layout()
fig.savefig(OUT / "fig1_token_cost.png", bbox_inches="tight")
plt.close(fig)
print("fig1_token_cost.png done")


# ── Fig 2: End-to-end latency ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Fig 2 — End-to-End Latency: Raw SDK (Haiku) vs BAML (Sonnet)", fontsize=14, fontweight="bold")

runs = DATA["runs"]
raw_means, raw_p95s, baml_means, baml_p95s = [], [], [], []

def p95(lst): return sorted(lst)[-1] if lst else None

for k in EMAIL_KEYS:
    raw_ok  = [r["total_ms"] for r in runs[k]["raw"]  if r.get("success")]
    baml_ok = [r["total_ms"] for r in runs[k]["baml"] if r.get("success")]
    raw_means.append(statistics.mean(raw_ok)  if raw_ok  else 0)
    raw_p95s.append(p95(raw_ok)               if raw_ok  else 0)
    baml_means.append(statistics.mean(baml_ok) if baml_ok else 0)
    baml_p95s.append(p95(baml_ok)              if baml_ok else 0)

x  = np.arange(len(EMAIL_KEYS))
w  = 0.32
b1 = ax.bar(x - w/2, raw_means,  w, label="Raw SDK mean",  color=RAW_C,  alpha=0.9)
b2 = ax.bar(x + w/2, baml_means, w, label="BAML mean",     color=BAML_C, alpha=0.9)

# p95 markers
for i, (xp, rv, bv) in enumerate(zip(x, raw_p95s, baml_p95s)):
    if rv: ax.plot(xp - w/2, rv, "v", color="#1A5276", markersize=8)
    if bv: ax.plot(xp + w/2, bv, "v", color="#784212", markersize=8)

# Value labels on bars
for bar in b1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 30, f"{h:.0f}ms",
                ha="center", va="bottom", fontsize=8, color=DARK)
for bar in b2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 30, f"{h:.0f}ms",
                ha="center", va="bottom", fontsize=8, color=DARK)

ax.set_xticks(x); ax.set_xticklabels(EMAIL_LABELS)
ax.set_ylabel("Latency (ms)")
ax.set_ylim(0, 6200)

# Note about model difference
ax.text(0.5, 0.97,
        "▼ = p95 (max of 10 runs)   |   Latency gap is model-driven (Haiku vs Sonnet), not framework-driven",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color=GREY,
        style="italic")

raw_patch  = mpatches.Patch(color=RAW_C,  label="Raw SDK (Haiku)")
baml_patch = mpatches.Patch(color=BAML_C, label="BAML (Sonnet)")
ax.legend(handles=[raw_patch, baml_patch], loc="upper right", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig2_latency.png", bbox_inches="tight")
plt.close(fig)
print("fig2_latency.png done")


# ── Fig 3: BAML framework overhead ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Fig 3 — BAML Framework Overhead vs Direct Sonnet Call", fontsize=14, fontweight="bold")

oh = DATA["overhead_analysis"]
baml_means2, direct_means, est_ohs = [], [], []

for k in EMAIL_KEYS:
    baml_ok = [r["total_ms"] for r in runs[k]["baml"] if r.get("success")]
    oh_runs  = oh.get(k, {}).get("direct_sonnet", [])
    direct_ok = [r["llm_ms"] for r in oh_runs if r.get("success") and r.get("llm_ms")]
    bm = statistics.mean(baml_ok)   if baml_ok   else None
    dm = statistics.mean(direct_ok) if direct_ok else None
    baml_means2.append(bm or 0)
    direct_means.append(dm or 0)
    est_ohs.append(round(bm - dm, 0) if bm and dm else 0)

x = np.arange(len(EMAIL_KEYS))
w = 0.32
ax.bar(x - w/2, direct_means, w, label="Direct Sonnet (no BAML)", color=GREY,  alpha=0.9)
ax.bar(x + w/2, baml_means2,  w, label="BAML total",              color=BAML_C, alpha=0.9)

# Overhead annotation
for i, (xp, oh_ms) in enumerate(zip(x, est_ohs)):
    if oh_ms != 0:
        color  = "#E74C3C" if oh_ms > 0 else "#27AE60"
        symbol = f"+{oh_ms:.0f}ms" if oh_ms > 0 else f"{oh_ms:.0f}ms"
        ypos   = max(baml_means2[i], direct_means[i]) + 100
        ax.text(xp + w/2, ypos, symbol, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

# email_bad has no BAML runs — annotate
ax.text(3 + w/2, 200, "10/10\nBAML fails\n(BamlValidationError)", ha="center",
        va="bottom", fontsize=8, color="#E74C3C", style="italic")

ax.set_xticks(x); ax.set_xticklabels(EMAIL_LABELS)
ax.set_ylabel("Latency (ms)")
ax.set_ylim(0, 6500)
ax.text(0.5, 0.97,
        "Overhead = BAML total − direct Sonnet mean  |  n=3 direct baseline runs",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color=GREY, style="italic")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig3_overhead.png", bbox_inches="tight")
plt.close(fig)
print("fig3_overhead.png done")


# ── Fig 4: Schema encoding delta (single striking bar) ────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Fig 4 — Schema Encoding: Token Cost Comparison (Email 1)", fontsize=14, fontweight="bold")

r = DATA["prompt_tokens"]["email_1"]["raw"]
b = DATA["prompt_tokens"]["email_1"]["baml"]

categories = ["Raw SDK\ntool schema\n(JSON Schema)", "BAML\nctx.output_format\n(compact text)"]
values     = [r["tool_schema"], b["ctx_output_format"]]
colors     = [RAW_C, BAML_C]
bars = ax.bar(categories, values, color=colors, width=0.4, alpha=0.9)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{val} tokens", ha="center", va="bottom", fontsize=14, fontweight="bold")

ax.set_ylabel("Input tokens consumed by schema encoding")
ax.set_ylim(0, 1200)

# Reduction annotation
reduction_pct = round((values[0] - values[1]) / values[0] * 100, 0)
ax.annotate("", xy=(1, values[1] + 20), xytext=(0, values[0] - 20),
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=2))
ax.text(0.5, (values[0] + values[1]) / 2, f"  {reduction_pct:.0f}%\n  smaller",
        ha="left", va="center", fontsize=13, color=DARK, fontweight="bold",
        transform=ax.get_yaxis_transform())

ax.text(0.5, 0.97,
        "Same schema, same 10-field extraction task — only the encoding format differs",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color=GREY, style="italic")

fig.tight_layout()
fig.savefig(OUT / "fig4_schema_tokens.png", bbox_inches="tight")
plt.close(fig)
print("fig4_schema_tokens.png done")

print(f"\nAll charts saved to {OUT.resolve()}")
