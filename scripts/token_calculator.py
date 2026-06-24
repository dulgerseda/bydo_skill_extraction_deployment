"""
=============================================================
TOKEN CALCULATOR — BYDO Project
=============================================================
Reads tokens/token_usage.json        → extraction pipeline
Reads evaluation/results/*.json      → evaluation pipeline

Shows cost breakdown by source and grand total.

Usage:
  python3 scripts/token_calculator.py
=============================================================
"""

import json
from pathlib import Path
from datetime import datetime

TOKEN_FILE   = Path("tokens/token_usage.json")
EVAL_DIR     = Path("evaluation/results")

PRICING = {
    "gpt-4o": {
        "input":  2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input":  0.15,
        "output": 0.60,
    },
}


def compute_cost(prompt_tokens, completion_tokens, model):
    pricing = PRICING.get(model, PRICING["gpt-4o"])
    return round(
        (prompt_tokens / 1_000_000 * pricing["input"]) +
        (completion_tokens / 1_000_000 * pricing["output"]),
        6
    )


def load_extraction_entries():
    """Load token entries from tokens/token_usage.json."""
    if not TOKEN_FILE.exists():
        return []
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        entry["source"] = "extraction"
    return entries


def load_evaluation_entries():
    """Load token entries from evaluation/results/*.json."""
    entries = []
    if not EVAL_DIR.exists():
        return entries
    for f in sorted(EVAL_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            usage = data.get("token_usage", {})
            if not usage:
                continue
            # Determine source from filename
            fname = f.stem
            if "extraction" in fname:
                source = "evaluation_extraction"
            elif "consistency" in fname:
                source = "evaluation_consistency"
            elif "esco" in fname:
                source = "evaluation_esco"
            else:
                source = "evaluation"
            entries.append({
                "source":                   source,
                "timestamp":                data.get("timestamp", fname),
                "model":                    "gpt-4o",
                "total_prompt_tokens":      usage.get("input_tokens", 0),
                "total_completion_tokens":  usage.get("output_tokens", 0),
                "total_cost_usd":           usage.get("total_cost_usd", 0.0),
                "jds":                      list(data.get("jds", {}).keys())
            })
        except Exception:
            continue
    return entries


if __name__ == "__main__":

    extraction_entries = load_extraction_entries()
    evaluation_entries = load_evaluation_entries()
    all_entries        = extraction_entries + evaluation_entries

    print(f"\n{'═' * 56}")
    print(f"  BYDO TOKEN USAGE REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═' * 56}")

    if not all_entries:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text("[]", encoding="utf-8")
        print("\n  No token data found.")
        print(f"{'═' * 56}\n")
        exit(0)

    by_source = {}
    for entry in all_entries:
        source = entry.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(entry)

    grand_prompt     = 0
    grand_completion = 0
    grand_cost       = 0.0

    for source, runs in sorted(by_source.items()):
        print(f"\n  {source.upper()}")
        print(f"  {'─' * 50}")

        src_prompt     = 0
        src_completion = 0
        src_cost       = 0.0

        for run in runs:
            ts             = run.get("timestamp", "unknown")
            model          = run.get("model", "gpt-4o")
            prompt_tok     = run.get("total_prompt_tokens", 0)
            completion_tok = run.get("total_completion_tokens", 0)
            cost           = run.get("total_cost_usd", compute_cost(prompt_tok, completion_tok, model))
            jd_count       = len(run.get("jds", []))

            print(f"  Run {ts}  |  model: {model}  |  {jd_count} JDs")
            print(f"    prompt tokens:     {prompt_tok:>8,}")
            print(f"    completion tokens: {completion_tok:>8,}")
            print(f"    cost:              ${cost:.4f}")

            src_prompt     += prompt_tok
            src_completion += completion_tok
            src_cost       += cost

        print(f"  {'─' * 50}")
        print(f"  Subtotal  →  {src_prompt:,} + {src_completion:,} tokens  |  ${src_cost:.4f}")

        grand_prompt     += src_prompt
        grand_completion += src_completion
        grand_cost       += src_cost

    print(f"\n{'═' * 56}")
    print(f"  GRAND TOTAL")
    print(f"  prompt tokens:     {grand_prompt:>10,}")
    print(f"  completion tokens: {grand_completion:>10,}")
    print(f"  total cost:        ${grand_cost:.4f}")
    print(f"{'═' * 56}\n")
