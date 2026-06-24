"""
extract_skills_api.py — BYDO Skill Extraction
=============================================================
Reads each job description from jobs/ and extracts structured
skills using GPT-4o based on the prompt in prompts/prompt_v12.txt.

Steps:
1. Loads config (model, temperature, max_tokens) from config.json
2. Loads prompt template from prompts/prompt_v12.txt
3. For each JD in jobs/:
   - Injects JD text into the prompt
   - Calls GPT-4o API
   - Parses JSON response
   - Saves result to answers/
4. Logs token usage and cost to tokens/token_usage.json

Reads from:  jobs/  +  prompts/prompt_v12.txt  +  config.json
Writes to:   answers/  +  tokens/token_usage.json

Usage:
    python3 scripts/extract_skills_api.py
    python3 scripts/extract_skills_api.py --file JD_01_DataScientist.txt
=============================================================
"""

import os
import json
import glob
from datetime import datetime
from openai import OpenAI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "prompt_v12.txt")
JOBS_DIR    = os.path.join(BASE_DIR, "jobs")
OUTPUT_DIR  = os.path.join(BASE_DIR, "answers")

# ── Token log ─────────────────────────────────────────────────────────────────
TOKEN_LOG_PATH = os.path.join(BASE_DIR, "tokens", "token_usage.json")

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")


# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

api_key     = os.environ.get("OPENAI_API_KEY") or config["openai_api_key"]
client      = OpenAI(api_key=api_key)
MODEL       = config["model"]
TEMPERATURE = config["temperature"]
MAX_COMPLETION_TOKENS  = config["max_tokens"]


# ── Load prompt template ──────────────────────────────────────────────────────
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# ── Prepare output folder ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Process each job ──────────────────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(JOBS_DIR, "*.txt")))
job_files = [f for f in all_files if os.path.basename(f) == args.file] if args.file else all_files

if not job_files:
    print("❌ No .txt files found in jobs/ folder.")
    exit(1)

print(f"📂 Found {len(job_files)} job file(s). Starting extraction...\n")

run_token_log = []  # ← NEW

for job_path in job_files:
    job_filename = os.path.basename(job_path)
    job_stem     = os.path.splitext(job_filename)[0]

    with open(job_path, "r", encoding="utf-8") as f:
        job_text = f.read().strip()

    full_prompt = prompt_template.replace('"……………………"', f'"{job_text}"')

    output_filename = f"{job_stem}_{TIMESTAMP}.json"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    print(f"🔄 Processing: {job_filename}")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE,
            messages=[
        {"role": "user", "content": full_prompt}
    ]
)

        raw_output = response.choices[0].message.content.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[-1]
            raw_output = raw_output.rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_output)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({job_stem: parsed}, f, ensure_ascii=False, indent=2)

        skill_count = len(parsed.get("skills", []))
        print(f"  ✓ Saved: {output_filename} ({skill_count} skills)")

        # ── Token tracking ────────────────────────────────────────────────────
        usage = response.usage
        cost  = (usage.prompt_tokens     / 1_000_000 * 2.50) + \
                (usage.completion_tokens / 1_000_000 * 10.00)
        run_token_log.append({
            "jd":                job_stem,
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens":      usage.total_tokens,
            "cost_usd":          round(cost, 6),
        })
        print(f"  tokens: {usage.total_tokens:,} | cost: ${cost:.4f}\n")
        # ─────────────────────────────────────────────────────────────────────

    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse error for {job_filename}: {e}")
        error_path = output_path.replace(".json", "_ERROR.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        print(f"   ⚠️  Raw output → {os.path.basename(error_path)}\n")

    except Exception as e:
        print(f"   ❌ API error for {job_filename}: {e}\n")

# ── Save token log ─────────────────────────────────────────────────────────────
total_prompt     = sum(j["prompt_tokens"]     for j in run_token_log)
total_completion = sum(j["completion_tokens"] for j in run_token_log)
total_cost       = sum(j["cost_usd"]          for j in run_token_log)

run_summary = {
    "timestamp":               TIMESTAMP,
    "model":                   MODEL,
    "jds":                     run_token_log,
    "total_prompt_tokens":     total_prompt,
    "total_completion_tokens": total_completion,
    "total_tokens":            total_prompt + total_completion,
    "total_cost_usd":          round(total_cost, 6),
}

os.makedirs(os.path.dirname(TOKEN_LOG_PATH), exist_ok=True)
existing = []
if os.path.exists(TOKEN_LOG_PATH):
    with open(TOKEN_LOG_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
existing.append(run_summary)
with open(TOKEN_LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

print("─" * 50)
print(f"✓ Token log → tokens/token_usage.json")
print(f"  This run: {total_prompt + total_completion:,} tokens | ${total_cost:.4f}")
print("\nAll done! Check answers/ folder.")
