"""
extract_skills_api.py — BYDO Skill Extraction
=============================================================
Reads each job description from jobs/ and extracts structured
skills using GPT-4o based on the prompt in prompts/. Where the
document reports work that was carried out, the same call also
returns that work as project_activities, in certificate wording.

Steps:
1. Reads the API key from OPENAI_API_KEY, from .env, or from config.json
2. Loads config (model, prompt, temperature, seed, max_tokens) from config.json
3. Loads the prompt template named in config
4. For each JD in jobs/:
   - Injects JD text into the prompt
   - Returns the cached answer if this exact request was made before
   - Otherwise calls the API with a schema-enforced response
   - Saves result to answers/
5. Logs token usage and cost to tokens/token_usage.json

Reproducibility: the model is pinned to a dated snapshot, temperature
and seed are fixed, the response shape is enforced by a JSON schema,
and identical requests are served from cache/extractions/ so the same
document can never come back with different skills. See README.

Reads from:  jobs/  +  prompts/  +  config.json  +  .env  +  cache/extractions/
Writes to:   answers/  +  tokens/token_usage.json  +  cache/extractions/

Usage:
    python3 scripts/extract_skills_api.py
    python3 scripts/extract_skills_api.py --file JD_01_DataScientist.txt
    python3 scripts/extract_skills_api.py --no-cache   # force fresh API calls
=============================================================
"""

import os
import json
import glob
import hashlib
from datetime import datetime
from openai import OpenAI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None)
parser.add_argument("--no-cache", action="store_true",
                    help="ignore the cache and call the API even for a known request")
parser.add_argument("--print-cache-key", action="store_true",
                    help="print the cache key of each job and exit, without calling the API")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
ENV_PATH    = os.path.join(BASE_DIR, ".env")
JOBS_DIR    = os.path.join(BASE_DIR, "jobs")
OUTPUT_DIR  = os.path.join(BASE_DIR, "answers")
CACHE_DIR   = os.path.join(BASE_DIR, "cache", "extractions")

# ── Token log ─────────────────────────────────────────────────────────────────
TOKEN_LOG_PATH = os.path.join(BASE_DIR, "tokens", "token_usage.json")

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")


# ── API key ───────────────────────────────────────────────────────────────────
# The key is a secret and the settings are not, so the two are kept apart: the key
# comes from the environment, config.json holds only what has to stay reproducible.

PLACEHOLDER_KEYS = ("your-api-key-here", "sk-...")


def load_dotenv(path):
    """Read KEY=value lines from .env into the environment, without a dependency.

    Variables that are already set win, so an export in the shell and `docker run
    -e` still override the file.
    """
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            if name and not os.environ.get(name):
                os.environ[name] = value.strip().strip('"').strip("'")


def resolve_api_key(config):
    """The key, and where it came from. Environment first, config.json last.

    config.json is still read so that installations which already keep the key
    there keep working, but nothing requires it to be written down any more.
    """
    candidates = [(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY"),
                  (config.get("openai_api_key"), "config.json")]
    for value, origin in candidates:
        if isinstance(value, str) and value.strip() and value.strip() not in PLACEHOLDER_KEYS:
            return value.strip(), origin

    print("❌ No OpenAI API key found. Set it in any one of these ways:\n"
          "\n"
          "     export OPENAI_API_KEY=sk-...          this shell only\n"
          "     echo 'OPENAI_API_KEY=sk-...' > .env   every run, Docker included\n"
          '     "openai_api_key": "sk-..."            in config.json (not recommended)\n'
          "\n"
          f"   .env and config.json both belong in {BASE_DIR}\n")
    exit(1)


# ── Load config ───────────────────────────────────────────────────────────────
if not os.path.isfile(CONFIG_PATH):
    print(f"❌ No config.json at {CONFIG_PATH}.\n"
          "   It holds the model and sampling settings; see the README for the contents.\n"
          "   The API key does not belong in it — put it in the environment instead.\n")
    exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

load_dotenv(ENV_PATH)
api_key, KEY_ORIGIN = resolve_api_key(config)
client      = OpenAI(api_key=api_key)
MODEL       = config["model"]
TEMPERATURE = config["temperature"]
SEED        = config.get("seed", 42)
PROMPT_FILE = config.get("prompt", "prompt_v13.txt")
JSON_MODE   = config.get("json_mode", True)
# Constrained decoding against the full schema costs a lot of recall: measured on
# the learning journal it returned 5 skills where JSON mode returned 18, on both
# prompt versions. So the schema validates the answer instead of shaping it.
STRICT_SCHEMA = config.get("strict_schema", False)
MAX_COMPLETION_TOKENS  = config["max_tokens"]

PROMPT_PATH = os.path.join(BASE_DIR, "prompts", PROMPT_FILE)

if MODEL in ("gpt-4o", "gpt-4o-mini"):
    print(f"⚠️  model '{MODEL}' is a floating alias that OpenAI repoints to new versions.\n"
          f"    Pin a dated snapshot (e.g. gpt-4o-2024-11-20) for reproducible output.\n")


# ── Response schema ───────────────────────────────────────────────────────────
# Used to check the answer after parsing, and only used to constrain generation
# when strict_schema is switched on in config.
SKILL_SCHEMA = {
    "name": "skill_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["domain", "project_activities", "skills"],
        "properties": {
            "domain": {"type": "string"},
            # What the document reports as done, in certificate wording. Empty for a
            # job description, which advertises a role instead of reporting work.
            "project_activities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["activity", "source"],
                    "properties": {
                        "activity": {"type": "string"},
                        "source":   {"type": "string"},
                    },
                },
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["noun", "verb", "esco_form", "category",
                                 "skill_type", "domain", "tool", "action", "source"],
                    "properties": {
                        "noun":       {"type": "string"},
                        "verb":       {"type": "string"},
                        "esco_form":  {"type": "string"},
                        "category":   {"type": "string", "enum": ["K", "OS", "SF", "IPS"]},
                        "skill_type": {"type": "string",
                                       "enum": ["Tool", "Methodology", "Knowledge", "Soft Skill"]},
                        "domain":     {"type": "string"},
                        "tool":       {"type": ["string", "null"]},
                        "action":     {"type": ["string", "null"]},
                        "source":     {"type": "string"},
                    },
                },
            },
        },
    },
}


# ── Load prompt template ──────────────────────────────────────────────────────
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()

PROMPT_HASH = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()[:12]

CATEGORIES  = SKILL_SCHEMA["schema"]["properties"]["skills"]["items"]["properties"]["category"]["enum"]
SKILL_TYPES = SKILL_SCHEMA["schema"]["properties"]["skills"]["items"]["properties"]["skill_type"]["enum"]
REQUIRED    = SKILL_SCHEMA["schema"]["properties"]["skills"]["items"]["required"]
ACTIVITY_REQUIRED = SKILL_SCHEMA["schema"]["properties"]["project_activities"]["items"]["required"]


# ── Answer validation ─────────────────────────────────────────────────────────

def validate_payload(parsed):
    """Report shape problems instead of letting them travel downstream silently."""
    problems = []

    if not isinstance(parsed, dict) or "skills" not in parsed:
        return ["payload has no 'skills' key"]
    if not isinstance(parsed.get("domain"), str) or not parsed["domain"].strip():
        problems.append("missing top-level 'domain'")

    for i, skill in enumerate(parsed["skills"]):
        label = skill.get("noun", f"row {i}")
        for field in REQUIRED:
            if field not in skill:
                problems.append(f"{label}: missing '{field}'")
        if skill.get("category") not in CATEGORIES:
            problems.append(f"{label}: category {skill.get('category')!r} not in {CATEGORIES}")
        if skill.get("skill_type") not in SKILL_TYPES:
            problems.append(f"{label}: skill_type {skill.get('skill_type')!r} not in {SKILL_TYPES}")
        if skill.get("category") != "IPS" and (skill.get("tool") or skill.get("action")):
            problems.append(f"{label}: tool/action set on a non-IPS skill")

    # An empty list is a legitimate answer, a missing key is not: it means the model
    # ignored the section rather than judging the document to report no project work.
    activities = parsed.get("project_activities")
    if not isinstance(activities, list):
        problems.append("missing top-level 'project_activities'"
                        if activities is None else "'project_activities' is not a list")
    else:
        for i, activity in enumerate(activities):
            if not isinstance(activity, dict):
                problems.append(f"activity {i}: not an object")
                continue
            for field in ACTIVITY_REQUIRED:
                if not isinstance(activity.get(field), str) or not activity[field].strip():
                    problems.append(f"activity {i}: missing '{field}'")

    return problems


# ── Request cache ─────────────────────────────────────────────────────────────

def cache_key(job_text):
    """Identity of a request: same key means the answer cannot legitimately differ."""
    parts = [job_text, MODEL, PROMPT_HASH, str(TEMPERATURE), str(SEED),
             str(MAX_COMPLETION_TOKENS),
             f"json_mode={JSON_MODE}", f"strict={STRICT_SCHEMA}"]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


def response_format_arg():
    if STRICT_SCHEMA:
        return {"response_format": {"type": "json_schema", "json_schema": SKILL_SCHEMA}}
    if JSON_MODE:
        return {"response_format": {"type": "json_object"}}
    return {}


def cache_read(key):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if args.no_cache or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_write(key, job_stem, parsed, meta):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "jd": job_stem,
            "model": MODEL,
            "prompt_file": PROMPT_FILE,
            "prompt_hash": PROMPT_HASH,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "cached_at": datetime.now().isoformat(timespec="seconds"),
            "meta": meta,
            "parsed": parsed,
        }, f, ensure_ascii=False, indent=2)

# ── Prepare output folder ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Process each job ──────────────────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(JOBS_DIR, "*.txt")))
job_files = [f for f in all_files if os.path.basename(f) == args.file] if args.file else all_files

if not job_files:
    print("❌ No .txt files found in jobs/ folder.")
    exit(1)

if args.print_cache_key:
    for job_path in job_files:
        with open(job_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"{os.path.splitext(os.path.basename(job_path))[0]} {cache_key(text)}")
    exit(0)

print(f"🔑 API key from {KEY_ORIGIN}")
if KEY_ORIGIN == "config.json":
    print("   The key is sitting on disk among your settings. To take it out, move it\n"
          "   into .env as OPENAI_API_KEY and delete it from config.json.")

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

    key    = cache_key(job_text)
    cached = cache_read(key)

    if cached:
        parsed = cached["parsed"]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({job_stem: parsed}, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved: {output_filename} ({len(parsed.get('skills', []))} skills, "
              f"{len(parsed.get('project_activities', []))} activities)")
        print(f"  cache hit {key[:12]} from {cached['cached_at']} | cost: $0.0000\n")
        run_token_log.append({
            "jd":        job_stem,
            "cache_key": key,
            "cached":    True,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cost_usd":  0.0,
        })
        continue

    raw_output = ""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            **response_format_arg(),
            messages=[
        {"role": "user", "content": full_prompt}
    ]
)

        choice = response.choices[0]

        if getattr(choice.message, "refusal", None):
            print(f"   ❌ Model refused {job_filename}: {choice.message.refusal}\n")
            continue

        if choice.finish_reason == "length":
            # the schema guarantees valid JSON only for a complete response, so a
            # truncated one would fail to parse with a misleading error
            print(f"   ❌ Response hit the {MAX_COMPLETION_TOKENS} token ceiling and was "
                  f"truncated. Raise max_tokens in config.json.\n")
            continue

        raw_output = choice.message.content.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[-1]
            raw_output = raw_output.rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_output)

        problems = validate_payload(parsed)
        if problems:
            print(f"   ⚠️  {len(problems)} shape problem(s):")
            for p in problems[:8]:
                print(f"       - {p}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({job_stem: parsed}, f, ensure_ascii=False, indent=2)

        skill_count = len(parsed.get("skills", []))
        print(f"  ✓ Saved: {output_filename} ({skill_count} skills, "
              f"{len(parsed.get('project_activities', []))} activities)")

        # ── Token tracking ────────────────────────────────────────────────────
        usage = response.usage
        cost  = (usage.prompt_tokens     / 1_000_000 * 2.50) + \
                (usage.completion_tokens / 1_000_000 * 10.00)
        run_token_log.append({
            "jd":                job_stem,
            "cache_key":         key,
            "cached":            False,
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens":      usage.total_tokens,
            "cost_usd":          round(cost, 6),
            # a changed fingerprint means OpenAI changed the backend, which is
            # the one thing that can still move the output at a fixed seed
            "system_fingerprint": response.system_fingerprint,
        })
        print(f"  tokens: {usage.total_tokens:,} | cost: ${cost:.4f} | "
              f"fingerprint: {response.system_fingerprint}\n")
        # ─────────────────────────────────────────────────────────────────────

        cache_write(key, job_stem, parsed, {
            "system_fingerprint": response.system_fingerprint,
            "total_tokens":       usage.total_tokens,
        })

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
    "temperature":             TEMPERATURE,
    "seed":                    SEED,
    "prompt_file":             PROMPT_FILE,
    "prompt_hash":             PROMPT_HASH,
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
