"""
normalize_answers.py — BYDO Skill Normalization
=============================================================
Reads raw extraction output from answers/ and applies the
following normalization steps:

1. Tool name expansion
   Expands short tool names to full names.
   e.g. "Excel" → "Microsoft Excel"

2. ESCO form cleaning
   Removes filler words from esco_form field.
   e.g. "Utilize Python effectively for data analysis" → "use Python"

2b. ESCO query derivation
   Builds esco_query from the stable fields (noun + category + skill_type
   + tool + action) so the ESCO lookup does not depend on how the model
   happened to word esco_form on that call.

3. Skill type normalization
   Determines skill_type based on category and noun:
   - SF  → Soft Skill
   - IPS → Tool (if valid tool + action) or Methodology
   - K   → Tool or Knowledge (based on noun signals)
   - OS  → Methodology

4. IPS validation
   Converts invalid IPS skills (e.g. tool = "machine learning") to OS.

5. Deduplication
   Removes duplicate skills using fuzzy matching (threshold: 80).

6. Project activity tidying
   Strips bullet characters and trailing full stops from project_activities
   so they render as certificate bullets. The wording is left untouched.

Reads from:  answers/
Writes to:   normalized_answers/

Usage:
    python3 scripts/normalize_answers.py answers
    python3 scripts/normalize_answers.py answers --file JD_01_DataScientist
=============================================================
"""

import json
import re
import argparse
from pathlib import Path
from copy import deepcopy
from rapidfuzz import fuzz

OUTPUT_DIR = Path("normalized_answers")

TOOL_EXPANSIONS = {
    "Excel":      "Microsoft Excel",
    "Word":       "Microsoft Word",
    "PowerPoint": "Microsoft PowerPoint",
    "Outlook":    "Microsoft Outlook",
    "Teams":      "Microsoft Teams",
    "Visio":      "Microsoft Visio",
}

KNOWLEDGE_OVERRIDE = [
    "fundamentals", "theory", "principles", "architecture",
    "engineering", "methodology", "intelligence", "processing",
    "augmented", "workflows", "generation", "modelling",
    "modeling", "science", "management", "analysis",
]

INVALID_IPS_TOOLS = [
    "artificial intelligence", "machine learning",
    "deep learning", "data science", "natural language processing",
]

# Artefact names the model keeps producing despite the prompt forbidding them,
# usually dressed up with an action ("Dockerfile Creation"). A file or a config is
# never the skill; the product behind it is. Checked in order, first match wins.
ARTEFACT_REWRITES = [
    (re.compile(r"docker[\s-]*compose", re.I),      "Docker"),
    (re.compile(r"\bdockerfile\b", re.I),           "Docker"),
    (re.compile(r"\brequirements\.txt\b", re.I),    "Python"),
    (re.compile(r"\bsetup\.py\b|\bpyproject\.toml\b", re.I), "Python"),
]

# anything else that is plainly a file name has no portable skill behind it
FILENAME_PATTERN = re.compile(r"\b[\w-]+\.(yml|yaml|py|txt|json|toml|cfg|ini|sh|md|lock)\b", re.I)

# suffixes that add no skill of their own, used to collapse two spellings of one
# skill; deliberately narrow, since "Development" or "Engineering" do carry meaning
EMPTY_SUFFIXES = {
    "implementation", "application", "usage", "utilisation",
    "utilization", "creation", "configuration",
}


def expand_tool_names(skill: dict) -> None:
    for field in ("noun", "verb", "esco_form"):
        value = skill.get(field)
        if not isinstance(value, str):
            continue
        for short, full in TOOL_EXPANSIONS.items():
            if full not in skill[field]:
                skill[field] = re.sub(rf'\b{re.escape(short)}\b', full, skill[field])


def clean_esco_form(skill: dict) -> None:
    esco = skill.get("esco_form", "")
    if not isinstance(esco, str):
        return
    esco = re.sub(r'\b(Utilize|Utilise)\b', 'use', esco, flags=re.IGNORECASE)
    esco = re.sub(r'\s+for\s+.+$', '', esco, flags=re.IGNORECASE)
    esco = re.sub(r'\s+(effectively|efficiently|accurately|properly|correctly|successfully)$', '', esco, flags=re.IGNORECASE)
    skill["esco_form"] = esco.strip()


def clean_activities(activities):
    """Put the certificate bullets into the shape the certificate needs.

    They are rendered as a list under a lead-in sentence, so a trailing full stop
    and a leading bullet character have to go. The wording itself is left alone:
    an activity is prose about one specific project and has no canonical form to
    normalise it towards.
    """
    if not isinstance(activities, list):
        return []

    cleaned = []
    for item in activities:
        text = item.get("activity", "") if isinstance(item, dict) else item
        if not isinstance(text, str):
            continue
        text = re.sub(r'^[\s•\-–—*]+', '', text).strip()
        text = re.sub(r'\.\s*$', '', text)
        if not text:
            continue
        if isinstance(item, dict):
            entry = dict(item)
            entry["activity"] = text
        else:
            entry = {"activity": text, "source": ""}
        cleaned.append(entry)
    return cleaned


def rewrite_artefacts(skill: dict):
    """Replace an artefact name with the product behind it, or drop it.

    Returns a note when something changed, so the run reports it rather than
    silently editing the extraction.
    """
    noun = str(skill.get("noun", "")).strip()
    if not noun:
        return None

    for pattern, product in ARTEFACT_REWRITES:
        if pattern.search(noun):
            if noun == product:
                return None
            skill["noun"]       = product
            skill["verb"]       = f"Use {product}"
            skill["esco_form"]  = f"use {product}"
            skill["category"]   = "K"
            skill["skill_type"] = "Tool"
            skill["tool"]       = None
            skill["action"]     = None
            return f"{noun!r} → {product!r} (artefact name)"

    if FILENAME_PATTERN.search(noun):
        skill["_drop"] = True
        return f"{noun!r} dropped (file name, no portable skill)"

    return None


def collapse_empty_suffix(skills: list):
    """Drop "Semantic Search Implementation" when "Semantic Search" is also present."""
    by_tokens = {}
    for skill in skills:
        by_tokens[tuple(str(skill.get("noun", "")).lower().split())] = skill

    notes = []
    for tokens, skill in list(by_tokens.items()):
        if len(tokens) < 2 or tokens[-1] not in EMPTY_SUFFIXES:
            continue
        shorter = tokens[:-1]
        if shorter in by_tokens:
            skill["_drop"] = True
            notes.append(f"{' '.join(tokens)!r} → {' '.join(shorter)!r} (same skill, empty suffix)")
    return notes


def build_esco_query(skill: dict) -> None:
    """Derive the ESCO lookup string from the stable fields, not from free prose.

    esco_form is written fresh by the model on every call, so "use Python" one run
    and "use Python 3.11" the next moved the ESCO match and flipped the decision on
    the same skill. The noun and the enum fields are stable, and the prompt already
    defines a template per category, so the lookup string can be derived instead of
    quoted. esco_form is kept untouched for display and traceability.
    """
    noun     = str(skill.get("noun", "")).strip()
    category = str(skill.get("category", "")).strip()
    kind     = str(skill.get("skill_type", "")).strip()
    tool     = skill.get("tool")
    action   = skill.get("action")

    if category == "IPS" and tool and action:
        query = f"Apply {str(tool).strip()}-based {str(action).strip()}"
    elif category == "K" and kind == "Tool":
        query = f"use {noun}"
    elif category == "K":
        query = f"Apply {noun} techniques"
    else:
        # OS and SF are verb+object in ESCO, which cannot be derived from a noun
        # phrase; the noun alone is the stable half and reads as a search query
        query = noun

    skill["esco_query"] = query.strip()


def normalize_skill_type(skill: dict) -> str:
    noun     = str(skill.get("noun",     "")).strip().lower()
    category = str(skill.get("category", "")).strip()
    domain   = str(skill.get("domain",   "")).strip().lower()
    tool     = skill.get("tool")
    action   = skill.get("action")

    if "gap" not in skill:
        skill["gap"] = False

    # OS ama tool+action dolu → IPS'e çevir
    if category == "OS" and tool and action:
        skill["category"] = "IPS"
        category = "IPS"

    if category == "SF":
        return "Soft Skill"

    if category == "IPS":
        if tool and action:
            tool_lower = str(tool).lower()
            if tool_lower in INVALID_IPS_TOOLS:
                skill["category"] = "OS"
                return "Methodology"
            if " and " in tool_lower or " or " in tool_lower:
                skill["category"] = "OS"
                return "Methodology"
            return "Tool"
        skill["category"] = "OS"
        return "Methodology"

    if category == "K":
        if "cloud" in domain:
            return "Tool"
        if any(sig in noun for sig in KNOWLEDGE_OVERRIDE):
            return "Knowledge"
        tool_signals = [
            "programming", "database", "stack", "library",
            "platform", "cloud", "containerization", "version control",
            "monitoring", "tool", "engine", "sdk", "api"
        ]
        if any(sig in noun for sig in tool_signals):
            return "Tool"
        if skill.get("skill_type") == "Tool" and len(noun.split()) <= 3:
            return "Tool"
        return "Knowledge"

    if category == "OS":
        return "Methodology"

    return "Methodology"


def deduplicate_skills(skills: list) -> list:
    seen_nouns = []
    deduplicated = []
    for skill in skills:
        noun = skill.get("noun", "").strip().lower()
        is_duplicate = False
        for seen in seen_nouns:
            if fuzz.token_sort_ratio(noun, seen) >= 80:
                is_duplicate = True
                break
        if not is_duplicate:
            seen_nouns.append(noun)
            deduplicated.append(skill)
    return deduplicated


def normalize_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized = deepcopy(raw_data)
    notes = []
    for jd_key, jd_value in normalized.items():
        skills = jd_value.get("skills", [])
        for skill in skills:
            expand_tool_names(skill)
            note = rewrite_artefacts(skill)
            if note:
                notes.append(note)
            clean_esco_form(skill)
            skill["skill_type"] = normalize_skill_type(skill)

        notes += collapse_empty_suffix(skills)
        skills = [s for s in skills if not s.pop("_drop", False)]

        for skill in skills:
            # after skill_type, because the query template depends on it
            build_esco_query(skill)
        jd_value["skills"] = deduplicate_skills(skills)

        if "project_activities" in jd_value:
            jd_value["project_activities"] = clean_activities(jd_value["project_activities"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"✓ {input_path.name} → {output_path}")
    for note in notes:
        print(f"    rewrote {note}")


def get_latest_files(input_dir: Path, jd_filter: str = None):
    files = {}
    for f in sorted(input_dir.glob("*.json")):
        parts = f.stem.rsplit("_", 2)
        base_stem = parts[0]
        files[base_stem] = f

    if jd_filter:
        files = {k: v for k, v in files.items() if jd_filter in k}

    return list(files.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default="answers", help="Input directory (default: answers)")
    parser.add_argument("--file", default=None, help="JD stem to filter (e.g. JD_14_CyberSecurityManager)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    jd_filter = args.file

    json_files = get_latest_files(input_dir, jd_filter)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}" + (f" matching '{jd_filter}'" if jd_filter else ""))

    for input_path in json_files:
        stem = input_path.stem
        output_filename = f"{stem}.json"
        output_path = OUTPUT_DIR / output_filename
        normalize_file(input_path, output_path)

    print(f"\nDone! {len(json_files)} file(s) normalized.")


if __name__ == "__main__":
    main()
