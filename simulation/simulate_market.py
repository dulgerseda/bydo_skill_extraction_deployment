"""
=============================================================
MARKET SIMULATION — BYDO skill network
=============================================================
Builds a bipartite "job ad x skill" graph to demonstrate the
co-occurrence idea while the real corpus is still tiny.

  - company side : the REAL skills extracted into results/
  - market side  : synthetic job ads sampled from latent roles

The generator plants structure on purpose (roles, hub skills,
substitute groups, prerequisites, rising and declining skills)
so analyze_network.py can be checked against known truth.

Run:
    python3 simulation/simulate_market.py --ads 400 --seed 42

Writes to simulation/out/:
    skills.csv          one row per skill node
    market_edges.csv    ad_id, skill_id, window        (bipartite)
    company_edges.csv   doc_id, skill_id               (bipartite)
    truth.json          the planted structure, for validation
=============================================================
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR = Path("simulation/out")


# ── Latent roles: what the market advertises ───────────────────────

ROLES = {
    "data_engineer": {
        "label": "Data Engineer",
        "core": ["ETL pipeline development", "data modelling", "Apache Airflow", "dbt",
                 "SQL", "Python", "data quality monitoring", "Apache Spark", "data engineering"],
        "optional": ["Apache Kafka", "Snowflake", "Databricks", "data governance",
                     "Docker", "Hadoop", "on-premise ETL"],
    },
    "ml_engineer": {
        "label": "ML Engineer / MLOps",
        "core": ["machine learning", "model deployment", "MLflow", "Python",
                 "PyTorch", "model monitoring", "Docker", "Kubernetes"],
        "optional": ["TensorFlow", "Keras", "feature store", "CI/CD", "deep learning",
                     "experiment tracking", "LLM fine-tuning"],
    },
    "data_scientist": {
        "label": "Data Scientist",
        "core": ["statistical modelling", "hypothesis testing", "Python", "scikit-learn",
                 "pandas", "experiment design", "mathematical modelling"],
        "optional": ["R", "time series forecasting", "causal inference", "machine learning",
                     "deep learning", "data storytelling", "MATLAB"],
    },
    "platform_engineer": {
        "label": "Data Platform / Cloud",
        "core": ["Terraform", "Amazon Web Services", "Linux administration",
                 "infrastructure as code", "observability", "Docker", "Kubernetes"],
        "optional": ["Azure", "Google Cloud", "cloud cost optimisation", "networking",
                     "security hardening", "CI/CD", "C++"],
    },
    "analytics_engineer": {
        "label": "BI / Analytics Engineer",
        "core": ["SQL", "dbt", "Power BI", "data visualisation", "KPI definition",
                 "dashboard design", "stakeholder reporting"],
        "optional": ["Tableau", "Excel modelling", "data modelling", "data governance",
                     "Python", "data storytelling"],
    },
    "ai_lead": {
        "label": "AI Product / Lead",
        "core": ["team leadership", "stakeholder management", "roadmap planning",
                 "requirements engineering", "agile methods"],
        "optional": ["budget planning", "vendor management", "presentation skills",
                     "artificial intelligence", "data governance", "AI governance"],
    },
}

# Roles that borrow from each other: the source of bridge skills.
ROLE_NEIGHBOURS = {
    "data_engineer": ["platform_engineer", "analytics_engineer", "ml_engineer"],
    "ml_engineer": ["data_scientist", "data_engineer", "platform_engineer"],
    "data_scientist": ["ml_engineer", "analytics_engineer"],
    "platform_engineer": ["data_engineer", "ml_engineer"],
    "analytics_engineer": ["data_engineer", "data_scientist"],
    "ai_lead": ["data_scientist", "ml_engineer", "analytics_engineer"],
}

# Appear in every kind of ad: the hubs of the projected network.
CROSS_CUTTING = ["communication", "teamwork", "problem solving",
                 "agile methods", "Git", "English"]

# Only ever appear together with a senior ad.
SENIOR_SKILLS = ["team leadership", "stakeholder management", "mentoring"]

# Rising and declining demand, applied as a multiplier per time window.
LIFECYCLE = {
    "retrieval augmented generation": "rising",
    "vector databases": "rising",
    "LLM fine-tuning": "rising",
    "prompt engineering": "rising",
    "AI governance": "rising",
    "semantic search": "rising",
    "retrieval reranking": "rising",
    "Hadoop": "declining",
    "on-premise ETL": "declining",
    "Excel modelling": "declining",
    "MATLAB": "declining",
}
RAMP = {
    "rising":    [0.10, 0.45, 1.10, 2.00],
    "declining": [1.80, 1.10, 0.55, 0.20],
    "stable":    [1.00, 1.00, 1.00, 1.00],
}

# Competing choices: an ad picks one, so they share context but rarely co-occur.
SUBSTITUTE_GROUPS = [
    ["PyTorch", "TensorFlow", "Keras"],
    ["Amazon Web Services", "Azure", "Google Cloud"],
    ["Power BI", "Tableau"],
    ["Snowflake", "Databricks"],
    ["Python", "R"],
]

# "a implies b": if a is sampled, b is almost certainly there too.
PREREQUISITES = {
    "Kubernetes": "Docker",
    "dbt": "SQL",
    "retrieval augmented generation": "vector databases",
    "MLflow": "machine learning",
    "model deployment": "Docker",
    "LLM fine-tuning": "PyTorch",
}

# Emerging skills are attached to the roles that will adopt them first.
EMERGING_BY_ROLE = {
    "ml_engineer": ["retrieval augmented generation", "vector databases",
                    "LLM fine-tuning", "prompt engineering", "retrieval reranking"],
    "data_engineer": ["vector databases", "semantic search"],
    "data_scientist": ["prompt engineering", "semantic search"],
    "ai_lead": ["AI governance", "prompt engineering"],
}

# Real extracted skills that are the same thing as a market skill.
# Keys are lower-cased nouns or ESCO labels from results/.
ALIASES = {
    "python": "Python",
    "python 3.11": "Python",
    "python (computer programming)": "Python",
    "sql": "SQL",
    "sql databases": "SQL",
    "machine learning": "machine learning",
    "utilise machine learning": "machine learning",
    "deep learning": "deep learning",
    "artificial intelligence": "artificial intelligence",
    "principles of artificial intelligence": "artificial intelligence",
    "data engineering": "data engineering",
    "data visualization": "data visualisation",
    "data visualisation software": "data visualisation",
    "mathematical modeling": "mathematical modelling",
    "mathematical modelling": "mathematical modelling",
    "matlab": "MATLAB",
    "c++": "C++",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "leadership": "team leadership",
    "lead a team": "team leadership",
    "prompt engineering": "prompt engineering",
    "semantic search": "semantic search",
    "cross-encoder reranking": "retrieval reranking",
    "docker deployment": "Docker",
    "dockerfile": "Docker",
    "google cloud": "Google Cloud",
    "amazon web services": "Amazon Web Services",
}

# Where an ESCO-matched real skill lands if it is not in ALIASES.
DOMAIN_TO_ROLE = {
    "Data Engineering": "data_engineer",
    "Database Management": "data_engineer",
    "Software Engineering": "platform_engineer",
    "Cloud Computing": "platform_engineer",
    "Machine Learning": "ml_engineer",
    "Artificial Intelligence": "ml_engineer",
    "Mathematics": "data_scientist",
    "Physics": "data_scientist",
    "Systems Analysis": "data_scientist",
    "Soft Skills": "ai_lead",
}


# ── Reading the real results ───────────────────────────────────────

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def latest_result_per_job():
    """results/ accumulates one file per run: keep only the newest per JD."""
    newest = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        parts = path.stem.split("_")
        stem = "_".join(parts[:-2]) if len(parts) > 2 else path.stem
        if stem not in newest or path.stat().st_mtime > newest[stem].stat().st_mtime:
            newest[stem] = path
    return newest


STOPWORDS = {"a", "an", "and", "of", "on", "or", "the", "to", "in", "for", "with",
             "use", "using", "utilise", "utilize", "apply", "principles", "software"}


def tokens(text):
    found = re.findall(r"[a-z0-9+#]+", (text or "").lower())
    return {t for t in found if t not in STOPWORDS} or set(found)


def label_coverage(noun, esco):
    """
    How much of the matched ESCO label the extracted noun actually accounts for.
    "Torch" against "operate plasma cutting torch" scores 0.25: the label is a
    different, far more specific concept, which is what a reviewer would catch.
    """
    tn, te = tokens(noun), tokens(esco)
    return len(tn & te) / len(te) if te else 0.0


def load_real_skills():
    """
    One record per extracted row, plus the identity decision for it:
      alias    -> same node as a market skill
      esco     -> its own node, placed in a role via its domain
      suspect  -> ESCO matched something unrelated, keep out of the market
      raw      -> no ESCO match and no alias, company-only skill
    """
    rows = []
    for stem, path in latest_result_per_job().items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for job_id, job in data.items():
            skills = job["skills"] if isinstance(job, dict) else job
            for s in skills:
                noun = s.get("noun") or ""
                esco = s.get("esco_match")
                decision = s.get("decision")
                matched = esco and decision in ("accept", "review")

                alias = ALIASES.get(noun.lower()) or (ALIASES.get(esco.lower()) if esco else None)
                if alias:
                    kind, target = "alias", alias
                elif matched and label_coverage(noun, esco) < 0.5:
                    kind, target = "suspect", esco
                elif matched:
                    kind, target = "esco", esco
                else:
                    kind, target = "raw", noun

                rows.append({
                    "doc_id": job_id,
                    "noun": noun,
                    "esco": esco,
                    "decision": decision,
                    "confidence": s.get("confidence"),
                    "category": s.get("category"),
                    "skill_type": s.get("skill_type"),
                    "domain": s.get("domain"),
                    "kind": kind,
                    "target": target,
                })
    return rows


# ── Building the skill universe ────────────────────────────────────

def build_universe(real_rows):
    """
    Returns skills{name -> attrs} and role_pools{role -> {core, optional}},
    with real skills folded into the market where they belong.
    """
    skills = {}
    role_pools = {r: {"core": list(v["core"]), "optional": list(v["optional"])}
                  for r, v in ROLES.items()}

    def add(name, origin, roles, lifecycle=None, emerging=False):
        node = skills.setdefault(name, {
            "name": name, "origin": origin, "roles": set(),
            "lifecycle": lifecycle or LIFECYCLE.get(name, "stable"),
            "emerging": emerging, "in_market": True, "real_nouns": set(),
        })
        node["roles"].update(roles)
        if origin == "real" and node["origin"] == "generated":
            node["origin"] = "real+market"
        return node

    for role, pool in role_pools.items():
        for name in pool["core"] + pool["optional"]:
            add(name, "generated", [role])
    catalogue = set(skills) | set(CROSS_CUTTING) | set(SENIOR_SKILLS) | {
        n for names in EMERGING_BY_ROLE.values() for n in names}
    dangling = sorted(set(ALIASES.values()) - catalogue)
    if dangling:
        raise SystemExit(f"ALIASES point at skills no role advertises: {dangling}")
    for name in CROSS_CUTTING + SENIOR_SKILLS:
        add(name, "generated", list(ROLES))
    for role, names in EMERGING_BY_ROLE.items():
        for name in names:
            add(name, "generated", [role], emerging=True)
            if name not in role_pools[role]["optional"]:
                role_pools[role]["optional"].append(name)

    # fold the real rows in
    for row in real_rows:
        name, kind = row["target"], row["kind"]
        if kind == "alias":
            node = add(name, "real", [])
            node["real_nouns"].add(row["noun"])
        elif kind == "esco":
            role = DOMAIN_TO_ROLE.get(row["domain"], "data_scientist")
            node = add(name, "real", [role])
            node["real_nouns"].add(row["noun"])
            if name not in role_pools[role]["optional"]:
                role_pools[role]["optional"].append(name)
        else:  # suspect / raw: company-only, never advertised
            node = skills.setdefault(name, {
                "name": name, "origin": "real", "roles": set(),
                "lifecycle": "stable", "emerging": False,
                "in_market": False, "real_nouns": set(),
            })
            node["in_market"] = False
            node["real_nouns"].add(row["noun"])

    return skills, role_pools


# ── Sampling job ads ───────────────────────────────────────────────

def ramp(name, skills, window, windows):
    """Lifecycle multiplier, interpolated onto the requested window count."""
    curve = RAMP[skills[name]["lifecycle"]]
    idx = min(int(window / max(windows, 1) * len(curve)), len(curve) - 1)
    return curve[idx]


def sample_ad(rng, role, skills, role_pools, window, windows):
    chosen = set()
    pool = role_pools[role]

    def maybe(name, p):
        if name in skills and rng.random() < min(1.0, p * ramp(name, skills, window, windows)):
            chosen.add(name)

    for name in pool["core"]:
        maybe(name, 0.78)
    for name in pool["optional"]:
        maybe(name, 0.34)

    # a slice of an adjacent role: this is what makes bridges appear
    for neighbour in rng.sample(ROLE_NEIGHBOURS[role], k=min(2, len(ROLE_NEIGHBOURS[role]))):
        for name in role_pools[neighbour]["core"]:
            maybe(name, 0.10)

    for name in CROSS_CUTTING:
        maybe(name, 0.42)

    if rng.random() < 0.28:  # senior ad
        for name in SENIOR_SKILLS:
            maybe(name, 0.65)
    else:
        chosen -= set(SENIOR_SKILLS)

    # prerequisites come along for the ride
    for dependent, required in PREREQUISITES.items():
        if dependent in chosen and required in skills and rng.random() < 0.9:
            chosen.add(required)

    # competing tools last, so a prerequisite cannot smuggle in a second option
    for group in SUBSTITUTE_GROUPS:
        picked = [n for n in group if n in chosen]
        if len(picked) > 1:
            chosen -= set(picked)
            chosen.add(rng.choice(picked))

    return chosen


def simulate(ads, windows, seed):
    rng = random.Random(seed)
    real_rows = load_real_skills()
    skills, role_pools = build_universe(real_rows)

    market_edges = []
    ad_roles = {}
    role_names = list(ROLES)
    role_weights = [0.24, 0.22, 0.18, 0.14, 0.14, 0.08]

    for i in range(ads):
        role = rng.choices(role_names, weights=role_weights, k=1)[0]
        window = i * windows // ads
        picked = sample_ad(rng, role, skills, role_pools, window, windows)
        if len(picked) < 5:
            continue
        ad_id = f"AD{i:04d}"
        ad_roles[ad_id] = role
        for name in sorted(picked):
            market_edges.append((ad_id, name, window))

    company_edges = sorted({(r["doc_id"], r["target"]) for r in real_rows})
    return skills, real_rows, market_edges, company_edges, ad_roles


# ── Writing ────────────────────────────────────────────────────────

def primary_role(node):
    """Ground-truth label for community validation."""
    roles = node["roles"]
    if not roles:
        return "company_only"
    if len(roles) >= 4:
        return "cross_cutting"
    return sorted(roles)[0]


def write_outputs(skills, real_rows, market_edges, company_edges, ad_roles, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_by_target = defaultdict(list)
    for row in real_rows:
        real_by_target[row["target"]].append(row)

    with open(OUT_DIR / "skills.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["skill", "origin", "in_market", "lifecycle", "emerging",
                    "primary_role", "roles", "real_nouns", "company"])
        for name, node in sorted(skills.items()):
            w.writerow([name, node["origin"], int(node["in_market"]), node["lifecycle"],
                        int(node["emerging"]), primary_role(node),
                        "|".join(sorted(node["roles"])),
                        "|".join(sorted(node["real_nouns"])),
                        int(name in real_by_target)])

    with open(OUT_DIR / "market_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ad_id", "skill", "window"])
        w.writerows(market_edges)

    with open(OUT_DIR / "company_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "skill"])
        w.writerows(company_edges)

    truth = {
        "params": {"ads": args.ads, "windows": args.windows, "seed": args.seed},
        "roles": {r: v["label"] for r, v in ROLES.items()},
        "ad_roles": ad_roles,
        "substitute_groups": SUBSTITUTE_GROUPS,
        "prerequisites": PREREQUISITES,
        "rising": sorted(n for n, k in LIFECYCLE.items() if k == "rising"),
        "declining": sorted(n for n, k in LIFECYCLE.items() if k == "declining"),
        "cross_cutting": CROSS_CUTTING,
        "identity": {
            "rows": len(real_rows),
            "nodes": len({r["target"] for r in real_rows}),
            "merges": {t: sorted({r["noun"] for r in rows})
                       for t, rows in real_by_target.items()
                       if len({r["noun"] for r in rows}) > 1},
            "suspect": [{"noun": r["noun"], "esco": r["esco"], "confidence": r["confidence"]}
                        for r in real_rows if r["kind"] == "suspect"],
            "company_only": sorted({r["target"] for r in real_rows if r["kind"] in ("raw", "suspect")}),
        },
    }
    (OUT_DIR / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Simulate a job-ad market around the real extracted skills.")
    ap.add_argument("--ads", type=int, default=800, help="number of synthetic job ads")
    ap.add_argument("--windows", type=int, default=4, help="time windows the ads are spread over")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    skills, real_rows, market_edges, company_edges, ad_roles = simulate(args.ads, args.windows, args.seed)
    write_outputs(skills, real_rows, market_edges, company_edges, ad_roles, args)

    market_skills = {name for name, node in skills.items() if node["in_market"]}
    print(f"ads written      : {len(ad_roles)}")
    print(f"skill nodes      : {len(skills)}  ({len(market_skills)} advertised, "
          f"{len(skills) - len(market_skills)} company-only)")
    print(f"market links     : {len(market_edges)}")
    print(f"company links    : {len(company_edges)} from {len({d for d, _ in company_edges})} real documents")
    print(f"real rows folded : {len(real_rows)}")
    print(f"output           : {OUT_DIR}/")


if __name__ == "__main__":
    main()
