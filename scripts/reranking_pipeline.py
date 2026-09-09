"""
=============================================================
RERANKING PIPELINE — BYDO Project
=============================================================
Reads JSON files from normalized_answers/
Runs semantic search + cross-encoder reranking per skill
Adds: esco_match, match_type, confidence, decision, gap
Carries domain and project_activities through unchanged
Saves each result to results/<JD_name>_<TIMESTAMP>.json
=============================================================
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR        = Path("data/processed")
EMBEDDINGS_PATH = DATA_DIR / "esco_embeddings.npy"
LABELS_PATH     = DATA_DIR / "esco_labels.json"
ESCO_JSON       = DATA_DIR / "esco_skill_dictionary.json"
INPUT_DIR       = Path("normalized_answers")
OUTPUT_DIR      = Path("results")

BI_ENCODER_MODEL    = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(ESCO_JSON, "r", encoding="utf-8") as f:
        esco_data = json.load(f)
    return embeddings, labels, esco_data

def semantic_search(term, embeddings, labels, bi_encoder, top_k=10):
    query_vec = bi_encoder.encode([term])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), labels[i]) for i in top_indices]

def rerank(term, lookup, candidates, esco_data, cross_encoder):
    query = f"{term}. {lookup}"
    pairs = []
    for _, label in candidates:
        info = esco_data.get(label, {})
        description = info.get("description", "")
        pairs.append([query, f"{label}. {description}"])
    scores = cross_encoder.predict(pairs)
    reranked = sorted(
        zip(scores, [c[1] for c in candidates]),
        key=lambda x: x[0], reverse=True
    )
    return reranked

def normalize_score(score):
    """Cross-encoder raw score normalised 0-1 with sigmoid"""
    return round(1 / (1 + math.exp(-score * 0.5)), 4)

def make_match_type(confidence):
    if confidence >= 0.85:
        return "exactMatch"
    elif confidence >= 0.65:
        return "closeMatch"
    else:
        return "emerging"

def make_decision(confidence):
    if confidence >= 0.8:
        return "accept"
    elif confidence >= 0.5:
        return "review"
    else:
        return "emerging"

def read_job(job_data):
    """One JD entry as (skills, domain, project_activities).

    Old answer files hold a bare skills list where newer ones hold an object, and
    a job description yields no project activities at all, so both are tolerated.
    """
    if isinstance(job_data, list):
        return job_data, "Unknown", []
    return (job_data["skills"],
            job_data.get("domain", "Unknown"),
            job_data.get("project_activities", []))

def get_latest_files(input_dir: Path, jd_filter: str = None):
    latest_per_jd = {}
    
    # Sıralı okuma sayesinde en güncel tarihli olan dosya her zaman en son işlenir 
    # ve sözlükte eskisinin üzerine yazılır.
    for f in sorted(input_dir.glob("*.json")):
        base_stem = "_".join(f.stem.split("_")[:3])
        latest_per_jd[base_stem] = f

    if jd_filter:
        latest_per_jd = {k: v for k, v in latest_per_jd.items() if jd_filter in k}

    return list(latest_per_jd.values())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="JD stem to filter (e.g. JD_14_CyberSecurityManager)")
    args = parser.parse_args()

    print("Loading index and models...")
    embeddings, labels, esco_data = load_index()
    bi_encoder    = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"ESCO index loaded: {embeddings.shape[0]} records\n")

    json_files = get_latest_files(INPUT_DIR, args.file)
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {INPUT_DIR}" +
            (f" matching '{args.file}'" if args.file else "")
        )
    print(f"Found {len(json_files)} JD(s) to process.")

    for input_path in json_files:
        base_name = "_".join(input_path.stem.split("_")[:3])
        output_path = OUTPUT_DIR / f"{base_name}_{TIMESTAMP}.json"

        with open(input_path, "r", encoding="utf-8") as f:
            answers = json.load(f)

        output = {}

        for job_id, job_data in answers.items():
            skills, job_domain, job_activities = read_job(job_data)

            print(f"\n{'='*60}")
            print(f"  {job_id}  |  domain: {job_domain}  |  {len(skills)} skills"
                  f"  |  {len(job_activities)} activities")
            print(f"{'='*60}")

            enriched_skills = []

            for skill in skills:
                noun      = skill["noun"]
                # esco_query is derived deterministically by normalize_answers;
                # older files without it fall back to the model's own wording
                lookup    = skill.get("esco_query") or skill.get("esco_form", noun)

                print(f"  -> {noun}")

                candidates1 = semantic_search(noun, embeddings, labels, bi_encoder, top_k=10)
                candidates2 = semantic_search(lookup, embeddings, labels, bi_encoder, top_k=10)

                seen = set()
                candidates = []
                for score, label in candidates1 + candidates2:
                    if label not in seen:
                        seen.add(label)
                        candidates.append((score, label))
                reranked   = rerank(noun, lookup, candidates, esco_data, cross_encoder)

                top_score, top_label = reranked[0]
                confidence = normalize_score(top_score)
                match_type = make_match_type(confidence)
                decision   = make_decision(confidence)
                gap        = decision == "emerging"

                enriched = {
                    **skill,
                    "esco_match": top_label if not gap else None,
                    "match_type": match_type,
                    "confidence": confidence,
                    "decision":   decision,
                    "gap":        gap
                }
                enriched_skills.append(enriched)

                print(f"     confidence: {confidence}  match_type: {match_type}  decision: {decision}  gap: {gap}")

            output[job_id] = {
                "domain": job_domain,
                # carried through unchanged: an activity describes one project and
                # has no counterpart in ESCO to be matched against
                "project_activities": job_activities,
                "skills": enriched_skills
            }

        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nDone! -> {output_path}")

        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        for job_id, job_data in output.items():
            skills   = job_data["skills"]
            accept   = sum(1 for s in skills if s["decision"] == "accept")
            review   = sum(1 for s in skills if s["decision"] == "review")
            emerging = sum(1 for s in skills if s["decision"] == "emerging")
            exact    = sum(1 for s in skills if s["match_type"] == "exactMatch")
            close    = sum(1 for s in skills if s["match_type"] == "closeMatch")
            print(f"  {job_id[:35]:<35} accept:{accept}  review:{review}  emerging:{emerging}  |  exact:{exact}  close:{close}")
