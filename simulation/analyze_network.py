"""
=============================================================
SKILL NETWORK ANALYSIS — BYDO
=============================================================
Reads the bipartite graph written by simulate_market.py,
projects it onto a skill-skill co-occurrence network and
reports the metrics that carry the business insight.

Run (after simulate_market.py):
    python3 simulation/analyze_network.py

Writes to simulation/out/:
    skill_nodes.csv   per-skill metrics
    skill_edges.csv   per-pair metrics, significant edges flagged
    metrics.json      every headline number
    report.md         the same numbers as a readable report

Pure standard library on purpose: it runs anywhere the pipeline runs.
=============================================================
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path

OUT_DIR = Path("simulation/out")
MIN_COOC = 3          # pairs below this are not even tested
FDR_ALPHA = 0.01      # Benjamini-Hochberg level for the backbone
DEMAND_FLOOR = 0.05   # a skill counts as "market demand" above this ad share


# ── loading ────────────────────────────────────────────────────────

def read_csv(name):
    with open(OUT_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load():
    skills = {r["skill"]: r for r in read_csv("skills.csv")}
    market = read_csv("market_edges.csv")
    company = read_csv("company_edges.csv")
    truth = json.loads((OUT_DIR / "truth.json").read_text(encoding="utf-8"))

    ads = defaultdict(set)
    windows = {}
    for row in market:
        ads[row["ad_id"]].add(row["skill"])
        windows[row["ad_id"]] = int(row["window"])

    docs = defaultdict(set)
    for row in company:
        docs[row["doc_id"]].add(row["skill"])

    return skills, dict(ads), windows, dict(docs), truth


# ── bipartite structure ────────────────────────────────────────────

def gini(values):
    xs = sorted(values)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def bipartite_metrics(ads, df):
    sizes = sorted(len(s) for s in ads.values())
    links = sum(sizes)
    top10 = sorted(df.values(), reverse=True)[:10]
    return {
        "ads": len(ads),
        "skills_advertised": len(df),
        "links": links,
        "skills_per_ad_mean": round(links / max(len(ads), 1), 2),
        "skills_per_ad_median": sizes[len(sizes) // 2] if sizes else 0,
        "density": round(links / max(len(ads) * len(df), 1), 4),
        "demand_gini": round(gini(list(df.values())), 3),
        "top10_demand_share": round(sum(top10) / max(links, 1), 3),
        "singleton_skills": sum(1 for v in df.values() if v == 1),
    }


# ── projection and significance ────────────────────────────────────

def log_choose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k, N, K, n):
    """P(X >= k) for X ~ Hypergeometric(N ads, K with skill a, n with skill b)."""
    hi = min(K, n)
    if k > hi:
        return 0.0
    terms = []
    denom = log_choose(N, n)
    for i in range(k, hi + 1):
        lp = log_choose(K, i) + log_choose(N - K, n - i) - denom
        if lp > float("-inf"):
            terms.append(lp)
    if not terms:
        return 1.0
    m = max(terms)
    return min(1.0, math.exp(m + math.log(sum(math.exp(t - m) for t in terms))))


def project(ads, df):
    """Bipartite projection with the weights that make co-occurrence readable."""
    cooc = Counter()
    for skills in ads.values():
        for a, b in combinations(sorted(skills), 2):
            cooc[(a, b)] += 1

    N = len(ads)
    pairs = []
    for (a, b), k in cooc.items():
        if k < MIN_COOC:
            continue
        da, db = df[a], df[b]
        expected = da * db / N
        pa, pb, pab = da / N, db / N, k / N
        npmi = math.log(pab / (pa * pb)) / -math.log(pab) if 0 < pab < 1 else 0.0
        pairs.append({
            "a": a, "b": b, "cooc": k,
            "ochiai": k / math.sqrt(da * db),
            "lift": k / expected if expected else 0.0,
            "npmi": npmi,
            "conf_a_to_b": k / da,
            "conf_b_to_a": k / db,
            "p_value": hypergeom_sf(k, N, da, db),
        })

    # Benjamini-Hochberg over everything we tested
    pairs.sort(key=lambda r: r["p_value"])
    m = len(pairs)
    cutoff = 0.0
    for i, row in enumerate(pairs, start=1):
        if row["p_value"] <= i / m * FDR_ALPHA:
            cutoff = row["p_value"]
    for row in pairs:
        row["significant"] = row["p_value"] <= cutoff
    return cooc, pairs


def backbone(pairs):
    graph = defaultdict(dict)
    for row in pairs:
        if row["significant"]:
            graph[row["a"]][row["b"]] = row["ochiai"]
            graph[row["b"]][row["a"]] = row["ochiai"]
    return graph


# ── communities ────────────────────────────────────────────────────

def modularity(graph, labels):
    total = sum(w for a in graph for w in graph[a].values()) / 2
    if total == 0:
        return 0.0
    internal, strength = Counter(), Counter()
    for a in graph:
        for b, w in graph[a].items():
            strength[labels[a]] += w
            if labels[a] == labels[b]:
                internal[labels[a]] += w
    return sum(internal[c] / (2 * total) - (strength[c] / (2 * total)) ** 2
               for c in set(labels.values()))


def label_propagation(graph, seed, restarts=30):
    """Weighted label propagation, best partition by modularity."""
    nodes = sorted(graph)
    best, best_q = None, float("-inf")
    for r in range(restarts):
        rng = random.Random(seed + r)
        labels = {n: i for i, n in enumerate(nodes)}
        for _ in range(60):
            order = nodes[:]
            rng.shuffle(order)
            changed = False
            for n in order:
                if not graph[n]:
                    continue
                weights = Counter()
                for nb, w in graph[n].items():
                    weights[labels[nb]] += w
                top = max(weights.values())
                winners = sorted(c for c, w in weights.items() if w == top)
                pick = rng.choice(winners)
                if pick != labels[n]:
                    labels[n] = pick
                    changed = True
            if not changed:
                break
        q = modularity(graph, labels)
        if q > best_q:
            best, best_q = dict(labels), q

    # relabel communities by size, largest first
    order = [c for c, _ in Counter(best.values()).most_common()]
    remap = {c: i for i, c in enumerate(order)}
    return {n: remap[c] for n, c in best.items()}, best_q


# ── node level metrics ─────────────────────────────────────────────

def betweenness(graph):
    """Brandes, unweighted, on the backbone."""
    nodes = list(graph)
    bc = dict.fromkeys(nodes, 0.0)
    for s in nodes:
        stack, preds, sigma, dist = [], defaultdict(list), Counter({s: 1}), {s: 0}
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(stack, 0.0)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    n = len(nodes)
    scale = 2 / ((n - 1) * (n - 2)) if n > 2 else 1
    return {k: v * scale for k, v in bc.items()}


def k_core(graph):
    degree = {n: len(graph[n]) for n in graph}
    core = {}
    remaining = dict(degree)
    k = 0
    while remaining:
        k = max(k, min(remaining.values()))
        peeled = [n for n, d in remaining.items() if d <= k]
        while peeled:
            n = peeled.pop()
            if n not in remaining:
                continue
            core[n] = k
            del remaining[n]
            for nb in graph[n]:
                if nb in remaining:
                    remaining[nb] -= 1
                    if remaining[nb] <= k:
                        peeled.append(nb)
    return core


def participation(graph, communities):
    """Guimera-Amaral: 0 = all ties inside one community, near 1 = spread across many."""
    scores, zscores = {}, {}
    by_comm = defaultdict(list)
    for n in graph:
        by_comm[communities[n]].append(n)

    within = {}
    for n in graph:
        within[n] = sum(1 for nb in graph[n] if communities[nb] == communities[n])
    for comm, members in by_comm.items():
        vals = [within[n] for n in members]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1.0
        for n in members:
            zscores[n] = (within[n] - mean) / sd

    for n in graph:
        k = len(graph[n])
        if k == 0:
            scores[n] = 0.0
            continue
        counts = Counter(communities[nb] for nb in graph[n])
        scores[n] = 1 - sum((c / k) ** 2 for c in counts.values())
    return scores, zscores


def neighbour_cosine(cooc_vectors, a, b):
    """Structural equivalence: same context, ignoring each other."""
    va = {k: v for k, v in cooc_vectors[a].items() if k != b}
    vb = {k: v for k, v in cooc_vectors[b].items() if k != a}
    shared = set(va) & set(vb)
    if not shared:
        return 0.0
    dot = sum(va[k] * vb[k] for k in shared)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


# ── company against market ─────────────────────────────────────────

def hop_distance(graph, sources):
    dist = {s: 0 for s in sources if s in graph}
    queue = deque(dist)
    while queue:
        n = queue.popleft()
        for nb in graph[n]:
            if nb not in dist:
                dist[nb] = dist[n] + 1
                queue.append(nb)
    return dist


def company_metrics(skills, df, ads, docs, graph, communities, comm_labels, held_out):
    company = set().union(*docs.values()) if docs else set()
    advertised = set(df)
    N = len(ads)

    demanded = {s for s, c in df.items() if c / N >= DEMAND_FLOOR}
    core = sorted(company & demanded, key=lambda s: -df[s])
    gaps = sorted(demanded - company, key=lambda s: -df[s])
    surplus = sorted(company - advertised)
    surplus_held_out = [x for x in surplus if x in held_out]
    surplus_real = [x for x in surplus if x not in held_out]

    total_demand = sum(df.values())
    covered_demand = sum(df[s] for s in company & advertised)

    dist = hop_distance(graph, company & set(graph))
    trainable = [g for g in gaps if g in graph and dist.get(g, 99) <= 1]
    distant = [g for g in gaps if g in graph and dist.get(g, 99) >= 2]
    generic = [g for g in gaps if g not in graph]

    fit = []
    for comm, label in comm_labels.items():
        members = [n for n in graph if communities[n] == comm]
        demand = sum(df[n] for n in members)
        mine = sum(df[n] for n in members if n in company)
        fit.append({"community": comm, "label": label, "skills": len(members),
                    "coverage": round(mine / demand, 3) if demand else 0.0})
    fit.sort(key=lambda r: -r["coverage"])

    return {
        "company_documents": len(docs),
        "company_skills": len(company),
        "company_skills_advertised": len(company & advertised),
        "demand_coverage": round(covered_demand / total_demand, 3),
        "core_count": len(core),
        "gap_count": len(gaps),
        "surplus_count": len(surplus),
        "gaps_trainable": len(trainable),
        "gaps_needing_hire": len(distant),
        "gaps_generic": len(generic),
        "core_top": core[:10],
        "gaps_trainable_top": [{"skill": g, "demand": round(df[g] / N, 3),
                                "hops": dist.get(g)} for g in trainable[:10]],
        "gaps_hire_top": [{"skill": g, "demand": round(df[g] / N, 3),
                           "hops": dist.get(g)} for g in distant[:10]],
        "gaps_generic_top": [{"skill": g, "demand": round(df[g] / N, 3)} for g in generic[:6]],
        "surplus_real": surplus_real,
        "surplus_held_out": surplus_held_out,
        "role_fit": fit,
    }, company


# ── time ───────────────────────────────────────────────────────────

def trend_metrics(ads, windows, nwindows):
    per_window = defaultdict(Counter)
    ads_per_window = Counter()
    for ad_id, skills in ads.items():
        w = windows[ad_id]
        ads_per_window[w] += 1
        for s in skills:
            per_window[s][w] += 1

    trends = {}
    first, last = 0, nwindows - 1
    for skill, counts in per_window.items():
        share_first = counts[first] / max(ads_per_window[first], 1)
        share_last = counts[last] / max(ads_per_window[last], 1)
        trends[skill] = {
            "share_first": round(share_first, 4),
            "share_last": round(share_last, 4),
            "growth": round((share_last + 0.005) / (share_first + 0.005), 2),
            "by_window": [counts[w] for w in range(nwindows)],
        }
    return trends, dict(ads_per_window)


# ── reporting ──────────────────────────────────────────────────────

def top(d, n, reverse=True):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)[:n]


def listing(items, empty="none"):
    return ", ".join(items) if items else empty


def main():
    ap = argparse.ArgumentParser(description="Project and measure the simulated skill network.")
    ap.add_argument("--top", type=int, default=8, help="rows per top-list in the report")
    args = ap.parse_args()

    skills, ads, windows, docs, truth = load()
    nwindows = truth["params"]["windows"]
    df = Counter()
    for s in ads.values():
        df.update(s)

    bip = bipartite_metrics(ads, df)
    cooc, pairs = project(ads, df)
    graph = backbone(pairs)
    communities, q = label_propagation(graph, truth["params"]["seed"])

    # planted role per community, and how pure that mapping is
    comm_members = defaultdict(list)
    for n, c in communities.items():
        comm_members[c].append(n)
    comm_labels, purity_hits, purity_total = {}, 0, 0
    for c, members in sorted(comm_members.items()):
        roles = Counter(skills[m]["primary_role"] for m in members
                        if skills[m]["primary_role"] not in ("cross_cutting", "company_only"))
        dominant, hits = (roles.most_common(1)[0] if roles else ("generic", 0))
        comm_labels[c] = truth["roles"].get(dominant, "Cross-cutting / generic")
        purity_hits += hits
        purity_total += sum(roles.values())

    bc = betweenness(graph)
    cores = k_core(graph)
    part, zscore = participation(graph, communities)

    # PPMI-weighted context vectors for structural equivalence
    vectors = defaultdict(dict)
    for row in pairs:
        ppmi = max(0.0, math.log(row["lift"])) if row["lift"] > 0 else 0.0
        if ppmi > 0:
            vectors[row["a"]][row["b"]] = ppmi
            vectors[row["b"]][row["a"]] = ppmi

    frequent = [s for s, c in df.items() if c >= 8]
    subs = []
    for a, b in combinations(sorted(frequent), 2):
        cos = neighbour_cosine(vectors, a, b)
        if cos < 0.5:
            continue
        k = cooc.get((a, b), cooc.get((b, a), 0))
        expected = df[a] * df[b] / len(ads)
        lift = k / expected if expected else 0.0
        if lift < 0.75:
            subs.append({"a": a, "b": b, "context_similarity": round(cos, 3),
                         "lift": round(lift, 2), "cooc": k})
    subs.sort(key=lambda r: (-r["context_similarity"], r["lift"]))

    rules = [r for r in pairs if r["significant"]]
    nads = len(ads)
    implications = []
    for r in rules:
        for src, dst, conf, back in ((r["a"], r["b"], r["conf_a_to_b"], r["conf_b_to_a"]),
                                     (r["b"], r["a"], r["conf_b_to_a"], r["conf_a_to_b"])):
            base = df[dst] / nads
            if (df[src] >= 20 and conf >= 0.8 and conf - back >= 0.2
                    and conf - base >= 0.25 and r["lift"] >= 1.3):
                implications.append({"if": src, "then": dst, "confidence": round(conf, 3),
                                     "reverse": round(back, 3), "base_rate": round(base, 3),
                                     "lift": round(r["lift"], 2), "support": r["cooc"]})
    implications.sort(key=lambda r: (-r["confidence"], -r["lift"]))
    per_target = Counter()
    display = []
    for rule in implications:          # keep the shown list varied, not one target repeated
        if per_target[rule["then"]] < 3:
            per_target[rule["then"]] += 1
            display.append(rule)

    trends, ads_per_window = trend_metrics(ads, windows, nwindows)
    risers = sorted(((s, t["growth"]) for s, t in trends.items() if df[s] >= 15),
                    key=lambda kv: -kv[1])
    decliners = sorted(((s, t["growth"]) for s, t in trends.items() if df[s] >= 15),
                       key=lambda kv: kv[1])

    held_out = {x["esco"] for x in truth["identity"]["suspect"]}
    comp, company = company_metrics(skills, df, ads, docs, graph, communities, comm_labels, held_out)

    # did the metrics recover what the generator planted?
    planted_subs = {tuple(sorted((a, b))) for group in truth["substitute_groups"]
                    for a, b in combinations(group, 2)}
    found_subs = {tuple(sorted((r["a"], r["b"]))) for r in subs}
    planted_rules = {(a, b) for a, b in truth["prerequisites"].items()}
    found_rules = {(r["if"], r["then"]) for r in implications}
    top_risers = {s for s, _ in risers[:8]}

    validation = {
        "community_purity": round(purity_hits / purity_total, 3) if purity_total else 0.0,
        "modularity": round(q, 3),
        "substitutes_recovered": f"{len(planted_subs & found_subs)}/{len(planted_subs)}",
        "substitute_pairs_flagged": len(subs),
        "prerequisites_recovered": f"{len(planted_rules & found_rules)}/{len(planted_rules)}",
        "implications_flagged": len(implications),
        "rising_recovered": f"{len(set(truth['rising']) & top_risers)}/"
                            f"{len([s for s in truth['rising'] if df.get(s, 0) >= 15])}",
    }

    metrics = {
        "identity": truth["identity"],
        "bipartite": bip,
        "projection": {
            "pairs_tested": len(pairs),
            "candidate_pairs_total": len(cooc),
            "significant_edges": sum(1 for r in pairs if r["significant"]),
            "backbone_reduction": round(1 - sum(1 for r in pairs if r["significant"]) / max(len(cooc), 1), 3),
            "nodes_in_backbone": len(graph),
            "mean_degree": round(sum(len(v) for v in graph.values()) / max(len(graph), 1), 2),
            "communities": len(comm_members),
            "community_labels": {str(c): comm_labels[c] for c in sorted(comm_labels)},
        },
        "validation": validation,
        "company_vs_market": comp,
        "trends": {
            "ads_per_window": ads_per_window,
            "top_rising": [{"skill": s, "growth": g, "by_window": trends[s]["by_window"]} for s, g in risers[:args.top]],
            "top_declining": [{"skill": s, "growth": g, "by_window": trends[s]["by_window"]} for s, g in decliners[:args.top]],
        },
        "structure": {
            "top_demand": [{"skill": s, "share": round(c / len(ads), 3)} for s, c in df.most_common(args.top)],
            "top_bridges": [{"skill": s, "betweenness": round(v, 4),
                             "participation": round(part[s], 3)} for s, v in top(bc, args.top)],
            "top_connectors": [{"skill": s, "participation": round(v, 3),
                                "within_module_z": round(zscore[s], 2)} for s, v in top(part, args.top)],
            "max_core": max(cores.values()) if cores else 0,
            "max_core_members": sorted(s for s, k in cores.items() if k == max(cores.values())) if cores else [],
            "top_substitutes": subs[:args.top],
            "top_implications": display[:args.top],
        },
    }

    write_nodes(skills, df, len(ads), graph, communities, comm_labels, bc, cores, part, zscore, trends, company)
    write_edges(pairs)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(metrics, args.top)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)


def write_nodes(skills, df, nads, graph, communities, comm_labels, bc, cores, part, zscore, trends, company):
    with open(OUT_DIR / "skill_nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["skill", "origin", "demand_share", "documents", "community", "community_label",
                    "degree", "betweenness", "core", "participation", "within_module_z",
                    "growth", "in_company", "planted_role", "lifecycle"])
        for skill in sorted(set(df) | set(company)):
            meta = skills.get(skill, {})
            comm = communities.get(skill)
            w.writerow([
                skill, meta.get("origin", "?"), round(df.get(skill, 0) / nads, 4), df.get(skill, 0),
                comm if comm is not None else "", comm_labels.get(comm, ""),
                len(graph.get(skill, {})), round(bc.get(skill, 0.0), 5), cores.get(skill, 0),
                round(part.get(skill, 0.0), 3), round(zscore.get(skill, 0.0), 2),
                trends.get(skill, {}).get("growth", ""), int(skill in company),
                meta.get("primary_role", ""), meta.get("lifecycle", ""),
            ])


def write_edges(pairs):
    with open(OUT_DIR / "skill_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "cooc", "ochiai", "lift", "npmi", "p_value",
                    "significant", "conf_a_to_b", "conf_b_to_a"])
        for r in sorted(pairs, key=lambda r: -r["ochiai"]):
            w.writerow([r["a"], r["b"], r["cooc"], round(r["ochiai"], 4), round(r["lift"], 3),
                        round(r["npmi"], 4), f"{r['p_value']:.3e}", int(r["significant"]),
                        round(r["conf_a_to_b"], 3), round(r["conf_b_to_a"], 3)])


def render_report(m, n):
    L = []
    add = L.append
    i, b, p, v, c, t, s = (m["identity"], m["bipartite"], m["projection"],
                           m["validation"], m["company_vs_market"], m["trends"], m["structure"])

    add("# Skill co-occurrence network: simulated market\n")

    add("## 1. Identity resolution (real extractions)")
    add(f"- {i['rows']} extracted rows collapsed into {i['nodes']} skill nodes")
    add(f"- {len(i['merges'])} nodes absorbed more than one surface form: "
        + "; ".join(f"{k} <- {', '.join(vals)}" for k, vals in list(i["merges"].items())[:4]))
    add(f"- {len(i['suspect'])} suspicious ESCO match(es) held out of the market: "
        + listing([f"\"{x['noun']}\" -> \"{x['esco']}\" ({x['confidence']})" for x in i["suspect"]]))
    add(f"- {len(i['company_only'])} skills are company-only, never advertised\n")

    add("## 2. Bipartite graph")
    add(f"- {b['ads']} ads x {b['skills_advertised']} skills, {b['links']} links, density {b['density']}")
    add(f"- skills per ad: mean {b['skills_per_ad_mean']}, median {b['skills_per_ad_median']}")
    add(f"- demand concentration: Gini {b['demand_gini']}, top 10 skills hold "
        f"{b['top10_demand_share']:.0%} of all links")
    add("- most demanded: " + ", ".join(f"{x['skill']} ({x['share']:.0%})" for x in s["top_demand"]) + "\n")

    add("## 3. Projection to a skill-skill network")
    add(f"- {p['candidate_pairs_total']} co-occurring pairs, {p['pairs_tested']} tested at cooc >= {MIN_COOC}")
    add(f"- {p['significant_edges']} survive hypergeometric testing at FDR {FDR_ALPHA} "
        f"({p['backbone_reduction']:.0%} of pairs discarded as chance)")
    add(f"- backbone: {p['nodes_in_backbone']} nodes, mean degree {p['mean_degree']}, "
        f"{p['communities']} communities, modularity {v['modularity']}")
    add("- communities found: " + ", ".join(f"{k}={lab}" for k, lab in p["community_labels"].items()) + "\n")

    add("## 4. Does the method recover what was planted?")
    add(f"- community purity against the planted roles: {v['community_purity']:.0%}")
    add(f"- planted substitute pairs found: {v['substitutes_recovered']} "
        f"({v['substitute_pairs_flagged']} pairs flagged in total)")
    add(f"- planted prerequisites found: {v['prerequisites_recovered']} "
        f"({v['implications_flagged']} implications flagged in total)")
    add(f"- rising skills recovered in top 8: {v['rising_recovered']}\n")

    add("## 5. Structure insights")
    add("**Bridging skills** (high betweenness: they connect role clusters)")
    for x in s["top_bridges"][:n]:
        add(f"- {x['skill']}: betweenness {x['betweenness']}, participation {x['participation']}")
    add("\n**Connectors vs specialists** (participation across communities)")
    for x in s["top_connectors"][:n]:
        add(f"- {x['skill']}: participation {x['participation']}, within-module z {x['within_module_z']}")
    add(f"\n**Core of the market**: {s['max_core']}-core, {len(s['max_core_members'])} skills: "
        f"{', '.join(s['max_core_members'])}")
    add("\n**Substitutes** (same context, rarely together: competing choices)")
    if not s["top_substitutes"]:
        add("- none flagged")
    for x in s["top_substitutes"][:n]:
        add(f"- {x['a']} vs {x['b']}: context similarity {x['context_similarity']}, lift {x['lift']}")
    add("\n**Implications** (if the ad asks for A it also asks for B)")
    for x in s["top_implications"][:n]:
        add(f"- {x['if']} -> {x['then']}: {x['confidence']:.0%} of ads "
            f"(reverse {x['reverse']:.0%}, base rate {x['base_rate']:.0%}, support {x['support']})")
    add("")

    add("## 6. Demand over time")
    add("- ads per window: " + ", ".join(f"w{k}={val}" for k, val in sorted(t["ads_per_window"].items())))
    add("**Rising**")
    for x in t["top_rising"][:n]:
        add(f"- {x['skill']}: x{x['growth']} ({' -> '.join(str(c) for c in x['by_window'])})")
    add("**Declining**")
    for x in t["top_declining"][:n]:
        add(f"- {x['skill']}: x{x['growth']} ({' -> '.join(str(c) for c in x['by_window'])})")
    add("")

    add("## 7. The company against the market")
    add(f"- {c['company_documents']} real documents, {c['company_skills']} skills, "
        f"{c['company_skills_advertised']} of them advertised in the market")
    add(f"- **demand coverage: {c['demand_coverage']:.0%}** of all market skill mentions")
    add(f"- Core {c['core_count']}, Gap {c['gap_count']}, Surplus {c['surplus_count']} "
        f"(demand floor {DEMAND_FLOOR:.0%} of ads)")
    add(f"- gaps split by distance in the network: {c['gaps_trainable']} adjacent to a skill you "
        f"already have (train), {c['gaps_needing_hire']} two or more hops away (hire), "
        f"{c['gaps_generic']} generic skills outside the backbone (baseline expectations)")
    add("- train first: " + listing([f"{g['skill']} ({g['demand']:.0%})" for g in c["gaps_trainable_top"][:n]]))
    add("- hire for: " + listing([f"{g['skill']} ({g['demand']:.0%}, {g['hops']} hops)" for g in c["gaps_hire_top"][:n]],
                                 "none, every structural gap is adjacent to a skill you already have"))
    add("- generic: " + listing([f"{g['skill']} ({g['demand']:.0%})" for g in c["gaps_generic_top"]]))
    add(f"- surplus, you have it and nobody advertises it ({len(c['surplus_real'])}): "
        + listing(c["surplus_real"][:10]))
    add(f"- held out by review, so not real surplus ({len(c['surplus_held_out'])}): "
        + listing(c["surplus_held_out"]))
    add("- role fit by community: " + ", ".join(f"{r['label']} {r['coverage']:.0%}" for r in c["role_fit"]))
    return "\n".join(L)


if __name__ == "__main__":
    main()
