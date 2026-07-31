"""
=============================================================
DASHBOARD BUILDER — BYDO skill network simulation
=============================================================
Turns simulation/out/ into one self-contained HTML file:
no server, no CDN, no build step, so it can simply be opened
or sent to someone.

Run (after simulate_market.py and analyze_network.py):
    python3 simulation/build_dashboard.py
    open simulation/out/dashboard.html

The network layout is computed here in Python (Fruchterman-
Reingold) and baked into the file, so the page only has to
handle interaction.
=============================================================
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("simulation/out")
WIDTH, HEIGHT = 1000, 700

PALETTE = ["#1B4B73", "#0E7C74", "#C9502A", "#8A6D1F", "#5E4B8B", "#2E7D52", "#6B7B8B"]

# repulsion strength and pull to the canvas centre, tuned so weakly connected
# clusters stay in frame instead of drifting into a corner
K_FACTOR = 0.85
GRAVITY = 0.05


# ── input ──────────────────────────────────────────────────────────

def read_csv(name):
    with open(OUT_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load():
    metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    truth = json.loads((OUT_DIR / "truth.json").read_text(encoding="utf-8"))
    nodes = read_csv("skill_nodes.csv")
    edges = [r for r in read_csv("skill_edges.csv") if r["significant"] == "1"]
    return metrics, truth, nodes, edges


# ── layout ─────────────────────────────────────────────────────────

GOLDEN_ANGLE = 2.39996323


def components(names, edges):
    """Connected components, largest first."""
    adj = defaultdict(set)
    for a, b, _ in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, found = set(), []
    for start in names:
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            stack.extend(adj[node] - seen)
        found.append(sorted(comp))
    found.sort(key=len, reverse=True)
    return found


def layout_graph(names, edges, community, radii):
    """Main component fills the canvas; stray components go in a bottom strip.

    Laying them out together lets the strays drift off under repulsion and
    squeezes the interesting part of the graph into the middle of the frame.
    """
    comps = components(names, edges)
    main, strays = comps[0], comps[1:]
    strip = 0 if not strays else 96

    inside = set(main)
    pos = spring_layout(main, [e for e in edges if e[0] in inside and e[1] in inside], community)
    box = (0, 0, WIDTH, HEIGHT - strip)
    pos = resolve_overlaps(rescale(pos, radii, box), radii, box)

    slot = WIDTH / max(len(strays), 1)
    for idx, comp in enumerate(strays):
        cx = slot * (idx + 0.5)
        cy = HEIGHT - strip / 2 + 6
        ring = 15 + 4 * len(comp)
        for j, name in enumerate(comp):
            angle = 2 * math.pi * j / len(comp) - math.pi / 2
            pos[name] = [cx + ring * math.cos(angle), cy + ring * math.sin(angle)]

    return pos, (HEIGHT - strip if strays else None)


def seed_positions(names, community):
    """Communities on a ring, members on a sunflower disc inside their community.

    A random start collapses this graph (74 nodes, 493 edges) into a ball, so the
    force pass gets handed a layout that already separates the clusters.
    """
    groups = defaultdict(list)
    for n in names:
        groups[community[n]].append(n)
    order = sorted(groups, key=lambda c: (-len(groups[c]), c))

    cx, cy = WIDTH / 2, HEIGHT / 2
    ring = min(WIDTH, HEIGHT) * 0.33
    pos = {}
    for idx, comm in enumerate(order):
        angle = 2 * math.pi * idx / len(order) - math.pi / 2
        gx, gy = cx + ring * math.cos(angle) * 1.25, cy + ring * math.sin(angle)
        members = sorted(groups[comm])
        radius = 26 + 9 * math.sqrt(len(members))
        for j, name in enumerate(members):
            r = radius * math.sqrt((j + 0.5) / len(members))
            a = j * GOLDEN_ANGLE
            pos[name] = [gx + r * math.cos(a), gy + r * math.sin(a)]
    return pos


def spring_layout(names, edges, community, iterations=300,
                  k_factor=K_FACTOR, gravity=GRAVITY):
    """Fruchterman-Reingold, deterministic, with a decaying pull to the cluster."""
    pos = seed_positions(names, community)
    if len(names) < 2:
        return pos

    k = math.sqrt(WIDTH * HEIGHT / len(names)) * k_factor
    temperature = WIDTH / 14
    cx, cy = WIDTH / 2, HEIGHT / 2

    for step in range(iterations):
        cohesion = 0.05 * (1 - step / iterations)
        centres = community_centres(pos, community)
        disp = {n: [0.0, 0.0] for n in names}

        for i, a in enumerate(names):
            ax, ay = pos[a]
            for b in names[i + 1:]:
                dx, dy = ax - pos[b][0], ay - pos[b][1]
                d = math.hypot(dx, dy) or 0.01
                force = k * k / d
                ux, uy = dx / d, dy / d
                disp[a][0] += ux * force
                disp[a][1] += uy * force
                disp[b][0] -= ux * force
                disp[b][1] -= uy * force

        for a, b, weight in edges:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.hypot(dx, dy) or 0.01
            force = d * d / k * (0.35 + weight)
            ux, uy = dx / d, dy / d
            disp[a][0] -= ux * force
            disp[a][1] -= uy * force
            disp[b][0] += ux * force
            disp[b][1] += uy * force

        for n in names:
            hx, hy = centres[community[n]]
            disp[n][0] += (hx - pos[n][0]) * cohesion + (cx - pos[n][0]) * gravity
            disp[n][1] += (hy - pos[n][1]) * cohesion + (cy - pos[n][1]) * gravity
            dx, dy = disp[n]
            d = math.hypot(dx, dy) or 1.0
            move = min(d, temperature)
            pos[n][0] += dx / d * move
            pos[n][1] += dy / d * move

        temperature = max(temperature * 0.975, 0.6)

    return pos


def community_centres(pos, community):
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    for name, (x, y) in pos.items():
        acc = sums[community[name]]
        acc[0] += x
        acc[1] += y
        acc[2] += 1
    return {c: (acc[0] / acc[2], acc[1] / acc[2]) for c, acc in sums.items()}


def rescale(pos, radii, box, pad=8, max_stretch=1.35):
    """Fit the layout into the box, allowing a little stretch to fill the frame."""
    x0, y0, x1, y1 = box
    inset = pad + max(radii[n] for n in pos)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    spanx, spany = max(xs) - min(xs) or 1, max(ys) - min(ys) or 1

    sx = (x1 - x0 - 2 * inset) / spanx
    sy = (y1 - y0 - 2 * inset) / spany
    base = min(sx, sy)
    sx, sy = min(sx, base * max_stretch), min(sy, base * max_stretch)

    offx = x0 + (x1 - x0 - spanx * sx) / 2 - min(xs) * sx
    offy = y0 + (y1 - y0 - spany * sy) / 2 - min(ys) * sy
    for p in pos.values():
        p[0] = p[0] * sx + offx
        p[1] = p[1] * sy + offy
    return pos


def resolve_overlaps(pos, radii, box, iterations=260, gap=3.0):
    """Push circles apart until they no longer touch, staying inside the box."""
    x0, y0, x1, y1 = box
    names = sorted(pos)
    for _ in range(iterations):
        collisions = 0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
                d = math.hypot(dx, dy)
                need = radii[a] + radii[b] + gap
                if d >= need:
                    continue
                if d < 1e-6:
                    # coincident: separate along a direction fixed by the pair
                    angle = (hash((a, b)) % 360) * math.pi / 180
                    dx, dy, d = math.cos(angle), math.sin(angle), 1.0
                push = (need - d) / 2
                ux, uy = dx / d, dy / d
                pos[a][0] -= ux * push
                pos[a][1] -= uy * push
                pos[b][0] += ux * push
                pos[b][1] += uy * push
                collisions += 1
        for name in names:
            r = radii[name] + 2
            pos[name][0] = min(x1 - r, max(x0 + r, pos[name][0]))
            pos[name][1] = min(y1 - r, max(y0 + r, pos[name][1]))
        if not collisions:
            break
    return pos


# ── assembling the payload ─────────────────────────────────────────

def display_labels(raw_labels, node_rows):
    """Label propagation can name two communities after the same role; number them."""
    size = defaultdict(int)
    for r in node_rows:
        if r["community"] != "":
            size[r["community"]] += 1

    same_name = defaultdict(list)
    for cid, label in raw_labels.items():
        same_name[label].append(cid)

    out = {}
    for label, cids in same_name.items():
        for rank, cid in enumerate(sorted(cids, key=lambda c: (-size[c], c))):
            out[cid] = label if rank == 0 else f"{label} {rank + 1}"
    return out


def build_payload(metrics, truth, node_rows, edge_rows):
    labels = display_labels(metrics["projection"]["community_labels"], node_rows)
    in_graph = {r["skill"] for r in node_rows if int(r["degree"]) > 0}

    edges = [(r["a"], r["b"], float(r["ochiai"])) for r in edge_rows
             if r["a"] in in_graph and r["b"] in in_graph]

    names = sorted(in_graph)
    community = {r["skill"]: labels.get(r["community"], r["community_label"])
                 for r in node_rows}
    demand = {r["skill"]: float(r["demand_share"]) for r in node_rows}
    max_demand = max(demand.values()) or 1
    radii = {n: round(4.5 + 9.5 * math.sqrt(demand[n] / max_demand), 1) for n in names}

    pos, strip_top = layout_graph(names, edges, community, radii)
    pos = {n: [round(p[0], 1), round(p[1], 1)] for n, p in pos.items()}

    nodes = []
    for r in node_rows:
        name = r["skill"]
        if name not in pos:
            continue
        share = demand[name]
        nodes.append({
            "id": name,
            "x": pos[name][0], "y": pos[name][1],
            "r": radii[name],
            "c": int(r["community"]) if r["community"] != "" else -1,
            "cl": community[name],
            "demand": share,
            "docs": int(r["documents"]),
            "deg": int(r["degree"]),
            "btw": float(r["betweenness"]),
            "part": float(r["participation"]),
            "growth": float(r["growth"]) if r["growth"] else None,
            "mine": r["in_company"] == "1",
            "origin": r["origin"],
            "life": r["lifecycle"],
        })

    generic = sorted(
        ({"id": r["skill"], "demand": demand[r["skill"]], "mine": r["in_company"] == "1"}
         for r in node_rows if int(r["degree"]) == 0 and int(r["documents"]) > 0),
        key=lambda r: -r["demand"])

    company_only = sorted(
        ({"id": r["skill"], "origin": r["origin"]}
         for r in node_rows if int(r["documents"]) == 0 and r["in_company"] == "1"),
        key=lambda r: r["id"])

    return {
        "params": truth["params"],
        "nodes": nodes,
        "strip_top": strip_top,
        "edges": [{"a": a, "b": b, "w": round(w, 3)} for a, b, w in edges],
        "communities": [{"id": int(c), "label": labels[c],
                         "colour": PALETTE[int(c) % len(PALETTE)]}
                        for c in sorted(labels, key=int)],
        "role_labels": labels,
        "generic": generic,
        "company_only": company_only,
        "metrics": metrics,
    }


# ── HTML ───────────────────────────────────────────────────────────

def kpi_cards(m):
    c, b, p, v = m["company_vs_market"], m["bipartite"], m["projection"], m["validation"]
    cards = [
        ("Demand coverage", f"{c['demand_coverage']:.0%}",
         "share of all market skill mentions the company can already cover", True),
        ("Job ads", f"{b['ads']:,}", f"{b['skills_advertised']} distinct skills, {b['links']:,} links", False),
        ("Backbone edges", f"{p['significant_edges']}",
         f"{p['backbone_reduction']:.0%} of pairs dropped as chance", False),
        ("Communities", f"{p['communities']}", f"modularity {v['modularity']}", False),
        ("Gaps to train", f"{c['gaps_trainable']}",
         f"{c['gaps_needing_hire']} need hiring, {c['gaps_generic']} generic", False),
        ("Surplus", f"{c['surplus_count']}",
         f"{len(c['surplus_held_out'])} of them held out by review", False),
    ]
    html = []
    for title, value, note, hero in cards:
        html.append(f"""<div class="kpi{' hero' if hero else ''}">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-note">{note}</div>
    </div>""")
    return "\n    ".join(html)


def bar_rows(items, value_key, label_key="skill", fmt="{:.0%}", colour="var(--navy)"):
    if not items:
        return '<div class="empty">none</div>'
    top = max(float(i[value_key]) for i in items) or 1
    rows = []
    for i in items:
        val = float(i[value_key])
        rows.append(f"""<div class="bar-row">
        <div class="bar-label">{i[label_key]}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{val / top * 100:.1f}%;background:{colour}"></div></div>
        <div class="bar-value">{fmt.format(val)}</div>
      </div>""")
    return "\n      ".join(rows)


def spark(values, colour):
    if not values:
        return ""
    hi = max(values) or 1
    step = 58 / max(len(values) - 1, 1)
    pts = " ".join(f"{i * step:.1f},{18 - 16 * (v / hi):.1f}" for i, v in enumerate(values))
    return (f'<svg class="spark" viewBox="0 0 58 20" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.6"/></svg>')


def trend_rows(items, colour):
    if not items:
        return '<div class="empty">none</div>'
    rows = []
    for i in items:
        rows.append(f"""<div class="trend-row">
        <div class="trend-label">{i['skill']}</div>
        {spark(i['by_window'], colour)}
        <div class="trend-value">x{i['growth']}</div>
      </div>""")
    return "\n      ".join(rows)


def table(headers, rows, empty="none"):
    if not rows:
        return f'<div class="empty">{empty}</div>'
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="grid"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def render(payload):
    m = payload["metrics"]
    c, s, t, v, i = (m["company_vs_market"], m["structure"], m["trends"],
                     m["validation"], m["identity"])
    p = payload["params"]

    legend = "\n      ".join(
        f'<button class="legend-item" data-community="{com["id"]}">'
        f'<span class="swatch" style="background:{com["colour"]}"></span>{com["label"]}</button>'
        for com in payload["communities"])

    def recovered(fraction, flagged=None):
        found, planted = fraction.split("/")
        tail = f' <span class="muted">({flagged} flagged)</span>' if flagged else ""
        return f"{found} of {planted}{tail}"

    validation_rows = [
        ("Communities match the planted roles", f"{v['community_purity']:.0%}"),
        ("Planted substitutes recovered",
         recovered(v["substitutes_recovered"], v["substitute_pairs_flagged"])),
        ("Planted prerequisites recovered",
         recovered(v["prerequisites_recovered"], v["implications_flagged"])),
        ("Planted rising skills recovered", recovered(v["rising_recovered"])),
    ]

    html = TEMPLATE
    html = html.replace("{{PARAMS}}", f"{m['bipartite']['ads']:,} ads &middot; "
                                      f"{p['windows']} time windows &middot; seed {p['seed']}")
    html = html.replace("{{KPIS}}", kpi_cards(m))
    html = html.replace("{{LEGEND}}", legend)
    html = html.replace("{{VALIDATION}}", "\n      ".join(
        f'<div class="stat"><span>{k}</span><b>{val}</b></div>' for k, val in validation_rows))
    role_labels = payload["role_labels"]
    html = html.replace("{{ROLEFIT}}", bar_rows(
        [{"skill": role_labels.get(str(r["community"]), r["label"]), "coverage": r["coverage"]}
         for r in c["role_fit"]], "coverage", colour="var(--teal)"))
    html = html.replace("{{DEMAND}}", bar_rows(s["top_demand"], "share"))
    html = html.replace("{{RISING}}", trend_rows(t["top_rising"][:6], "#0E7C74"))
    html = html.replace("{{DECLINING}}", trend_rows(t["top_declining"][:6], "#C9502A"))
    html = html.replace("{{TRAIN}}", bar_rows(c["gaps_trainable_top"], "demand", colour="var(--coral)"))
    html = html.replace("{{HIRE}}", table(
        ["Skill", "Demand", "Hops"],
        [(g["skill"], f"{g['demand']:.0%}", g["hops"]) for g in c["gaps_hire_top"]],
        "none: every structural gap is adjacent to a skill already held"))
    html = html.replace("{{GENERIC}}", " ".join(
        f'<span class="chip">{g["id"]} <b>{g["demand"]:.0%}</b></span>' for g in payload["generic"]))
    html = html.replace("{{SUBS}}", table(
        ["Option A", "Option B", "Same context", "Together vs chance"],
        [(x["a"], x["b"], f"{x['context_similarity']:.2f}", f"{x['lift']:.2f}x")
         for x in s["top_substitutes"]]))
    html = html.replace("{{RULES}}", table(
        ["If the ad asks for", "it also asks for", "Confidence", "Reverse", "Base rate"],
        [(x["if"], x["then"], f"{x['confidence']:.0%}", f"{x['reverse']:.0%}",
          f"{x['base_rate']:.0%}") for x in s["top_implications"]]))
    html = html.replace("{{BRIDGES}}", table(
        ["Skill", "Betweenness", "Participation"],
        [(x["skill"], x["betweenness"], x["participation"]) for x in s["top_bridges"]]))
    html = html.replace("{{CORE}}", " ".join(f'<span class="chip own">{x}</span>' for x in c["core_top"]))
    html = html.replace("{{SURPLUS}}", " ".join(f'<span class="chip">{x}</span>' for x in c["surplus_real"]))
    html = html.replace("{{HELDOUT}}", " ".join(
        f'<span class="chip warn">{x["noun"]} &rarr; {x["esco"]} <b>{x["confidence"]}</b></span>'
        for x in i["suspect"]))
    html = html.replace("{{MERGES}}", " ".join(
        f'<span class="chip">{k} &larr; {", ".join(vals)}</span>' for k, vals in i["merges"].items()))
    html = html.replace("{{MAXCORE}}", f"{s['max_core']}-core, {len(s['max_core_members'])} skills: "
                        + ", ".join(s["max_core_members"]))
    html = html.replace("{{DATA}}", json.dumps({
        "nodes": payload["nodes"], "edges": payload["edges"],
        "communities": payload["communities"], "stripTop": payload["strip_top"],
    }, ensure_ascii=False))
    return html


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BYDO skill network dashboard</title>
<style>
  :root {
    --ink: #12263A; --navy: #1B4B73; --teal: #0E7C74; --coral: #C9502A;
    --line: #DDE5EC; --grey: #5A6B7B; --bg: #F4F7FA;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
  }
  .band {
    background: var(--ink); color: #fff; padding: 16px 28px;
    display: flex; align-items: baseline; justify-content: space-between;
  }
  .band .brand { font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; }
  .band .sub { color: #A9C0D4; font-size: 12px; letter-spacing: 0.05em; }
  .wrap { max-width: 1440px; margin: 0 auto; padding: 22px 28px 60px; }

  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.11em; margin: 0 0 12px; }
  h2 .hint { text-transform: none; letter-spacing: 0; font-weight: 400; color: var(--grey); }
  .card {
    background: #fff; border: 1px solid var(--line); border-radius: 10px;
    padding: 16px 18px; margin-bottom: 18px;
  }
  .cols { display: grid; gap: 18px; }
  .c-main { grid-template-columns: minmax(0, 2.1fr) minmax(0, 1fr); }
  .c-three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .c-two { grid-template-columns: repeat(2, minmax(0, 1fr)); }

  /* KPI row */
  .kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
  .kpi { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .kpi.hero { background: var(--navy); border-color: var(--navy); color: #fff; }
  .kpi-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em; color: var(--grey); }
  .kpi.hero .kpi-title, .kpi.hero .kpi-note { color: #C7DAEA; }
  .kpi-value { font-size: 27px; font-weight: 700; letter-spacing: -0.03em; margin: 4px 0 2px; }
  .kpi-note { font-size: 11px; color: var(--grey); line-height: 1.35; }

  /* network */
  .net-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }
  .legend-item {
    display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
    border: 1px solid var(--line); background: #fff; border-radius: 999px;
    padding: 3px 10px; font-size: 11.5px; font-family: inherit; color: var(--ink);
  }
  .legend-item.off { opacity: 0.35; }
  .swatch { width: 9px; height: 9px; border-radius: 50%; }
  .controls { margin-left: auto; display: flex; align-items: center; gap: 10px; }
  .controls input[type=search] {
    border: 1px solid var(--line); border-radius: 6px; padding: 5px 9px;
    font: inherit; font-size: 12.5px; width: 170px;
  }
  .controls label { font-size: 12px; color: var(--grey); display: inline-flex; align-items: center; gap: 5px; }
  svg#net { width: 100%; height: auto; display: block; background: #fff; border-radius: 6px; }
  svg#net line { stroke: #9FB4C7; }
  svg#net circle { cursor: pointer; stroke: #fff; stroke-width: 1.2; }
  svg#net circle.mine { stroke: var(--ink); stroke-width: 2.4; }
  svg#net text {
    font-size: 9.5px; fill: #2A3E52; pointer-events: none;
    paint-order: stroke; stroke: #fff; stroke-width: 2.6px; stroke-linejoin: round;
  }
  svg#net text.caption { font-size: 9px; fill: #93A5B5; stroke: none; }
  .dim { opacity: 0.12 !important; }
  .dim-text { opacity: 0.15 !important; }
  #tip {
    position: fixed; pointer-events: none; opacity: 0; transition: opacity 0.1s;
    background: var(--ink); color: #fff; border-radius: 8px; padding: 9px 11px;
    font-size: 12px; line-height: 1.45; max-width: 250px; z-index: 20;
  }
  #tip b { display: block; font-size: 13px; margin-bottom: 3px; }
  #tip span { color: #A9C0D4; }

  /* bars, sparklines, tables */
  .bar-row { display: grid; grid-template-columns: 1fr 90px 44px; gap: 8px; align-items: center; margin-bottom: 5px; font-size: 12.5px; }
  .bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { background: #EDF2F7; border-radius: 3px; height: 8px; }
  .bar-fill { height: 8px; border-radius: 3px; }
  .bar-value { text-align: right; font-variant-numeric: tabular-nums; color: var(--grey); }
  .trend-row { display: grid; grid-template-columns: 1fr 58px 44px; gap: 8px; align-items: center; margin-bottom: 5px; font-size: 12.5px; }
  .trend-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .trend-value { text-align: right; font-variant-numeric: tabular-nums; color: var(--grey); }
  .spark { height: 20px; width: 58px; }
  table.grid { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.grid th {
    text-align: left; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--grey); font-weight: 600; padding: 0 8px 6px 0; white-space: nowrap;
  }
  table.grid td { padding: 5px 8px 5px 0; border-top: 1px solid #EEF3F7; vertical-align: top; }
  .chip {
    display: inline-block; border: 1px solid var(--line); background: #FAFCFD;
    border-radius: 999px; padding: 3px 9px; font-size: 11.5px; margin: 0 3px 5px 0;
  }
  .chip b { font-weight: 600; }
  .chip.own { border-color: var(--navy); color: var(--navy); }
  .chip.warn { border-color: var(--coral); color: var(--coral); }
  .stat { display: flex; justify-content: space-between; gap: 10px; font-size: 12.5px; padding: 5px 0; border-top: 1px solid #EEF3F7; }
  .stat:first-child { border-top: 0; }
  .empty { color: var(--grey); font-size: 12.5px; font-style: italic; }
  .note { color: var(--grey); font-size: 12px; margin: 10px 0 0; }
  .muted { color: var(--grey); font-weight: 400; }
  .foot {
    background: var(--ink); color: #A9C0D4; padding: 13px 28px;
    font-size: 12px; display: flex; justify-content: space-between; gap: 16px;
  }
  .foot code { color: #fff; font-size: 11.5px; }
</style>
</head>
<body>

<div class="band">
  <div>
    <span class="brand">BYDO</span>
    <span style="margin-left:14px">Skill co-occurrence network</span>
  </div>
  <div class="sub">Simulated market &middot; {{PARAMS}}</div>
</div>

<div class="wrap">

  <div class="kpis">
    {{KPIS}}
  </div>

  <div class="cols c-main">
    <div class="card">
      <div class="net-head">
        <h2 style="margin:0">The market as a network
          <span class="hint">significant co-occurrence only, size = demand, ring = company holds it</span></h2>
        <div class="controls">
          <label><input type="checkbox" id="only-mine"> only company skills</label>
          <input type="search" id="find" placeholder="Find a skill">
        </div>
      </div>
      <div class="legend">
      {{LEGEND}}
      </div>
      <svg id="net" viewBox="0 0 1000 700"></svg>
      <p class="note">Click a node to isolate its neighbourhood, click again to release.</p>
    </div>

    <div>
      <div class="card">
        <h2>Does the method work? <span class="hint">against planted structure</span></h2>
        {{VALIDATION}}
      </div>
      <div class="card">
        <h2>Role fit <span class="hint">company coverage per cluster</span></h2>
        {{ROLEFIT}}
      </div>
      <div class="card">
        <h2>Most demanded</h2>
        {{DEMAND}}
      </div>
    </div>
  </div>

  <div class="cols c-three">
    <div class="card">
      <h2>Train first <span class="hint">gaps one hop from a skill already held</span></h2>
      {{TRAIN}}
      <h2 style="margin-top:16px">Hire for <span class="hint">two or more hops away</span></h2>
      {{HIRE}}
    </div>
    <div class="card">
      <h2>Rising</h2>
      {{RISING}}
      <h2 style="margin-top:16px">Declining</h2>
      {{DECLINING}}
    </div>
    <div class="card">
      <h2>Bridging skills <span class="hint">hold clusters together</span></h2>
      {{BRIDGES}}
    </div>
  </div>

  <div class="cols c-two">
    <div class="card">
      <h2>Substitutes <span class="hint">same context, rarely in the same ad</span></h2>
      {{SUBS}}
    </div>
    <div class="card">
      <h2>Implications <span class="hint">asymmetric conditional demand</span></h2>
      {{RULES}}
    </div>
  </div>

  <div class="cols c-two">
    <div class="card">
      <h2>Core of the market</h2>
      <p class="note" style="margin-top:0">{{MAXCORE}}</p>
      <h2 style="margin-top:14px">Company core <span class="hint">held and demanded</span></h2>
      {{CORE}}
    </div>
    <div class="card">
      <h2>Generic demand <span class="hint">no structure, expected everywhere</span></h2>
      {{GENERIC}}
      <h2 style="margin-top:14px">Surplus <span class="hint">held, nobody advertises it</span></h2>
      {{SURPLUS}}
    </div>
  </div>

  <div class="card">
    <h2>Identity resolution <span class="hint">what happened to the real extracted rows</span></h2>
    <div style="margin-bottom:8px">Merged surface forms: {{MERGES}}</div>
    <div>Held out after review: {{HELDOUT}}</div>
  </div>

</div>

<div class="foot">
  <div>Synthetic market seeded with the real extracted skills. Every number on this page
    comes from the co-occurrence graph, nothing is hand-written.</div>
  <div><code>python3 simulation/simulate_market.py &amp;&amp; python3 simulation/analyze_network.py
    &amp;&amp; python3 simulation/build_dashboard.py</code></div>
</div>

<div id="tip"></div>

<script>
const DATA = {{DATA}};

const svg = document.getElementById("net");
const tip = document.getElementById("tip");
const colours = {};
DATA.communities.forEach(c => colours[c.id] = c.colour);

const pos = {};
DATA.nodes.forEach(n => pos[n.id] = n);

const NS = "http://www.w3.org/2000/svg";
function el(name, attrs) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
}

if (DATA.stripTop) {
  svg.appendChild(el("line", {
    x1: 0, y1: DATA.stripTop, x2: 1000, y2: DATA.stripTop,
    stroke: "#C8D5E0", "stroke-dasharray": "4 4", "stroke-width": 1,
  }));
  const caption = el("text", { x: 6, y: DATA.stripTop + 14, class: "caption" });
  caption.textContent = "no significant link to the main network";
  svg.appendChild(caption);
}

// edges first so nodes sit on top
const edgeEls = DATA.edges.map(e => {
  const a = pos[e.a], b = pos[e.b];
  const line = el("line", {
    x1: a.x, y1: a.y, x2: b.x, y2: b.y,
    "stroke-width": (0.35 + e.w * 2.1).toFixed(2),
    "stroke-opacity": (0.14 + e.w * 0.4).toFixed(2),
  });
  svg.appendChild(line);
  return { el: line, a: e.a, b: e.b };
});

const nodeEls = {}, labelEls = {};
DATA.nodes.forEach(n => {
  const circle = el("circle", {
    cx: n.x, cy: n.y, r: n.r, fill: colours[n.c] || "#6B7B8B",
    class: n.mine ? "mine" : "",
  });
  circle.addEventListener("mousemove", ev => showTip(ev, n));
  circle.addEventListener("mouseleave", hideTip);
  circle.addEventListener("click", () => toggleFocus(n.id));
  svg.appendChild(circle);
  nodeEls[n.id] = circle;
});
// labels: biggest node first, keep one only if it collides with nothing
const placed = [];
const boxesOverlap = (a, b) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const hitsNode = box => DATA.nodes.some(m => {
  const nx = Math.max(box.x, Math.min(m.x, box.x + box.w));
  const ny = Math.max(box.y, Math.min(m.y, box.y + box.h));
  return Math.hypot(m.x - nx, m.y - ny) < m.r + 1;
});

[...DATA.nodes].sort((a, b) => b.r - a.r).forEach(n => {
  const text = el("text", {});
  text.textContent = n.id.length > 24 ? n.id.slice(0, 23) + "\\u2026" : n.id;
  svg.appendChild(text);
  const spots = [
    [n.x, n.y - n.r - 4.5, "middle"],
    [n.x, n.y + n.r + 10, "middle"],
    [n.x + n.r + 4, n.y + 3.4, "start"],
    [n.x - n.r - 4, n.y + 3.4, "end"],
  ];
  const fits = spots.some(([x, y, anchor]) => {
    text.setAttribute("x", x);
    text.setAttribute("y", y);
    text.setAttribute("text-anchor", anchor);
    const b = text.getBBox();
    if (b.x < 0 || b.y < 0 || b.x + b.width > 1000 || b.y + b.height > 700) return false;
    const box = { x: b.x - 2, y: b.y - 1, w: b.width + 4, h: b.height + 2 };
    if (placed.some(p => boxesOverlap(p, box)) || hitsNode(box)) return false;
    placed.push(box);
    return true;
  });
  if (fits) labelEls[n.id] = text; else svg.removeChild(text);
});

const neighbours = {};
DATA.edges.forEach(e => {
  (neighbours[e.a] = neighbours[e.a] || new Set()).add(e.b);
  (neighbours[e.b] = neighbours[e.b] || new Set()).add(e.a);
});

function showTip(ev, n) {
  const rows = [
    `<b>${n.id}</b>`,
    `<span>demand</span> ${(n.demand * 100).toFixed(0)}% of ads (${n.docs})`,
    `<span>cluster</span> ${n.cl}`,
    `<span>degree</span> ${n.deg} &middot; <span>betweenness</span> ${n.btw.toFixed(3)}`,
    `<span>participation</span> ${n.part.toFixed(2)} &middot; <span>${n.life}</span>`,
  ];
  if (n.growth !== null) rows.push(`<span>growth</span> x${n.growth}`);
  rows.push(n.mine ? "<span>company holds this skill</span>" : "<span>gap</span>");
  if (n.origin !== "generated") rows.push(`<span>source</span> ${n.origin}`);
  tip.innerHTML = rows.join("<br>");
  tip.style.opacity = 1;
  const box = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + box.width > window.innerWidth - 10) x = ev.clientX - box.width - 14;
  if (y + box.height > window.innerHeight - 10) y = ev.clientY - box.height - 14;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
function hideTip() { tip.style.opacity = 0; }

let focused = null;
const hidden = new Set();
const onlyMine = document.getElementById("only-mine");
const find = document.getElementById("find");

let query = "";

function visible(id) {
  const n = pos[id];
  if (hidden.has(n.c)) return false;
  if (onlyMine.checked && !n.mine) return false;
  if (query && !n.id.toLowerCase().includes(query)) return false;
  if (focused && id !== focused && !(neighbours[focused] || new Set()).has(id)) return false;
  return true;
}

function apply() {
  query = find.value.trim().toLowerCase();
  DATA.nodes.forEach(n => {
    const show = visible(n.id);
    nodeEls[n.id].classList.toggle("dim", !show);
    if (labelEls[n.id]) labelEls[n.id].classList.toggle("dim-text", !show);
  });
  edgeEls.forEach(e => {
    const show = visible(e.a) && visible(e.b) &&
      (!focused || e.a === focused || e.b === focused);
    e.el.classList.toggle("dim", !show);
  });
}

function toggleFocus(id) {
  focused = focused === id ? null : id;
  apply();
}

document.querySelectorAll(".legend-item").forEach(button => {
  button.addEventListener("click", () => {
    const c = parseInt(button.dataset.community, 10);
    if (hidden.has(c)) hidden.delete(c); else hidden.add(c);
    button.classList.toggle("off", hidden.has(c));
    apply();
  });
});
onlyMine.addEventListener("change", apply);
find.addEventListener("input", apply);
</script>

</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build the self-contained dashboard HTML.")
    ap.add_argument("--out", default=str(OUT_DIR / "dashboard.html"))
    args = ap.parse_args()

    metrics, truth, node_rows, edge_rows = load()
    payload = build_payload(metrics, truth, node_rows, edge_rows)
    html = render(payload)

    path = Path(args.out)
    path.write_text(html, encoding="utf-8")
    print(f"nodes drawn   : {len(payload['nodes'])}")
    print(f"edges drawn   : {len(payload['edges'])}")
    print(f"generic chips : {len(payload['generic'])}")
    print(f"written       : {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
