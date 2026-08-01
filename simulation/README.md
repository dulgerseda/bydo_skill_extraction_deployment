# Skill co-occurrence network: simulation

A demo of the "your skills against the market" idea while the real corpus is
still a handful of documents. The **company side is real** (every skill
extracted into `results/`, currently 85 rows from 6 documents), the **market
side is synthetic** (job ads sampled from latent roles). Nothing here touches
the extraction pipeline.

```bash
python3 simulation/simulate_market.py --ads 800 --seed 42
python3 simulation/analyze_network.py
python3 simulation/build_dashboard.py
open simulation/out/dashboard.html
```

All three scripts are standard library only, run in well under a second, and
write to `simulation/out/` (git-ignored).

| File | Contents |
|------|----------|
| `market_edges.csv` | bipartite graph: `ad_id, skill, window` |
| `company_edges.csv` | bipartite graph: `doc_id, skill` from the real results |
| `skills.csv` | every skill node with origin, lifecycle and planted role |
| `skill_nodes.csv` | per-skill metrics after analysis |
| `skill_edges.csv` | per-pair metrics, significant edges flagged |
| `metrics.json` | every headline number |
| `report.md` | the same numbers as a readable report |
| `dashboard.html` | the same numbers as one self-contained page |

## The dashboard

`build_dashboard.py` reads `metrics.json` and the two CSVs and writes a single
HTML file with no server, no CDN and no build step, so it can be opened from
disk or mailed to someone.

The network is the backbone graph: node size is demand, colour is the
discovered community, a dark ring means the company already holds that skill.
Hover a node for its metrics, click it to isolate its neighbourhood, click a
legend entry to hide a cluster, or search to pick out one skill. Skills with no
significant edges sit below the dashed line, because having no structure is
itself the finding: they are the generic expectations.

The layout is computed in Python (Fruchterman-Reingold seeded per community,
then a collision pass so no two circles touch) and baked into the file, so the
page only handles interaction and renders instantly. Labels are placed greedily
from the largest node down and dropped when they would collide, which is why
only some nodes are labelled.

## How the market is generated

Six latent roles (Data Engineer, ML Engineer, Data Scientist, Platform, BI,
AI Lead) each have core and optional skills. An ad picks a role, takes most of
its core skills, some optional ones, a slice of an adjacent role, and some
cross-cutting skills. On top of that the generator deliberately plants:

- **hub skills** (communication, Git, English) that appear in every role
- **substitute groups** where an ad picks exactly one option (PyTorch or
  TensorFlow or Keras, AWS or Azure or GCP, Power BI or Tableau)
- **prerequisites** where one skill drags another in (Kubernetes implies Docker)
- **rising and declining skills** whose probability changes across four windows

The point of planting structure is that section 4 of the report can check
whether the metrics actually find it. If they cannot recover known structure
from clean data, they will not find real structure in messy data.

## How real skills enter

Each extracted row is resolved to a node in one of four ways:

- **alias**: the same thing as a market skill, so it merges into that node
  (`Python 3.11` and `Python` become one node, `Dockerfile` and
  `Docker Deployment` fold into `Docker`). The extraction pipeline now strips
  version numbers and rewrites artefact names itself, so these aliases mostly
  cover rows extracted before those rules landed. They stay as a safety net,
  since a corpus always outlives the prompt that produced it.
- **esco**: has a solid ESCO match, becomes a market skill in the role its
  domain implies
- **suspect**: ESCO matched something much more specific than the extracted
  noun, so it is held out of the market. The rule is label coverage below 0.5,
  which catches `Torch` being matched to *operate plasma cutting torch* at 0.92
  confidence. This is what a human review queue would catch.
- **raw**: no ESCO match and no alias, so it is a company-only skill

## The metrics, and what each one is for

### Bipartite graph

- **density**, **skills per ad**: basic shape of the corpus.
- **demand Gini** and **top-10 share**: how concentrated demand is. A high Gini
  means a few skills appear everywhere and carry no information.

### Projection

Every pair of skills in the same ad gets an edge. Raw counts only rank popular
skills, so each pair also gets:

- **Ochiai cosine**: co-occurrence over the square root of the two document
  frequencies. Stable on small corpora, used as the edge weight.
- **lift**: observed over expected co-occurrence. Below 1 means they avoid
  each other.
- **NPMI**: normalised pointwise mutual information, for surprising pairs.
- **hypergeometric p-value**: probability of seeing this much co-occurrence by
  chance given the two skills' frequencies, with Benjamini-Hochberg control at
  FDR 1%. Only the surviving pairs form the **backbone**, which is the graph all
  later metrics use. About three quarters of pairs are discarded, and the ones
  that go are mostly hub skills, exactly as intended.

### Structure

- **communities** (weighted label propagation, best of 30 restarts by
  modularity): the roles the market actually advertises, discovered rather than
  assumed. **Modularity** around 0.6 means the clusters are real.
- **community purity**: share of skills whose discovered community matches the
  role the generator planted. This is the honest score for "did it work".
- **betweenness**: skills that sit on the paths between clusters. These are the
  hybrid profiles that are hard to hire and expensive to lose.
- **participation coefficient** and **within-module z-score** (Guimera and
  Amaral): participation near 0 is a specialist whose ties stay inside one
  role, near 0.5 is a connector spread across many. Combined with the z-score
  it separates provincial hubs from true connectors.
- **k-core**: the densest mutually-demanded block, the part of the market you
  cannot enter with a partial skill set.
- **substitutes**: pairs with high context similarity (PPMI-weighted cosine
  over their neighbourhoods, ignoring each other) but lift below 0.75. Same
  contexts, never together, which is what competing tools look like. PPMI
  weighting matters here: with plain co-occurrence weights, hub skills look like
  substitutes for everything.
- **implications**: asymmetric conditional probability. `dbt -> SQL` at 98% with
  a 77% reverse means dbt implies SQL but not the other way round. Filtered so
  the target is not simply ubiquitous (confidence must beat the target's base
  rate by 25 points and lift must exceed 1.3).

### Time

Document frequency per window, and growth as the ratio of last to first window
share. The planted rising skills should come out on top. Note that in the real
pipeline the timestamps in `results/` are extraction times, not posting dates,
so real trend analysis needs the ad's publication date captured as metadata.

### The company against the market

- **demand coverage**: share of all market skill mentions the company can
  already cover. The single headline number.
- **Core / Gap / Surplus** against a demand floor of 5% of ads.
- **gaps split by network distance**: a gap one hop from a skill you already
  have is a training target, two or more hops away is a hiring target, and a gap
  with no backbone edges at all is a generic baseline expectation rather than a
  strategic gap. This is the metric that turns a list of missing skills into a
  decision.
- **surplus** is split in two, because "nobody advertises this" and "a reviewer
  would have rejected this row" are very different findings. The second group is
  a data-quality signal about the extraction, not about the market.
- **role fit**: coverage per discovered community, which says which role the
  organisation is currently closest to.

## Reading the current output

With the real 85 extracted rows and 800 generated ads, the company covers 31%
of market demand, is closest to the ML Engineer / MLOps cluster (55%) and to
Data Scientist (41%), and furthest from AI Product / Lead (3%). Of 47 gaps, 42
sit one hop from a skill already held and none are two or more hops away, so it
is a training story rather than a hiring story. The clusters themselves hold up:
modularity 0.64, and 77% of skills land in the community the generator planted.

The surplus list is instructive in a different way. `FastAPI Upload Service` and
`Confidence Score Normalization` are not market skills at all, they are
over-extraction from a learning journal, and the second group of ten is held out
because ESCO matched something far more specific than the extracted noun. Both
groups say more about extraction quality than about the market.

These numbers move whenever `results/` changes. Re-run the three scripts above
to refresh them.
