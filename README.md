# BYDO — Skill Extraction Pipeline

This pipeline extracts professional skills from job descriptions (JDs), normalizes them, and matches them against the ESCO taxonomy using a bi-encoder + cross-encoder reranking pipeline.

---

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- An OpenAI API key

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/dulgerseda/bydo_skill_extraction_deployment.git
cd bydo_skill_extraction_deployment
```

### 2. Provide your API key

The key is read from the environment, so it never has to be written into a file that also
holds settings. Either export it for the current shell:

```bash
export OPENAI_API_KEY=sk-...
```

or write it once into a `.env` file in the project root, which `docker compose` and the
scripts both pick up on every run:

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

`.env` is gitignored. An exported variable wins over the file, so you can override it for a
single run without editing anything.

> Keeping `"openai_api_key"` in `config.json` still works, for installations that already do.
> The run then says so on startup, and the key sits on disk next to your settings.

### 3. Create your config.json
Create a `config.json` file in the project root. It holds the settings, not the key:
```json
{
  "model": "gpt-4o-2024-11-20",
  "prompt": "prompt_v13.txt",
  "temperature": 0,
  "seed": 42,
  "json_mode": true,
  "strict_schema": false,
  "max_tokens": 8126
}
```
> ⚠️ Leave these settings alone: they are what makes the output reproducible.
> See [Reproducibility](#reproducibility) for what each one does.

### 4. Start the service
First time (builds the image — takes 5-10 minutes):
```bash
docker compose up --build
```

After first run (starts instantly):
```bash
docker compose up
```

### 5. Open in browser
```
http://localhost:8000
```
Upload a job description → get a JSON response.

### 6. Stop the service
```bash
docker compose down
```

---

## Useful commands

| Command | Description |
|---------|-------------|
| `docker compose up --build` | First run — builds image and starts |
| `docker compose up` | Start without rebuilding |
| `docker compose down` | Stop the service |
| `docker compose logs -f` | View live logs |
| `docker compose run --rm --entrypoint python3 bydo -m unittest discover -s tests` | Run the offline regression tests |

---

## How it works

1. Upload a PDF, DOCX, or TXT job description
2. Skills are extracted using GPT-4o, together with the project activities the document reports
3. Skills are normalized and matched against the ESCO taxonomy
4. The extracted skills are shown as a table, and saved as JSON in `results/`

Uploading a document that is already in `jobs/` re-analyzes it under the job id it already
has. See [One document, one job id](#one-document-one-job-id).

---

## Where the API key comes from

The key is a secret and the settings are not, so the two are kept apart. Three sources are
checked, first match wins, and every run prints where the key came from:

| Source | Set it with | Use it for |
|--------|-------------|------------|
| `OPENAI_API_KEY` in the environment | `export OPENAI_API_KEY=sk-...` | one shell, one run, CI |
| `.env` in the project root | `echo 'OPENAI_API_KEY=sk-...' > .env` | the normal case, Docker included |
| `openai_api_key` in `config.json` | editing the file | installations that already do this |

`.env` is loaded into the environment by the script itself, with no extra dependency, and
`docker compose` reads the same file when it substitutes `${OPENAI_API_KEY}`. A key from
`.env` is therefore reported as `API key from OPENAI_API_KEY`: the file is a way of setting
the variable, not a source of its own. An already exported variable is never overwritten by
the file, so a single run can be redirected to a different key without editing anything:

```bash
OPENAI_API_KEY=sk-other docker compose run --rm --entrypoint python3 bydo \
  scripts/extract_skills_api.py --file JD_01.txt --no-cache
```

If no source has a key, the run stops before the first API call and prints the three ways to
set one. The README placeholder `your-api-key-here` counts as no key, so a half-finished
setup fails with that message instead of a confusing authentication error.

---

## One document, one job id

`jobs/` is the corpus: the set of documents the pipeline knows about. Every upload used to
be numbered, which quietly turned it into a log of uploads instead — the same learning
journal sent in five times became `JD_04`, `JD_05`, `JD_06`, `JD_07` and `JD_10`, and every
count taken over `results/` then treated one document as five.

An upload is now matched against the corpus by the SHA-256 of its cleaned text. If that
document is already there, it is re-analyzed under the job id it already has and no second
entry appears; the page says so instead of pretending a new document arrived. The extraction
is a cache hit in that case, so re-analysing costs nothing.

Identity is the content, not the file name or the title given at upload: the same journal
uploaded as *Student* and as *Learning Journal* is one document. Change a word and it is
legitimately a different one, with a job id of its own.

> Documents that entered the corpus before this rule are still there in duplicate. Deleting
> the redundant `jobs/*.txt` and their `results/*.json` is safe — the remaining copy keeps
> the skills, and the extraction cache means nothing has to be paid for again.

---

## Reproducibility

Uploading the same document twice used to return different skills, because a language
model samples a fresh answer every time it is called. Four things now stand between the
document and that variability.

**A pinned model.** `model` names a dated snapshot, not the `gpt-4o` alias. OpenAI
repoints the alias to new versions without notice, which silently changes results.

**A fixed sampling setup.** `temperature: 0` and a fixed `seed` make the call as close to
deterministic as the API allows. The `system_fingerprint` of every call is written to
`tokens/token_usage.json`: when it changes, OpenAI changed the backend, and that is the
one remaining thing that can move the output on its own.

**A cache, keyed by content.** Every request is hashed over the document text, the model,
the prompt file, the temperature, the seed, the token limit and the `json_mode` and
`strict_schema` switches: everything that can legitimately move the answer. If that hash
has been seen before, the recorded answer is returned and no API call is made. This is what
turns "the same document gives the same skills" from a hope into a guarantee, and it also
means re-uploading a document is free and instant.

```bash
# what a document's cache entry is called
docker compose run --rm --entrypoint python3 bydo \
  scripts/extract_skills_api.py --file JD_01.txt --print-cache-key

# deliberately ignore the cache and pay for a fresh answer
docker compose run --rm --entrypoint python3 bydo \
  scripts/extract_skills_api.py --file JD_01.txt --no-cache
```

To invalidate a cached answer, delete the file under `cache/extractions/`. Editing the
prompt, the model or the seed changes the hash, so those never need clearing by hand.

**An ESCO lookup that does not depend on prose.** The ESCO query used to be built from
`esco_form`, which the model rewrites on every call, so `use Python` one run and
`use Python 3.11` the next moved the match and flipped the decision on an unchanged skill.
`normalize_answers.py` now derives an `esco_query` from the stable fields instead, using
the template the prompt already defines per category. `esco_form` is still reported, but
nothing downstream depends on its wording.

### What is guaranteed, and what is not

| | |
|---|---|
| Same document, cache warm | Byte-identical skills, no API call, no cost |
| Same document, cache bypassed | Roughly 70% of skills agree on a long unstructured document, higher on a job description |
| Same document, different prompt or model | Treated as a different question, new cache entry |

The residual disagreement on a bypassed cache is real and comes from the document, not the
plumbing: a 20-page learning journal has no agreed answer to "how many skills is this",
so the model draws the line in a slightly different place each time. A job description is
short and explicit and does not have this problem. Because the cache is keyed by content,
this variability is invisible in normal use.

### Regression tests

The guarantees above are pinned by an offline suite. No API key, no network:

```bash
docker compose run --rm --entrypoint python3 bydo -m unittest discover -s tests -v
```

---

## Viewing results

After an upload finishes, the page opens with the project activities the document reports,
as a certificate would list them, and hovering one shows the sentence it came from. A
document that reports no project work has no such block.

Below it the skills are shown as a table with the ESCO match, confidence and decision for
each one. Use the filter buttons (Accept / Review / Emerging / Gaps) and the search box to
narrow the list down.

Every past run stays available under **Results** ([http://localhost:8000/results](http://localhost:8000/results)),
newest first. The raw JSON is always one click away from any run.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Upload form |
| POST | `/upload` | Upload JD file, runs the pipeline, shows the result table |
| POST | `/upload?format=json` | Same, but returns the raw JSON (for scripts) |
| GET | `/results` | All past runs, newest first |
| GET | `/results/{filename}` | One run as a readable table |
| GET | `/results/{filename}/json` | That run's raw JSON |

---

## Output format

```json
{
  "JD_01_JobTitle": {
    "domain": "Data Science",
    "project_activities": [
      {
        "activity": "exploratory analysis of internal item and tariff data on Google Cloud",
        "source": "I ran exploratory SQL queries over the internal item and tariff tables."
      }
    ],
    "skills": [
      {
        "noun": "Python",                    
        "verb": "Use Python",                
        "esco_form": "use Python",           
        "category": "K",                     
        "skill_type": "Tool",                
        "domain": "Software Engineering",    
        "tool": null,                        
        "action": null,                      
        "source": "Proficiency in Python...",
        "gap": false,                        
        "esco_query": "use Python",          
        "esco_match": "Python (computer programming)",
        "match_type": "exactMatch",          
        "confidence": 0.95,                  
        "decision": "accept"                 
      }
    ]
  }
}
```

### Field descriptions

| Field | Description |
|-------|-------------|
| `noun` | Skill name extracted from the JD |
| `verb` | Action associated with the skill |
| `esco_form` | The model's ESCO-style phrasing, reported for reference |
| `esco_query` | Derived from the stable fields; this is what ESCO is searched with |
| `category` | `K` (Knowledge) · `OS` (Operational Skill) · `SF` (Soft Fluency) · `IPS` (Integrated Power-Skill) |
| `skill_type` | `Tool` · `Methodology` · `Knowledge` · `Soft Skill` |
| `domain` | Professional domain of the skill |
| `tool` | Tool name if category is IPS, otherwise null |
| `action` | Domain action if category is IPS, otherwise null |
| `source` | Original sentence from the JD |
| `gap` | `true` if no ESCO match found |
| `esco_match` | Matched ESCO skill label |
| `match_type` | `exactMatch` · `closeMatch` · `emerging` |
| `confidence` | Match confidence score (0–1) |
| `decision` | `accept` · `review` · `emerging` |

### Project activities

A skill is portable: it has to stay true at another employer, so the prompt strips every trace of
the project out of it. That is the right form for matching against ESCO and the wrong form for a
certificate, which has to say what the person actually did. `project_activities` is that second
reading of the same document, in the wording a certificate of participation uses:

> The activities within the Learning Box included in particular:
> - collection and web scraping of external datasets from customs websites
> - processing and merging of internal datasets
> - exploratory SQL analysis of internal item and tariff data on Google Cloud

| Field | Description |
|-------|-------------|
| `activity` | One bullet, as it appears on the certificate: noun phrase, lower case, no full stop |
| `source` | The sentence from the document that evidences it, as for a skill |

The list is aggregated over the whole document, so a learning journal that describes the same
workstream for eight weeks yields one bullet, not eight. It comes back from the same model call as
the skills and is carried through normalization and ESCO reranking untouched, because an activity
describes one project and has no counterpart in the taxonomy to be matched against.

A document that advertises a role rather than reporting work that was carried out — an ordinary job
description — returns an empty list.

---

## Notes

- Processing takes 1-2 minutes per JD, or seconds when the document is already cached
- The API key is never in the image and never in a tracked file: it is read from
  `OPENAI_API_KEY`, which `docker compose` takes from your shell or from `.env`
- `config.json` is never included in the Docker image — always provided externally
- Results are saved to the `results/` folder
- `cache/` holds recorded answers and can be deleted at any time; it only costs API calls to rebuild
- Only data folders are mounted, so after changing any code — `scripts/`, `prompts/`,
  `tests/`, `html_upload.py` — rebuild with `docker compose up --build`, otherwise the
  container keeps running the version baked into the image. This applies to the test command
  above too: without a rebuild it runs the suite as it was at build time
