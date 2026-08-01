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

### 2. Create your config.json
Create a `config.json` file in the project root:
```json
{
  "openai_api_key": "your-api-key-here",
  "model": "gpt-4o-2024-11-20",
  "prompt": "prompt_v13.txt",
  "temperature": 0,
  "seed": 42,
  "json_mode": true,
  "strict_schema": false,
  "max_tokens": 8126
}
```
> ⚠️ Only replace `your-api-key-here` with your OpenAI API key.
> Leave the other settings alone: they are what makes the output reproducible.
> See [Reproducibility](#reproducibility) for what each one does.

### 3. Start the service
First time (builds the image — takes 5-10 minutes):
```bash
docker compose up --build
```

After first run (starts instantly):
```bash
docker compose up
```

### 4. Open in browser
```
http://localhost:8000
```
Upload a job description → get a JSON response.

### 5. Stop the service
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
2. Skills are extracted using GPT-4o
3. Skills are normalized and matched against the ESCO taxonomy
4. The extracted skills are shown as a table, and saved as JSON in `results/`

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
the prompt file, the temperature, the seed and the token limit. If that hash has been seen
before, the recorded answer is returned and no API call is made. This is what turns
"the same document gives the same skills" from a hope into a guarantee, and it also means
re-uploading a document is free and instant.

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

After an upload finishes, the skills are shown as a table with the ESCO match,
confidence and decision for each one. Use the filter buttons (Accept / Review /
Emerging / Gaps) and the search box to narrow the list down.

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
| GET | `/results/{file}` | One run as a readable table |
| GET | `/results/{file}/json` | That run's raw JSON |

---

## Output format

```json
{
  "JD_01_JobTitle": {
    "domain": "Data Science",
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

---

## Notes

- Processing takes 1-2 minutes per JD, or seconds when the document is already cached
- `config.json` is never included in the Docker image — always provided externally
- Results are saved to the `results/` folder
- `cache/` holds recorded answers and can be deleted at any time; it only costs API calls to rebuild
- After changing anything under `scripts/` or `prompts/`, rebuild with `docker compose up --build`,
  otherwise the container keeps running the version baked into the image
