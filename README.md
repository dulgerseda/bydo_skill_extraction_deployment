# BYDO — Skill Extraction Pipeline

BYDO extracts professional skills from job descriptions (JDs), normalizes them, and matches them against the ESCO taxonomy using a bi-encoder + cross-encoder reranking pipeline.

---

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- An OpenAI API key

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/dulgerseda/bydo_skill_extraction.git
cd bydo_skill_extraction
```

### 2. Create your config.json
Create a `config.json` file in the project root:
```json
{
  "openai_api_key": "your-api-key-here",
  "model": "gpt-4o",
  "temperature": 0.1,
  "max_tokens": 8126
}
```
> ⚠️ Only replace `your-api-key-here` with your OpenAI API key.
> Do not change `model`, `temperature`, or `max_tokens`.

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

---

## How it works

1. Upload a PDF, DOCX, or TXT job description
2. Skills are extracted using GPT-4o
3. Skills are normalized and matched against the ESCO taxonomy
4. A structured JSON is returned

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Upload form |
| POST | `/upload` | Upload JD file, returns JSON |

---

## Output format

```json
{
  "JD_01_JobTitle": {
    "domain": "Data Science",
    "skills": [
      {
        "noun": "Python Programming",        
        "verb": "Program using Python",      
        "esco_form": "use Python",           
        "category": "K",                     
        "skill_type": "Tool",                
        "domain": "Software Engineering",    
        "tool": null,                        
        "action": null,                      
        "source": "Proficiency in Python...",
        "gap": false,                        
        "esco_match": "Python (computer programming)",
        "match_type": "exactMatch",          
        "confidence": 0.97,                  
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
| `esco_form` | Normalized form used for ESCO matching |
| `category` | `K` (Knowledge) · `OS` (Operational Skill) · `SF` (Soft Fluency) · `IPS` (Integrated Power-Skill) |
| `skill_type` | Tool · Method · Concept · Behavior |
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

- Processing takes 1-2 minutes per JD
- `config.json` is never included in the Docker image — always provided externally
- Results are saved to the `results/` folder
