"""
html_upload.py — BYDO simple FastAPI upload service
=============================================================
Replaces upload.py (Streamlit) with a minimal FastAPI + plain HTML page.
Same job as upload.py: accept a PDF/DOCX/TXT job description, clean it,
save it to jobs/, run the extract -> normalize -> rerank pipeline, and
return the resulting JSON.

Run locally (from the project root, where scripts/ and jobs/ live):
    uvicorn html_upload:app --reload --port 8000

Then open http://localhost:8000 in your browser.
=============================================================
"""

import io
import json
import re
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

app = FastAPI(title="BYDO Upload API")

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")


# ── Helpers (same logic as upload.py) ──────────────────────────────

def get_next_job_number():
    existing = list(JOBS_DIR.glob("JD_*.txt"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        m = re.match(r"JD_(\d+)_", f.name)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers) + 1 if numbers else 1


def to_pascal_case(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.strip().split()
    return ''.join(w.capitalize() for w in words)


def extract_text(filename: str, contents: bytes):
    """Detects file type by extension and extracts plain text."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber not installed")
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
        return "\n\n".join(pages)

    elif ext == ".docx":
        if python_docx is None:
            raise RuntimeError("python-docx not installed")
        doc = python_docx.Document(io.BytesIO(contents))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    elif ext == ".txt":
        return contents.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text):
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'https?://\S+', '', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def suggest_title(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()][:10]
    keywords = ['engineer', 'scientist', 'analyst', 'manager', 'designer',
                'developer', 'consultant', 'advisor', 'nurse', 'officer', 'intern']
    for line in lines:
        if len(line.split()) <= 6:
            for kw in keywords:
                if kw.lower() in line.lower():
                    return to_pascal_case(line)
    return to_pascal_case(lines[0]) if lines else "JobTitle"


def get_latest_result(jd_stem: str):
    """After the pipeline runs, find the newest results/ file for this JD."""
    matches = sorted(
        RESULTS_DIR.glob(f"{jd_stem}_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


# ── The one and only page: a plain upload form ─────────────────────

UPLOAD_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"><title>BYDO Upload</title>
<style>
  body { font-family: sans-serif; max-width: 480px; margin: 60px auto; }
  #file-input { display: none; }
  .file-button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: #f5f5f5;
    cursor: pointer;
  }
  .file-button svg { width: 16px; height: 16px; }
  #file-name { margin-left: 10px; color: #555; }
</style>
</head>
<body>
  <h2>BYDO &middot; Upload Job Description</h2>
  <p>Upload a PDF, DOCX, or TXT job description.</p>
  <form action="/upload" method="post" enctype="multipart/form-data">
    <p>
      <label for="file-input" class="file-button">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3">
          <path d="M2 10v3a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-3"/>
          <path d="M8 10V2M8 2L5 5M8 2l3 3"/>
        </svg>
        Choose file
      </label>
      <input type="file" id="file-input" name="file" accept=".pdf,.docx,.txt" required
             onchange="document.getElementById('file-name').textContent = this.files.length ? this.files[0].name : 'No file chosen';">
      <span id="file-name">No file chosen</span>
    </p>
    <p><input type="text" name="job_title" placeholder="Job title (optional)"></p>
    <p><button type="submit">Upload &amp; analyze</button></p>
  </form>
  <p style="color:#888; font-size: 13px;">
    This can take a minute or two — it calls the model and re-ranks against ESCO.
  </p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return UPLOAD_PAGE


@app.post("/upload")
async def upload(file: UploadFile = File(...), job_title: str = Form("")):
    contents = await file.read()

    try:
        raw_text = extract_text(file.filename, contents)
    except (RuntimeError, ValueError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "Could not extract any text from the file."})

    cleaned_text = clean_text(raw_text)
    title = to_pascal_case(job_title) if job_title.strip() else suggest_title(cleaned_text)

    job_num = get_next_job_number()
    fname = f"JD_{job_num:02d}_{title}.txt"
    jd_stem = fname.replace(".txt", "")

    save_path = JOBS_DIR / fname
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # ── Step 1: extract skills ──────────────────────────────────────
    extract_result = subprocess.run(
        ["python3", "scripts/extract_skills_api.py", "--file", fname],
        capture_output=True, text=True
    )
    if extract_result.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": "Skill extraction failed.",
            "details": extract_result.stderr[-2000:],
        })

    # ── Step 2: normalize ────────────────────────────────────────────
    normalize_result = subprocess.run(
        ["python3", "scripts/normalize_answers.py", "answers", "--file", jd_stem],
        capture_output=True, text=True
    )
    if normalize_result.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": "Normalization failed.",
            "details": normalize_result.stderr[-2000:],
        })

    # ── Step 3: rerank against ESCO ──────────────────────────────────
    rerank_result = subprocess.run(
        ["python3", "scripts/reranking_pipeline.py", "--file", jd_stem],
        capture_output=True, text=True
    )
    if rerank_result.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": "ESCO reranking failed.",
            "details": rerank_result.stderr[-2000:],
        })

    # ── Return the result JSON ───────────────────────────────────────
    result_path = get_latest_result(jd_stem)
    if result_path is None:
        return JSONResponse(status_code=500, content={"error": "Pipeline finished but no results file was found."})

    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    return JSONResponse(content=result_data)
