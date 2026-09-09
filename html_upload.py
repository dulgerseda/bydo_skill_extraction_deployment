"""
html_upload.py — BYDO simple FastAPI upload service
=============================================================
Replaces upload.py (Streamlit) with a minimal FastAPI + plain HTML page.
Same job as upload.py: accept a PDF/DOCX/TXT job description, clean it,
save it to jobs/, run the extract -> normalize -> rerank pipeline, and
show the resulting skills.

Pages:
    GET  /                       upload form
    POST /upload                 run the pipeline, show the result table
                                 (add ?format=json for the raw JSON)
    GET  /results                every past run, newest first
    GET  /results/{file}         one run as a readable table
    GET  /results/{file}/json    that run's raw JSON

Run locally (from the project root, where scripts/ and jobs/ live):
    uvicorn html_upload:app --reload --port 8000

Then open http://localhost:8000 in your browser.
=============================================================
"""

import html as html_lib
import io
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
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


# ── Reading results/ ───────────────────────────────────────────────

def safe_result_path(filename: str):
    """Resolve a results/ filename, refusing anything outside that folder."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.json", filename):
        return None
    path = (RESULTS_DIR / filename).resolve()
    if path.parent != RESULTS_DIR.resolve() or not path.is_file():
        return None
    return path


def split_result_name(stem: str):
    """'JD_03_DataEngineering_20260731_130759' -> job id + readable timestamp."""
    parts = stem.split("_")
    if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[-2]) and re.fullmatch(r"\d{6}", parts[-1]):
        job_id = "_".join(parts[:-2])
        try:
            stamp = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
            return job_id, stamp.strftime("%d %b %Y, %H:%M")
        except ValueError:
            return job_id, f"{parts[-2]} {parts[-1]}"
    return stem, ""


def iter_jobs(data):
    """Yield (job_id, domain, skills, activities) — files hold one job, but loop anyway."""
    for job_id, job_data in data.items():
        if isinstance(job_data, list):
            yield job_id, "Unknown", job_data, []
        else:
            yield (job_id,
                   job_data.get("domain", "Unknown"),
                   job_data.get("skills", []),
                   job_data.get("project_activities", []))


def count_skills(skills):
    decisions = Counter(s.get("decision") for s in skills)
    matches = Counter(s.get("match_type") for s in skills)
    return {
        "total": len(skills),
        "accept": decisions.get("accept", 0),
        "review": decisions.get("review", 0),
        "emerging": decisions.get("emerging", 0),
        "gaps": sum(1 for s in skills if s.get("gap")),
        "exact": matches.get("exactMatch", 0),
        "close": matches.get("closeMatch", 0),
    }


def list_results():
    """Every results/ file, newest first, with a small summary of each."""
    if not RESULTS_DIR.is_dir():
        return []

    entries = []
    for path in sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        job_id, when = split_result_name(path.stem)
        entry = {"file": path.name, "job_id": job_id, "when": when,
                 "domain": "—", "broken": False}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            skills = []
            domains = []
            for _, domain, job_skills, _ in iter_jobs(data):
                skills.extend(job_skills)
                domains.append(domain)
            entry["domain"] = ", ".join(dict.fromkeys(domains)) or "—"
            entry["counts"] = count_skills(skills)
        except (json.JSONDecodeError, OSError, AttributeError):
            entry["broken"] = True
            entry["counts"] = count_skills([])
        entries.append(entry)
    return entries


# ── Rendering ──────────────────────────────────────────────────────

def esc(value):
    return html_lib.escape("" if value is None else str(value))


PAGE_SHELL = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root {
    --line: #e4e4e7;
    --muted: #6b7280;
    --bg: #fafafa;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #18181b; background: var(--bg);
    margin: 0; padding: 0 24px 80px;
    line-height: 1.5;
  }
  .wrap { max-width: {{WIDTH}}; margin: 0 auto; }
  nav {
    display: flex; align-items: center; gap: 20px;
    padding: 18px 0; margin-bottom: 28px;
    border-bottom: 1px solid var(--line);
  }
  nav .brand { font-weight: 650; letter-spacing: -0.01em; }
  nav a { color: var(--muted); text-decoration: none; font-size: 14px; }
  nav a:hover, nav a.active { color: #18181b; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 16px; margin: 32px 0 12px; }
  .sub { color: var(--muted); font-size: 14px; margin: 0 0 24px; }
  a.link { color: #1f6feb; }
  .card {
    background: #fff; border: 1px solid var(--line);
    border-radius: 10px; padding: 20px;
  }

  /* upload form */
  #file-input { display: none; }
  .file-button {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 16px; border: 1px solid #ccc; border-radius: 6px;
    background: #f5f5f5; cursor: pointer; font-size: 14px;
  }
  .file-button svg { width: 16px; height: 16px; }
  #file-name { margin-left: 10px; color: #555; font-size: 14px; }
  input[type=text] {
    padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px;
    font-size: 14px; width: 260px; font-family: inherit;
  }
  button[type=submit] {
    padding: 9px 18px; border: 0; border-radius: 6px;
    background: #18181b; color: #fff; font-size: 14px; cursor: pointer;
  }
  button[type=submit]:hover { background: #333; }

  /* summary chips */
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 20px; }
  .chip {
    border: 1px solid var(--line); background: #fff;
    border-radius: 999px; padding: 4px 12px; font-size: 13px;
  }
  .chip b { font-weight: 650; }

  /* project activities */
  .activities { margin: 0; padding-left: 20px; font-size: 14px; }
  .activities li { margin-bottom: 5px; }
  .activities li:last-child { margin-bottom: 0; }

  /* tables */
  table { width: 100%; border-collapse: collapse; background: #fff; font-size: 14px; }
  .tablewrap { border: 1px solid var(--line); border-radius: 10px; overflow: auto; }
  th {
    text-align: left; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted); font-weight: 600;
    padding: 10px 14px; background: #fbfbfb; border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }
  td { padding: 11px 14px; border-bottom: 1px solid #f1f1f3; vertical-align: top; }
  tr:last-child td { border-bottom: 0; }
  tbody tr:hover { background: #fcfcfd; }
  .skill { font-weight: 550; }
  .verb, .src { color: var(--muted); font-size: 12.5px; }
  .src { max-width: 260px; }
  .nomatch { color: #b42318; font-style: italic; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .when { white-space: nowrap; color: var(--muted); }

  /* badges */
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 12px; font-weight: 550; white-space: nowrap;
  }
  .b-accept   { background: #eaf6ec; color: #1a7f37; }
  .b-review   { background: #fff6e0; color: #9a6700; }
  .b-emerging { background: #f3ecff; color: #6f42c1; }
  .b-cat      { background: #f1f1f4; color: #3f3f46; }
  .b-exactMatch { background: #eaf6ec; color: #1a7f37; }
  .b-closeMatch { background: #fff6e0; color: #9a6700; }

  /* confidence bar */
  .bar { height: 5px; width: 54px; background: #eeeef1; border-radius: 3px; margin-top: 5px; }
  .bar span { display: block; height: 100%; border-radius: 3px; }
  .f-accept   { background: #2da44e; }
  .f-review   { background: #d4a72c; }
  .f-emerging { background: #8250df; }

  /* filter bar */
  .filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 14px; }
  .filters button {
    padding: 5px 12px; font-size: 13px; border: 1px solid var(--line);
    background: #fff; border-radius: 999px; cursor: pointer; font-family: inherit;
  }
  .filters button.on { background: #18181b; color: #fff; border-color: #18181b; }
  .filters input {
    padding: 6px 10px; font-size: 13px; border: 1px solid var(--line);
    border-radius: 6px; width: 200px; margin-left: auto; font-family: inherit;
  }
  .empty { color: var(--muted); padding: 28px; text-align: center; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <nav>
    <span class="brand">BYDO</span>
    <a href="/" class="{{NAV_UPLOAD}}">Upload</a>
    <a href="/results" class="{{NAV_RESULTS}}">Results</a>
  </nav>
{{BODY}}
</div>
</body>
</html>
"""


def page(title, body, active="", width="880px"):
    return (PAGE_SHELL
            .replace("{{TITLE}}", esc(title))
            .replace("{{WIDTH}}", width)
            .replace("{{NAV_UPLOAD}}", "active" if active == "upload" else "")
            .replace("{{NAV_RESULTS}}", "active" if active == "results" else "")
            .replace("{{BODY}}", body))


UPLOAD_FORM = """
  <h1>Upload job description</h1>
  <p class="sub">PDF, DOCX or TXT. Extraction takes a minute or two &mdash; it calls the
  model and re-ranks every skill against ESCO.</p>
  <div class="card">
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
      <p style="margin-bottom:0"><button type="submit">Upload &amp; analyze</button></p>
    </form>
  </div>
"""


FILTER_SCRIPT = """
<script>
  (function () {
    var rows = Array.prototype.slice.call(document.querySelectorAll("tr[data-decision]"));
    var buttons = Array.prototype.slice.call(document.querySelectorAll(".filters button"));
    var search = document.getElementById("search");
    var active = "all";

    function apply() {
      var q = (search.value || "").toLowerCase().trim();
      rows.forEach(function (row) {
        var okFilter = active === "all"
          || (active === "gap" ? row.dataset.gap === "true" : row.dataset.decision === active);
        var okSearch = !q || row.dataset.text.indexOf(q) !== -1;
        row.classList.toggle("hidden", !(okFilter && okSearch));
      });
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        active = button.dataset.filter;
        buttons.forEach(function (b) { b.classList.toggle("on", b === button); });
        apply();
      });
    });
    search.addEventListener("input", apply);
  })();
</script>
"""


def chips(counts):
    items = [
        f'<span class="chip"><b>{counts["total"]}</b> skills</span>',
        f'<span class="chip"><b>{counts["accept"]}</b> accept</span>',
        f'<span class="chip"><b>{counts["review"]}</b> review</span>',
        f'<span class="chip"><b>{counts["emerging"]}</b> emerging</span>',
        f'<span class="chip"><b>{counts["exact"]}</b> exact match</span>',
        f'<span class="chip"><b>{counts["close"]}</b> close match</span>',
    ]
    if counts["gaps"]:
        items.append(f'<span class="chip"><b>{counts["gaps"]}</b> gaps</span>')
    return f'<div class="chips">{"".join(items)}</div>'


def render_skill_row(skill):
    decision = skill.get("decision") or "review"
    match_type = skill.get("match_type") or ""
    gap = bool(skill.get("gap"))
    esco_match = skill.get("esco_match")

    confidence = skill.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    percent = max(0, min(100, round(confidence * 100)))

    match_html = esc(esco_match) if esco_match else '<span class="nomatch">no ESCO match</span>'
    match_badge = (f'<span class="badge b-{esc(match_type)}">{esc(match_type)}</span>'
                   if match_type in ("exactMatch", "closeMatch") else esc(match_type))

    searchable = " ".join(str(skill.get(k) or "") for k in
                          ("noun", "verb", "esco_form", "esco_match", "domain",
                           "skill_type", "category", "source")).lower()

    return f"""      <tr data-decision="{esc(decision)}" data-gap="{'true' if gap else 'false'}"
          data-text="{esc(searchable)}">
        <td>
          <div class="skill">{esc(skill.get('noun'))}</div>
          <div class="verb">{esc(skill.get('verb'))}</div>
        </td>
        <td><span class="badge b-cat">{esc(skill.get('category'))}</span></td>
        <td>{esc(skill.get('skill_type'))}</td>
        <td>{esc(skill.get('domain'))}</td>
        <td>{match_html}<div class="verb">{match_badge}</div></td>
        <td class="num">{confidence:.2f}<div class="bar"><span class="f-{esc(decision)}" style="width:{percent}%"></span></div></td>
        <td><span class="badge b-{esc(decision)}">{esc(decision)}</span></td>
        <td class="src" title="{esc(skill.get('source'))}">{esc(skill.get('source'))}</td>
      </tr>"""


def render_activities(activities):
    """The project activities as the certificate lists them. Empty for a job description."""
    items = []
    for a in activities:
        text = a.get("activity") if isinstance(a, dict) else a
        if not text:
            continue
        # the sentence the activity was read from, on hover, as for a skill
        source = a.get("source", "") if isinstance(a, dict) else ""
        items.append(f'      <li title="{esc(source)}">{esc(text)}</li>')
    if not items:
        return ""

    bullets = "\n".join(items)
    return f"""
  <h2>Project activities</h2>
  <div class="card">
    <p class="sub" style="margin:0 0 10px">The activities within the Learning Box
      included in particular:</p>
    <ul class="activities">
{bullets}
    </ul>
  </div>
"""


def render_result_page(path: Path, data, banner=""):
    """One results/ file as a page of tables."""
    job_id, when = split_result_name(path.stem)

    all_skills = []
    all_activities = []
    sections = []
    for name, domain, skills, activities in iter_jobs(data):
        all_skills.extend(skills)
        all_activities.extend(activities)
        rows = "\n".join(render_skill_row(s) for s in skills)
        sections.append(f"""
  <h2>{esc(name)} <span class="verb" style="font-weight:400">&middot; {esc(domain)}</span></h2>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>Skill</th><th>Cat</th><th>Type</th><th>Domain</th>
        <th>ESCO match</th><th>Confidence</th><th>Decision</th><th>Source</th>
      </tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>""")

    counts = count_skills(all_skills)
    filters = """
  <div class="filters">
    <button data-filter="all" class="on">All</button>
    <button data-filter="accept">Accept</button>
    <button data-filter="review">Review</button>
    <button data-filter="emerging">Emerging</button>
    <button data-filter="gap">Gaps</button>
    <input type="search" id="search" placeholder="Filter skills&hellip;">
  </div>"""

    body = f"""{banner}
  <h1>{esc(job_id)}</h1>
  <p class="sub">{esc(when)} &middot; <code>results/{esc(path.name)}</code> &middot;
    <a class="link" href="/results/{esc(path.name)}/json">raw JSON</a> &middot;
    <a class="link" href="/results">all results</a></p>
{render_activities(all_activities)}
{chips(counts)}{filters}
{"".join(sections)}
{FILTER_SCRIPT}"""

    return page(f"{job_id} — BYDO", body, active="results", width="1100px")


def render_results_index(entries):
    if not entries:
        rows = '<div class="empty">No results yet. <a class="link" href="/">Upload a job description</a> to get started.</div>'
        body = f'  <h1>Results</h1>\n  <p class="sub">Every extraction run, newest first.</p>\n  <div class="card">{rows}</div>'
        return page("Results — BYDO", body, active="results")

    rows = []
    for e in entries:
        counts = e["counts"]
        if e["broken"]:
            summary = '<td colspan="4" class="nomatch">could not be read</td>'
        else:
            summary = (f'<td class="num">{counts["total"]}</td>'
                       f'<td class="num"><span class="badge b-accept">{counts["accept"]}</span></td>'
                       f'<td class="num"><span class="badge b-review">{counts["review"]}</span></td>'
                       f'<td class="num"><span class="badge b-emerging">{counts["emerging"]}</span></td>')
        rows.append(f"""      <tr>
        <td><a class="link skill" href="/results/{esc(e['file'])}">{esc(e['job_id'])}</a>
            <div class="verb">{esc(e['file'])}</div></td>
        <td class="when">{esc(e['when'])}</td>
        <td>{esc(e['domain'])}</td>
        {summary}
      </tr>""")

    body = f"""  <h1>Results</h1>
  <p class="sub">{len(entries)} extraction run{"s" if len(entries) != 1 else ""}, newest first.</p>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>Job</th><th>Run</th><th>Domain</th>
        <th>Skills</th><th>Accept</th><th>Review</th><th>Emerging</th>
      </tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>"""

    return page("Results — BYDO", body, active="results", width="1000px")


def error_page(title, message, details=""):
    detail_html = f'<pre class="verb" style="white-space:pre-wrap">{esc(details)}</pre>' if details else ""
    body = (f'  <h1>{esc(title)}</h1>\n'
            f'  <div class="card"><p style="margin-top:0">{esc(message)}</p>{detail_html}'
            f'<p style="margin-bottom:0"><a class="link" href="/">Back to upload</a></p></div>')
    return page(f"{title} — BYDO", body)


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return page("BYDO Upload", UPLOAD_FORM, active="upload")


@app.get("/results", response_class=HTMLResponse)
def results_index():
    return HTMLResponse(render_results_index(list_results()))


@app.get("/results/{filename}", response_class=HTMLResponse)
def result_detail(filename: str):
    path = safe_result_path(filename)
    if path is None:
        return HTMLResponse(error_page("Not found", f"No results file named {filename}."), status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return HTMLResponse(error_page("Unreadable result", f"{filename} could not be read.", str(e)), status_code=500)
    return HTMLResponse(render_result_page(path, data))


@app.get("/results/{filename}/json")
def result_json(filename: str):
    path = safe_result_path(filename)
    if path is None:
        return JSONResponse(status_code=404, content={"error": f"No results file named {filename}."})
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.post("/upload")
async def upload(file: UploadFile = File(...), job_title: str = Form(""), format: str = "html"):
    wants_json = format == "json"

    def fail(status, message, details=""):
        if wants_json:
            content = {"error": message}
            if details:
                content["details"] = details
            return JSONResponse(status_code=status, content=content)
        return HTMLResponse(error_page("Upload failed", message, details), status_code=status)

    contents = await file.read()

    try:
        raw_text = extract_text(file.filename, contents)
    except (RuntimeError, ValueError) as e:
        return fail(400, str(e))

    if not raw_text:
        return fail(400, "Could not extract any text from the file.")

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
        return fail(500, "Skill extraction failed.", extract_result.stderr[-2000:])

    # ── Step 2: normalize ────────────────────────────────────────────
    normalize_result = subprocess.run(
        ["python3", "scripts/normalize_answers.py", "answers", "--file", jd_stem],
        capture_output=True, text=True
    )
    if normalize_result.returncode != 0:
        return fail(500, "Normalization failed.", normalize_result.stderr[-2000:])

    # ── Step 3: rerank against ESCO ──────────────────────────────────
    rerank_result = subprocess.run(
        ["python3", "scripts/reranking_pipeline.py", "--file", jd_stem],
        capture_output=True, text=True
    )
    if rerank_result.returncode != 0:
        return fail(500, "ESCO reranking failed.", rerank_result.stderr[-2000:])

    # ── Show the result ──────────────────────────────────────────────
    result_path = get_latest_result(jd_stem)
    if result_path is None:
        return fail(500, "Pipeline finished but no results file was found.")

    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    if wants_json:
        return JSONResponse(content=result_data)

    banner = (f'  <p class="sub" style="margin-bottom:16px">Saved as '
              f'<code>jobs/{esc(fname)}</code> and analyzed.</p>')
    return HTMLResponse(render_result_page(result_path, result_data, banner=banner))
