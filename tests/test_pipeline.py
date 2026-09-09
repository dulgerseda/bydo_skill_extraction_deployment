"""
Regression tests for the reproducibility guarantees — BYDO Project
=============================================================
Offline: no API key, no network, no model downloads beyond the image.

Run inside the container, where the dependencies live:

    docker compose run --rm --entrypoint python3 bydo -m unittest discover -s tests -v

What is pinned here is the behaviour a caller depends on:
  1. an identical document returns an identical answer (cache)
  2. the ESCO lookup does not depend on how the model worded esco_form
  3. artefact names never reach the results
  4. confidence thresholds keep their meaning
  5. project activities reach results/ in certificate wording
  6. the API key can be supplied from outside, and its absence is readable
=============================================================
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import normalize_answers as norm  # noqa: E402


def skill(**kw):
    base = {
        "noun": "Python", "verb": "Use Python", "esco_form": "use Python",
        "category": "K", "skill_type": "Tool", "domain": "Software Engineering",
        "tool": None, "action": None, "source": "We use Python.",
    }
    base.update(kw)
    return base


def load_reranker():
    """reranking_pipeline.py does its work under __main__, so importing it is cheap."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rr", str(REPO / "scripts" / "reranking_pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtefactNames(unittest.TestCase):
    """A file, a config or an internal service is never a skill."""

    def test_dockerfile_becomes_docker(self):
        s = skill(noun="Dockerfile Creation", category="OS", skill_type="Methodology")
        note = norm.rewrite_artefacts(s)
        self.assertEqual(s["noun"], "Docker")
        self.assertEqual(s["skill_type"], "Tool")
        self.assertIn("artefact", note)

    def test_docker_compose_variants_all_collapse(self):
        for name in ("docker-compose.yml", "Docker Compose Configuration",
                     "docker compose setup", "DockerCompose"):
            s = skill(noun=name)
            norm.rewrite_artefacts(s)
            self.assertEqual(s["noun"], "Docker", f"{name} should collapse to Docker")

    def test_requirements_txt_becomes_python(self):
        s = skill(noun="requirements.txt Management")
        norm.rewrite_artefacts(s)
        self.assertEqual(s["noun"], "Python")

    def test_unknown_file_name_is_dropped(self):
        s = skill(noun="html_upload.py")
        note = norm.rewrite_artefacts(s)
        self.assertTrue(s.get("_drop"))
        self.assertIn("file name", note)

    def test_real_skills_survive(self):
        for name in ("Docker Deployment", "Python", "Data Visualization",
                     "Requirements Engineering", "Pipeline Development"):
            s = skill(noun=name)
            norm.rewrite_artefacts(s)
            self.assertEqual(s["noun"], name, f"{name} must not be rewritten")
            self.assertFalse(s.get("_drop"), f"{name} must not be dropped")


class EmptySuffixes(unittest.TestCase):
    """Two spellings of one skill must not both survive."""

    def test_implementation_collapses_onto_base(self):
        skills = [skill(noun="Semantic Search"), skill(noun="Semantic Search Implementation")]
        norm.collapse_empty_suffix(skills)
        kept = [s["noun"] for s in skills if not s.get("_drop")]
        self.assertEqual(kept, ["Semantic Search"])

    def test_meaningful_suffix_is_kept(self):
        skills = [skill(noun="Docker"), skill(noun="Docker Deployment")]
        norm.collapse_empty_suffix(skills)
        kept = [s["noun"] for s in skills if not s.get("_drop")]
        self.assertEqual(kept, ["Docker", "Docker Deployment"])

    def test_suffix_alone_is_kept_when_base_absent(self):
        skills = [skill(noun="Pipeline Execution")]
        norm.collapse_empty_suffix(skills)
        self.assertFalse(skills[0].get("_drop"))


class EscoQuery(unittest.TestCase):
    """The ESCO lookup is derived from stable fields, never from model prose."""

    def test_templates_per_category(self):
        cases = [
            (skill(noun="Python", category="K", skill_type="Tool"), "use Python"),
            (skill(noun="Signal Processing", category="K", skill_type="Knowledge"),
             "Apply Signal Processing techniques"),
            (skill(noun="Python for Anomaly Detection", category="IPS", skill_type="Tool",
                   tool="Python", action="Anomaly Detection"),
             "Apply Python-based Anomaly Detection"),
            (skill(noun="Budget Management", category="OS", skill_type="Methodology"),
             "Budget Management"),
            (skill(noun="Collaboration", category="SF", skill_type="Soft Skill"),
             "Collaboration"),
        ]
        for s, expected in cases:
            norm.build_esco_query(s)
            self.assertEqual(s["esco_query"], expected)

    def test_query_ignores_esco_form_wording(self):
        wordings = ["use Python", "use Python 3.11", "Utilize Python for data analysis"]
        queries = set()
        for wording in wordings:
            s = skill(esco_form=wording)
            norm.build_esco_query(s)
            queries.add(s["esco_query"])
        self.assertEqual(queries, {"use Python"},
                         "esco_form must not influence the ESCO lookup")


class Thresholds(unittest.TestCase):
    """Confidence boundaries decide what a human has to look at."""

    @classmethod
    def setUpClass(cls):
        cls.rr = load_reranker()

    def test_decision_boundaries(self):
        self.assertEqual(self.rr.make_decision(0.80), "accept")
        self.assertEqual(self.rr.make_decision(0.7999), "review")
        self.assertEqual(self.rr.make_decision(0.50), "review")
        self.assertEqual(self.rr.make_decision(0.4999), "emerging")

    def test_match_type_boundaries(self):
        self.assertEqual(self.rr.make_match_type(0.85), "exactMatch")
        self.assertEqual(self.rr.make_match_type(0.8499), "closeMatch")
        self.assertEqual(self.rr.make_match_type(0.65), "closeMatch")
        self.assertEqual(self.rr.make_match_type(0.6499), "emerging")

    def test_score_normalisation_is_stable(self):
        self.assertEqual(self.rr.normalize_score(0.0), 0.5)
        self.assertEqual(self.rr.normalize_score(6.0), self.rr.normalize_score(6.0))


class ProjectActivities(unittest.TestCase):
    """The certificate bullets have to reach results/ in the shape a certificate needs."""

    @classmethod
    def setUpClass(cls):
        cls.rr = load_reranker()

    def activity(self, text, source="We did it."):
        return {"activity": text, "source": source}

    def test_trailing_full_stop_and_bullet_are_stripped(self):
        cleaned = norm.clean_activities([
            self.activity("processing and merging of internal datasets."),
            self.activity("• exploratory analysis of tariff data"),
            self.activity("  development and evaluation of classification models  "),
        ])
        self.assertEqual([a["activity"] for a in cleaned], [
            "processing and merging of internal datasets",
            "exploratory analysis of tariff data",
            "development and evaluation of classification models",
        ])

    def test_wording_is_otherwise_left_alone(self):
        text = "exploratory SQL analysis of internal item and tariff data on Google Cloud"
        cleaned = norm.clean_activities([self.activity(text)])
        self.assertEqual(cleaned[0]["activity"], text)

    def test_source_is_kept(self):
        cleaned = norm.clean_activities([self.activity("web scraping of customs websites", "I scraped it.")])
        self.assertEqual(cleaned[0]["source"], "I scraped it.")

    def test_empty_items_are_dropped(self):
        cleaned = norm.clean_activities([self.activity(""), self.activity("•"), self.activity("real work")])
        self.assertEqual([a["activity"] for a in cleaned], ["real work"])

    def test_missing_field_is_an_empty_list_not_a_crash(self):
        self.assertEqual(norm.clean_activities(None), [])
        self.assertEqual(norm.clean_activities([]), [])

    def test_survives_normalization_end_to_end(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            payload = {"JD_TEST": {
                "domain": "Data Science",
                "project_activities": [self.activity("processing and merging of internal datasets.")],
                "skills": [skill()],
            }}
            src = tmp / "in.json"
            src.write_text(json.dumps(payload), encoding="utf-8")
            out = tmp / "out.json"
            norm.normalize_file(src, out)

            got = json.loads(out.read_text(encoding="utf-8"))["JD_TEST"]
            self.assertEqual([a["activity"] for a in got["project_activities"]],
                             ["processing and merging of internal datasets"])
            self.assertEqual(len(got["skills"]), 1, "activities must not disturb the skills")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reranking_carries_them_through(self):
        activities = [self.activity("processing and merging of internal datasets")]
        skills, domain, got = self.rr.read_job({
            "domain": "Data Science", "project_activities": activities, "skills": [skill()],
        })
        self.assertEqual(got, activities)
        self.assertEqual(domain, "Data Science")
        self.assertEqual(len(skills), 1)

    def test_a_document_without_activities_is_not_an_error(self):
        # a job description advertises a role, so it reports no project work
        _, _, got = self.rr.read_job({"domain": "Data Science", "skills": [skill()]})
        self.assertEqual(got, [])

        # and an old answer file holds a bare skills list
        _, _, got = self.rr.read_job([skill()])
        self.assertEqual(got, [])


class ScriptTree(unittest.TestCase):
    """A throwaway copy of the extraction script, with a tree of its own.

    The script derives every path from its own location, so a copy in a temporary
    tree reads and writes only inside that tree and never touches the repository.
    """

    SETTINGS = {
        "model": "gpt-4o-2024-11-20",
        "prompt": "prompt_v13.txt",
        "temperature": 0, "seed": 42,
        "json_mode": True, "strict_schema": False,
        "max_tokens": 8126,
    }

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "scripts").mkdir()
        shutil.copy(REPO / "scripts" / "extract_skills_api.py", self.tmp / "scripts")
        shutil.copytree(REPO / "prompts", self.tmp / "prompts")
        (self.tmp / "jobs").mkdir()
        (self.tmp / "jobs" / "JD_TEST.txt").write_text(
            "We need Python and Docker. Strong collaboration skills.", encoding="utf-8")
        self.write_config(openai_api_key="sk-not-used-offline")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_config(self, **extra):
        """A config.json holding the pinned settings, plus whatever a test adds."""
        (self.tmp / "config.json").write_text(
            json.dumps({**self.SETTINGS, **extra}), encoding="utf-8")

    def run_script(self, *extra, key=None):
        """Each call states its own key, so a key exported in the developer's shell
        can neither reach this suite nor pay for a real API call."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        if key is not None:
            env["OPENAI_API_KEY"] = key
        return subprocess.run(
            [sys.executable, "scripts/extract_skills_api.py", "--file", "JD_TEST.txt", *extra],
            cwd=self.tmp, capture_output=True, text=True, timeout=120, env=env)


class ApiKey(ScriptTree):
    """The key is a secret, so it must not have to live in a file of settings."""

    def test_the_environment_alone_is_enough(self):
        self.write_config()  # settings only, no key anywhere on disk
        out = self.run_script("--print-cache-key", key="sk-from-environment")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_a_dotenv_file_is_read_like_an_export(self):
        self.write_config()
        (self.tmp / ".env").write_text("# my key\nOPENAI_API_KEY=sk-from-dotenv\n",
                                       encoding="utf-8")
        # the config carries no key, so getting this far proves .env was read
        out = self.run_script("--print-cache-key")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_the_environment_wins_over_dotenv_and_config(self):
        self.write_config(openai_api_key="sk-from-config")
        (self.tmp / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
        out = self.run_script(key="sk-from-environment")
        self.assertIn("API key from OPENAI_API_KEY", out.stdout)

    def test_a_key_in_config_json_still_works(self):
        out = self.run_script()
        self.assertIn("API key from config.json", out.stdout)

    def test_no_key_anywhere_says_how_to_set_one(self):
        self.write_config()
        out = self.run_script()
        self.assertNotEqual(out.returncode, 0, "a missing key must not look like success")
        self.assertIn("No OpenAI API key found", out.stdout)
        self.assertIn("OPENAI_API_KEY", out.stdout)

    def test_the_readme_placeholder_does_not_count_as_a_key(self):
        self.write_config(openai_api_key="your-api-key-here")
        out = self.run_script()
        self.assertIn("No OpenAI API key found", out.stdout)


class CacheGuarantee(ScriptTree):
    """The promise a caller cares about: same document in, same skills out.

    With the cache pre-seeded there is no API call, which keeps this test offline.
    """

    FIXTURE = REPO / "tests" / "fixtures" / "extraction_answer.json"

    def cache_key(self):
        out = self.run_script("--print-cache-key")
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.split()[-1]

    def seed_cache(self, payload):
        cache = self.tmp / "cache" / "extractions"
        cache.mkdir(parents=True)
        (cache / f"{self.cache_key()}.json").write_text(json.dumps({
            "jd": "JD_TEST", "model": "gpt-4o-2024-11-20",
            "prompt_file": "prompt_v13.txt", "prompt_hash": "test",
            "temperature": 0, "seed": 42, "cached_at": "2026-01-01T00:00:00",
            "meta": {}, "parsed": payload,
        }), encoding="utf-8")

    def test_key_is_stable_and_content_dependent(self):
        first = self.cache_key()
        self.assertEqual(first, self.cache_key(), "same request must give the same key")

        (self.tmp / "jobs" / "JD_TEST.txt").write_text("Different document.", encoding="utf-8")
        self.assertNotEqual(first, self.cache_key(), "changed document must give a new key")

    def test_key_changes_with_the_seed(self):
        first  = self.cache_key()
        config = json.loads((self.tmp / "config.json").read_text())
        config["seed"] = 43
        (self.tmp / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.assertNotEqual(first, self.cache_key(), "changed seed must give a new key")

    def test_cached_document_returns_the_recorded_answer(self):
        payload = json.loads(self.FIXTURE.read_text(encoding="utf-8"))
        self.seed_cache(payload)

        for attempt in range(2):
            result = self.run_script()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cache hit", result.stdout, f"attempt {attempt} called the API")

            written = sorted((self.tmp / "answers").glob("*.json"))[-1]
            got = json.loads(written.read_text(encoding="utf-8"))["JD_TEST"]
            self.assertEqual(got, payload, "cached answer must come back unchanged")

    def test_no_cache_flag_bypasses_the_cache(self):
        self.seed_cache(json.loads(self.FIXTURE.read_text(encoding="utf-8")))
        result = self.run_script("--no-cache")
        # no valid key offline, so the call fails: the point is that it was attempted
        self.assertNotIn("cache hit", result.stdout)
        self.assertIn("API error", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
