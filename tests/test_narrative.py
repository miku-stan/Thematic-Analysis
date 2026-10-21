"""Unit tests for Qualitative Thematic Analysis Narrative Synthesis Generation."""

import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app import app, thematic_coder


class NarrativeSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Load sample SGLG dataset for testing
        sample_res = cls.client.get("/api/samples/csv")
        cls.sample_data = sample_res.json()["data"]

        # Run analysis to have an active analysis result
        analysis_res = cls.client.post(
            "/api/analyze",
            json={
                "records": cls.sample_data["records"],
                "column_mapping": cls.sample_data["suggested_mapping"],
            },
        )
        cls.analysis = analysis_res.json()["data"]

    def test_narrative_generation_endpoint(self):
        res = self.client.post("/api/narrative", json={"analysis": self.analysis})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()["data"]

        self.assertIn("title", data)
        self.assertIn("sections", data)
        self.assertIn("markdown", data)
        self.assertIn("html_document", data)
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["total_records"], 18)

        # Check sections structure
        section_ids = [s["id"] for s in data["sections"]]
        self.assertIn("overview", section_ids)
        self.assertIn("operational_barriers", section_ids)
        self.assertIn("culminating_synthesis", section_ids)
        self.assertIn("author_guidance", section_ids)

        # Verify participant quotes are present
        all_quotes = []
        for s in data["sections"]:
            all_quotes.extend(s.get("quotes", []))
        self.assertGreater(len(all_quotes), 0)
        self.assertTrue(any("LCE-03" in q["participant_id"] or "LCE-01" in q["participant_id"] for q in all_quotes))

    def test_narrative_case_filter(self):
        res = self.client.post("/api/narrative", json={"analysis": self.analysis, "case_filter": "LGU-01"})
        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        self.assertLess(data["metadata"]["total_records"], 18)
        self.assertGreater(data["metadata"]["total_records"], 0)

    def test_narrative_export_endpoints(self):
        # Test .doc export
        res_doc = self.client.post("/api/narrative/export", json={"analysis": self.analysis, "format": "doc"})
        self.assertEqual(res_doc.status_code, 200)
        self.assertEqual(res_doc.headers["content-type"], "application/msword")
        self.assertIn(b"Qualitative Thematic Analysis Narrative Synthesis", res_doc.content)

        # Test .md export
        res_md = self.client.post("/api/narrative/export", json={"analysis": self.analysis, "format": "md"})
        self.assertEqual(res_md.status_code, 200)
        self.assertIn("text/markdown", res_md.headers["content-type"])
        self.assertIn(b"# Qualitative Thematic Analysis Narrative Synthesis", res_md.content)

        # Test .txt export
        res_txt = self.client.post("/api/narrative/export", json={"analysis": self.analysis, "format": "txt"})
        self.assertEqual(res_txt.status_code, 200)
        self.assertIn("text/plain", res_txt.headers["content-type"])
        self.assertIn(b"QUALITATIVE THEMATIC ANALYSIS NARRATIVE SYNTHESIS", res_txt.content)

    def test_narrative_no_asterisks(self):
        res = self.client.post("/api/narrative", json={"analysis": self.analysis})
        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        for sec in data["sections"]:
            for p in sec.get("paragraphs", []):
                self.assertNotIn("**", p, f"Found double asterisks in paragraph: {p}")
                self.assertNotIn("*", p, f"Found asterisk in paragraph: {p}")

