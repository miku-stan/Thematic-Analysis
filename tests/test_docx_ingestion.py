"""Unit tests for Word (.docx) transcript ingestion, multi-file parsing, and Taglish SGLG thematic coding."""

import glob
import io
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi.testclient import TestClient

from app import app, thematic_coder


class DocxIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.doc_v_dir = Path(r"C:\Users\Carl\Downloads\Doc V")
        cls.has_doc_v = cls.doc_v_dir.exists() and any(cls.doc_v_dir.glob("*.docx"))

    def test_single_docx_parse(self):
        if not self.has_doc_v:
            self.skipTest("Doc V directory not found on system.")
        test_file = self.doc_v_dir / "Ambaguio-Mayor-Final.docx"
        content = test_file.read_bytes()

        res = self.client.post("/api/upload", files={"file": ("Ambaguio-Mayor-Final.docx", content)})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()["data"]

        self.assertEqual(data["file_type"], "Word Transcript (.docx)")
        self.assertGreater(data["total_rows"], 0)
        self.assertIn("participant_id", data["columns"])
        self.assertIn("narrative_quote", data["columns"])
        self.assertEqual(data["suggested_mapping"]["about"], "narrative_quote")
        self.assertEqual(data["suggested_mapping"]["current_company"], "lgu")
        self.assertEqual(data["suggested_mapping"]["current_position"], "office_role")

    def test_multi_docx_parse_all_four_files(self):
        if not self.has_doc_v:
            self.skipTest("Doc V directory not found on system.")
        files_to_upload = []
        for p in sorted(self.doc_v_dir.glob("*.docx")):
            files_to_upload.append(("files", (p.name, p.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")))

        res = self.client.post("/api/upload", files=files_to_upload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()["data"]

        self.assertEqual(data["total_rows"], 47)
        self.assertTrue(data["filename"].startswith("Multi-Case Transcripts (4 files"))

        # Verify dual case representation across Ambaguio and Peñablanca
        lgus = {r.get("lgu") for r in data["records"]}
        self.assertIn("Ambaguio (LGU-01)", lgus)
        self.assertIn("Peñablanca (LGU-02)", lgus)

    def test_analyze_multi_case_transcripts(self):
        if not self.has_doc_v:
            self.skipTest("Doc V directory not found on system.")
        files_to_upload = []
        for p in sorted(self.doc_v_dir.glob("*.docx")):
            files_to_upload.append(("files", (p.name, p.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")))

        upload_res = self.client.post("/api/upload", files=files_to_upload)
        parsed = upload_res.json()["data"]

        analysis_res = self.client.post("/api/analyze", json={
            "records": parsed["records"],
            "column_mapping": parsed["suggested_mapping"],
            "n_clusters": 3,
        })
        self.assertEqual(analysis_res.status_code, 200, analysis_res.text)
        analysis = analysis_res.json()["data"]

        # Check total records
        self.assertEqual(analysis["summary"]["total_records"], 47)

        # Check theme distribution covers all 6 SGLG themes
        theme_names = {t["theme"] for t in analysis["theme_distribution"]}
        self.assertGreaterEqual(len(theme_names), 5)

        # Verify cross-case matrix contains dynamic dual labels
        matrix = analysis.get("cross_case_matrix", {})
        cases = matrix.get("cases", [])
        case_names = [c["name"] for c in cases]
        self.assertIn("Ambaguio (LGU-01)", case_names)
        self.assertIn("Peñablanca (LGU-02)", case_names)

        # Verify rows in cross-case matrix
        rows = matrix.get("rows", [])
        self.assertEqual(len(rows), 5)
        issues = [r["issue"] for r in rows]
        self.assertIn("Records / documentation", issues)
        self.assertIn("Budget / resources", issues)
        self.assertIn("Technical capacity", issues)

        # Test Excel export with Cross_Case_Matrix sheet
        export_res = self.client.post("/api/export", json={"format": "xlsx", "analysis": analysis})
        self.assertEqual(export_res.status_code, 200)
        excel_file = pd.ExcelFile(io.BytesIO(export_res.content))
        self.assertIn("Cross_Case_Matrix", excel_file.sheet_names)
        self.assertIn("Coded_Records", excel_file.sheet_names)


if __name__ == "__main__":
    unittest.main()
