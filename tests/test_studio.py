"""End-to-end API checks using the locally installed ONNX model."""

import io
import json
import unittest
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi.testclient import TestClient

from app import app, distilbert_service


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_onnx_inference(self):
        self.assertTrue(distilbert_service.loaded, distilbert_service.load_error)
        result = distilbert_service.predict_single({
            "office_role": "Municipal Planning and Development Coordinator",
            "lgu": "Ambaguio",
            "governance_area": "Financial Administration & Sustainability",
            "narrative_quote": "We institutionalized year-round documentation and continuous tracking across departments.",
        })
        self.assertEqual(set(result["probabilities"]), {"sustained", "constrained", "deficient"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=3)
        self.assertEqual(result["tokens_preview"][0], "[CLS]")
        self.assertGreater(result["latency_ms"], 0)

    def test_all_sample_formats_and_column_mapping(self):
        for extension in ("csv", "xlsx", "json", "xml", "txt"):
            with self.subTest(extension=extension):
                sample = self.client.get(f"/api/samples/{extension}").json()["data"]
                self.assertGreater(sample["total_rows"], 0)
                self.assertTrue(sample["columns"])
                if extension == "xlsx":
                    content = self.client.get("/api/samples/xlsx?download=true").content
                else:
                    sample_file = ROOT / "sample_data" / f"sample_thematic_dataset.{extension}"
                    content = sample_file.read_bytes()
                parsed = self.client.post("/api/parse", files={"file": (f"sample.{extension}", content)})
                self.assertEqual(parsed.status_code, 200, parsed.text)
                self.assertEqual(parsed.json()["data"]["total_rows"], sample["total_rows"])
        nested = self.client.post("/api/parse", files={"file": ("nested.json", json.dumps({"data": {"records": [{"narrative": "Data analyst"}]}}).encode())})
        self.assertEqual(nested.json()["data"]["records"][0]["narrative"], "Data analyst")
        self.assertEqual(self.client.post("/api/parse", files={"file": ("broken.xml", b"<records><record>")}).status_code, 400)
        self.assertEqual(self.client.post("/api/parse", files={"file": ("old.xls", b"not excel")}).status_code, 400)

    def test_analyze_export_and_validation(self):
        sample = self.client.get("/api/samples/csv").json()["data"]
        response = self.client.post("/api/analyze", json={
            "records": sample["records"],
            "column_mapping": sample["suggested_mapping"],
            "n_clusters": 3,
        })
        self.assertEqual(response.status_code, 200, response.text)
        analysis = response.json()["data"]
        self.assertEqual(analysis["summary"]["total_records"], sample["total_rows"])
        self.assertEqual(len(analysis["records"]), sample["total_rows"])
        self.assertTrue(analysis["theme_distribution"])
        self.assertTrue(analysis["global_tfidf_keyphrases"])
        for fmt in ("csv", "xlsx", "json", "xml", "doc"):
            with self.subTest(format=fmt):
                result = self.client.post("/api/export", json={"format": fmt, "analysis": analysis})
                self.assertEqual(result.status_code, 200, result.text)
                if fmt == "xlsx":
                    self.assertEqual(len(pd.read_excel(io.BytesIO(result.content))), sample["total_rows"])
                elif fmt == "xml":
                    self.assertEqual(len(ET.fromstring(result.content).find("CodedRecords")), sample["total_rows"])
                elif fmt == "json":
                    self.assertEqual(len(result.json()["records"]), sample["total_rows"])
                elif fmt == "doc":
                    self.assertIn(b"Executive Analytics", result.content)
                    self.assertIn(b"<table", result.content)
                else:
                    self.assertIn(b"Primary_Theme", result.content)
        self.assertEqual(self.client.post("/api/export", json={"format": "exe"}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json={"raw_text": " "}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json={"raw_text": "Mayor executive order", "n_clusters": 100}).status_code, 400)

    def test_direct_text_and_codebook(self):
        response = self.client.post("/api/analyze", json={"raw_text": "Mayor executive order institutionalized SOP\nCompliance cramming and rushed folder preparation", "split_mode": "lines"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"]["summary"]["total_records"], 2)
        custom = [{"id": "custom_qa", "name": "Quality assurance", "color": "#345678", "keywords": ["testing"]}]
        self.assertEqual(self.client.post("/api/themes", json={"themes": custom}).status_code, 200)
        self.assertEqual(self.client.get("/api/themes").json()["themes"][0]["id"], "custom_qa")
        self.client.post("/api/themes/reset")

    def test_respondent_groups_and_breakdown(self):
        from engine.thematic_coder import classify_respondent_group, RESPONDENT_GROUPS

        # 1. Test role classification mappings
        self.assertEqual(classify_respondent_group("Municipal Mayor"), "Mayor / Admin")
        self.assertEqual(classify_respondent_group("City Mayor"), "Mayor / Admin")
        self.assertEqual(classify_respondent_group("Vice Mayor & Sanggunian Lead"), "Mayor / Admin")
        self.assertEqual(classify_respondent_group("Executive Assistant to the Mayor"), "Mayor / Admin")
        self.assertEqual(classify_respondent_group("BAC Secretariat"), "Mayor / Admin")

        self.assertEqual(classify_respondent_group("SGLG Focal Person"), "LCE / Focal")
        self.assertEqual(classify_respondent_group("Planning & Development Coordinator"), "LCE / Focal")
        self.assertEqual(classify_respondent_group("Municipal Budget Officer"), "LCE / Focal")
        self.assertEqual(classify_respondent_group("Disaster Risk Reduction Officer"), "LCE / Focal")

        self.assertEqual(classify_respondent_group("DILG Field Officer"), "DILG / PLGU")
        self.assertEqual(classify_respondent_group("MLGOO"), "DILG / PLGU")
        self.assertEqual(classify_respondent_group("Provincial Assessment Team Lead"), "DILG / PLGU")

        # 2. Test analysis output contains respondent_group and respondent_group_breakdown
        sample = self.client.get("/api/samples/csv").json()["data"]
        response = self.client.post("/api/analyze", json={
            "records": sample["records"],
            "column_mapping": sample["suggested_mapping"],
            "n_clusters": 3,
        })
        self.assertEqual(response.status_code, 200, response.text)
        analysis = response.json()["data"]

        for rec in analysis["records"]:
            self.assertIn("respondent_group", rec)
            self.assertIn(rec["respondent_group"], RESPONDENT_GROUPS)

        rgb = analysis.get("respondent_group_breakdown")
        self.assertIsNotNone(rgb)
        self.assertEqual(rgb["groups"], RESPONDENT_GROUPS)
        self.assertTrue(len(rgb["themes"]) > 0)
        total_from_groups = sum(rgb["group_totals"].values())
        self.assertEqual(total_from_groups, analysis["summary"]["total_records"])

        # 3. Test export contains Respondent_Group in CSV and XLSX sheet
        export_csv = self.client.post("/api/export", json={"format": "csv", "analysis": analysis})
        self.assertEqual(export_csv.status_code, 200)
        self.assertIn(b"Respondent_Group", export_csv.content)

        export_xlsx = self.client.post("/api/export", json={"format": "xlsx", "analysis": analysis})
        self.assertEqual(export_xlsx.status_code, 200)
        xls_file = pd.ExcelFile(io.BytesIO(export_xlsx.content))
        self.assertIn("Respondent_Group_Breakdown", xls_file.sheet_names)
        df_records = pd.read_excel(xls_file, sheet_name="Coded_Records")
        self.assertIn("Respondent_Group", df_records.columns)
        df_rgb = pd.read_excel(xls_file, sheet_name="Respondent_Group_Breakdown")
        for g in RESPONDENT_GROUPS:
            self.assertIn(g, df_rgb.columns)

    def test_taglish_and_emergent_coding(self):
        from engine.thematic_coder import DOMAIN_STOP_WORDS

        # 1. Tagalog stop words verification
        self.assertIn("ang", DOMAIN_STOP_WORDS)
        self.assertIn("mga", DOMAIN_STOP_WORDS)
        self.assertIn("dahil", DOMAIN_STOP_WORDS)

        # 2. Taglish morphological matching
        taglish_res = self.client.post("/api/analyze", json={
            "raw_text": "Nag-cramming kami bago dumating ang DILG dahil kinulang sa tao at pondo.",
            "split_mode": "single",
        })
        self.assertEqual(taglish_res.status_code, 200)
        data = taglish_res.json()["data"]
        rec = data["records"][0]
        self.assertTrue(rec.get("is_taglish"))
        self.assertIn("Theme 3", rec["primary_theme"])
        matched_str = " ".join(rec["matched_codes"]).lower()
        self.assertTrue(any(k in matched_str for k in ["cramming", "kulang sa tao", "pondo", "kakulangan"]))

        # 3. Emergent candidate code extraction for unassigned records
        unmatched_res = self.client.post("/api/analyze", json={
            "raw_text": "Coastal mangrove reforestation and solar streetlight maintenance.",
            "split_mode": "single",
        })
        self.assertEqual(unmatched_res.status_code, 200)
        unmatched_rec = unmatched_res.json()["data"]["records"][0]
        self.assertEqual(unmatched_rec["primary_theme"], "Unassigned / Needs Review")
        self.assertTrue(len(unmatched_rec["matched_codes"]) > 0)
        self.assertTrue(any("[Emergent]" in c for c in unmatched_rec["matched_codes"]))


if __name__ == "__main__":
    unittest.main()

