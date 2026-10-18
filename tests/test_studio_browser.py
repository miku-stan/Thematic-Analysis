"""Headless browser smoke test; run while app.py serves localhost:8050."""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def run():
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: errors.append(f"HTTP {response.status} {response.url}") if response.status >= 400 else None)
        page.goto("http://127.0.0.1:8050", wait_until="networkidle", timeout=120000)
        page.locator("#modelStatusText").get_by_text("ONNX Engine Ready").wait_for(timeout=15000)
        page.locator("#fileInput").set_input_files(ROOT / "sample_data" / "sample_thematic_dataset.csv")
        page.locator("#uploadedFileMeta").get_by_text("records", exact=False).wait_for()
        page.get_by_role("button", name="Run DistilBERT Thematic Analysis").click()
        page.locator("#themeDistributionChart .theme-bar-row").first.wait_for(timeout=120000)
        if os.environ.get("THEMATIC_SCREENSHOT"):
            page.screenshot(path=os.environ["THEMATIC_SCREENSHOT"])
        assert page.locator("#themeDistributionChart .theme-bar-row").count() > 0
        page.locator(".nav-tab[data-tab='records']").click()
        assert page.locator("#codedRecordsTable tbody tr").count() > 0
        page.locator("#filterConfidenceSelect").select_option("0.5")
        page.locator("#codedRecordsTable tbody button").first.click()
        page.locator("#inspectorModal.open").wait_for()
        assert page.locator("#modalInspectorBody .token-chip").count() > 0
        page.get_by_role("button", name="Apply override").click()
        page.get_by_role("button", name="Close").click()
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Download Excel (.xlsx)").click()
        assert download_info.value.suggested_filename.endswith(".xlsx")
        page.locator(".nav-tab[data-tab='ingest']").click()
        page.locator("#fileInput").set_input_files(ROOT / "sample_data" / "sample_thematic_dataset.xml")
        page.locator("#uploadedFileName").get_by_text("sample_thematic_dataset.xml").wait_for()
        page.locator(".nav-tab[data-tab='codebook']").click()
        page.get_by_role("button", name="+ Add Custom Theme").click()
        assert page.locator(".theme-editor").count() == 9
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator(".nav-tab[data-tab='dashboard']").click()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        assert not errors, errors
        browser.close()
    print("PASS browser: model ready, sample analysis, chart, records, inspector, override, XLSX download, XML upload, codebook")


if __name__ == "__main__":
    run()
