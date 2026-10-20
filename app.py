"""Standalone FastAPI Server for the DistilBERT Thematic Analysis System."""

import io
import os
import threading
import webbrowser
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import BadZipFile

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.distilbert_service import DistilBertService
from engine.file_ingestor import (
    apply_column_mapping,
    export_results,
    parse_multiple_uploaded_files,
    parse_uploaded_file,
)
from engine.thematic_coder import DEFAULT_THEMES, THEME_PRESETS, ThematicCoder

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_DIR = BASE_DIR / "sample_data"


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("THEMATIC_OPEN_BROWSER") == "1":
        threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8050")).start()
    yield


app = FastAPI(
    title="DistilBERT Thematic Analysis",
    description="Standalone AI System for Thematic Coding, Unsupervised Theme Discovery, and DistilBERT Sequence Classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core singleton engines
distilbert_service = DistilBertService()
thematic_coder = ThematicCoder()
latest_analysis_cache: Optional[Dict[str, Any]] = None


class AnalyzeRequest(BaseModel):
    records: Optional[List[Dict[str, Any]]] = None
    raw_text: Optional[str] = None
    split_mode: str = "lines"  # "lines" | "paragraphs" | "single"
    single_profile: Optional[Dict[str, Any]] = None
    column_mapping: Optional[Dict[str, str]] = None
    use_hybrid_rules: bool = True
    n_clusters: int = 4
    custom_themes: Optional[List[Dict[str, Any]]] = None


class ThemesUpdateRequest(BaseModel):
    themes: List[Dict[str, Any]]


class ExportRequest(BaseModel):
    format: str = "csv"
    analysis: Optional[Dict[str, Any]] = None


@app.get("/api/status")
async def get_status():
    """Return DistilBERT model status, architecture specs, and active thematic codebook."""
    return {
        "ok": True,
        "model": distilbert_service.get_model_status(),
        "themes": thematic_coder.get_themes(),
        "presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "theme_count": len(p["themes"]),
            }
            for p in THEME_PRESETS.values()
        ],
        "supported_formats": [".csv", ".xlsx", ".json", ".xml", ".txt"],
    }


@app.get("/api/themes")
async def get_themes():
    return {"ok": True, "themes": thematic_coder.get_themes()}


@app.post("/api/themes")
async def update_themes(req: ThemesUpdateRequest):
    updated = thematic_coder.set_themes(req.themes)
    return {"ok": True, "themes": updated}


@app.post("/api/themes/reset")
async def reset_themes():
    updated = thematic_coder.set_themes([dict(t) for t in DEFAULT_THEMES])
    return {"ok": True, "themes": updated}


@app.get("/api/presets")
async def get_presets():
    return {
        "ok": True,
        "presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "theme_count": len(p["themes"]),
            }
            for p in THEME_PRESETS.values()
        ],
    }


@app.post("/api/presets/{preset_id}")
async def load_preset(preset_id: str):
    if preset_id not in THEME_PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{preset_id}'")
    updated = thematic_coder.set_preset(preset_id)
    return {"ok": True, "preset": preset_id, "themes": updated}


@app.post("/api/parse")
@app.post("/api/upload")
async def upload_file(request: Request):
    """Ingest and parse single or multiple uploaded files (.csv, .xlsx, .json, .xml, .txt, .docx)."""
    try:
        form = await request.form()
        uploaded_files = []
        for _, val in form.multi_items():
            if hasattr(val, "read") and hasattr(val, "filename"):
                uploaded_files.append(val)

        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        file_items: List[Tuple[str, bytes]] = []
        for uf in uploaded_files:
            content = await uf.read(10_000_001)
            if not content:
                continue
            if len(content) > 10_000_000:
                raise HTTPException(status_code=413, detail=f"File '{uf.filename}' exceeds 10 MB limit.")
            file_items.append((uf.filename or "uploaded.docx", content))

        if not file_items:
            raise HTTPException(status_code=400, detail="Uploaded file(s) are empty.")

        if len(file_items) == 1:
            parsed = parse_uploaded_file(file_items[0][0], file_items[0][1])
        else:
            parsed = parse_multiple_uploaded_files(file_items)

        if len(parsed["records"]) > 5000:
            raise HTTPException(status_code=413, detail="Datasets are limited to 5,000 records per analysis.")
        return {"ok": True, "data": parsed}
    except HTTPException:
        raise
    except (ValueError, ET.ParseError, BadZipFile, pd.errors.ParserError) as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {exc}")



@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest):
    """Run DistilBERT sequence classification and Thematic Coding on records or text."""
    global latest_analysis_cache
    try:
        if req.custom_themes:
            thematic_coder.set_themes(req.custom_themes)

        records_to_analyze: List[Dict[str, Any]] = []

        if req.records:
            records_to_analyze = apply_column_mapping(req.records, req.column_mapping)
        elif req.single_profile:
            records_to_analyze = [req.single_profile]
        elif req.raw_text and req.raw_text.strip():
            raw = req.raw_text.strip()
            if req.split_mode == "single":
                records_to_analyze = [{"full_name": "Direct Entry #1", "response": raw}]
            elif req.split_mode == "paragraphs":
                blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
                records_to_analyze = [
                    {"full_name": f"Paragraph #{i + 1}", "response": blk}
                    for i, blk in enumerate(blocks)
                ]
            else:
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                records_to_analyze = [
                    {"full_name": f"Entry #{i + 1}", "response": ln}
                    for i, ln in enumerate(lines)
                ]

        if not records_to_analyze:
            raise HTTPException(
                status_code=400,
                detail="No records or text provided for analysis.",
            )
        if len(records_to_analyze) > 5000:
            raise HTTPException(status_code=413, detail="Analyze at most 5,000 records at a time.")
        if not 2 <= req.n_clusters <= 8:
            raise HTTPException(status_code=400, detail="Cluster count must be between 2 and 8.")

        result = thematic_coder.analyze_corpus(
            records=records_to_analyze,
            distilbert_service=distilbert_service,
            use_hybrid_rules=req.use_hybrid_rules,
            n_clusters=req.n_clusters,
        )
        latest_analysis_cache = result
        return {"ok": True, "data": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.get("/api/samples/{sample_type}")
async def load_or_download_sample(
    sample_type: str,
    download: bool = Query(False),
):
    """Provide downloadable dataset templates (.csv, .xlsx, .json, .xml, .txt)."""
    sample_type = sample_type.lower().strip()

    if sample_type == "xlsx":
        csv_path = SAMPLE_DIR / "sample_thematic_dataset.csv"
        if not csv_path.exists():
            csv_path = SAMPLE_DIR / "sample_alumni_themes.csv"
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Thematic_Dataset")
        content = buf.getvalue()
        if download:
            return Response(
                content=content,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": 'attachment; filename="template_thematic_dataset.xlsx"'},
            )
        parsed = parse_uploaded_file("template_thematic_dataset.xlsx", content)
        return {"ok": True, "data": parsed}

    file_map = {
        "csv": ("sample_thematic_dataset.csv", "text/csv", "template_thematic_dataset.csv"),
        "json": ("sample_thematic_dataset.json", "application/json", "template_thematic_dataset.json"),
        "xml": ("sample_thematic_dataset.xml", "application/xml", "template_thematic_dataset.xml"),
        "txt": ("sample_thematic_dataset.txt", "text/plain", "template_thematic_dataset.txt"),
    }

    if sample_type not in file_map:
        raise HTTPException(status_code=404, detail=f"Unknown template type '{sample_type}'")

    source_fname, media_type, download_fname = file_map[sample_type]
    fpath = SAMPLE_DIR / source_fname
    if not fpath.exists():
        fpath = SAMPLE_DIR / source_fname.replace("sample_thematic_dataset", "sample_alumni_themes")
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Template file {source_fname} not found")

    content = fpath.read_bytes()
    if download:
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{download_fname}"'},
        )

    parsed = parse_uploaded_file(download_fname, content)
    return {"ok": True, "data": parsed}


@app.post("/api/export")
async def export_analysis(req: ExportRequest):
    """Export analyzed records and thematic summary as CSV, XLSX, JSON, or XML."""
    target = req.analysis or latest_analysis_cache
    if not target:
        raise HTTPException(status_code=400, detail="No analysis results available to export.")

    if req.format.lower() not in {"csv", "xlsx", "json", "xml"}:
        raise HTTPException(status_code=400, detail="Unsupported export format.")
    payload, media_type, filename = export_results(target, fmt=req.format)
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"ok": True, "message": "DistilBERT Thematic Analysis API running."})
    return FileResponse(str(index_file))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8050, reload=False)
