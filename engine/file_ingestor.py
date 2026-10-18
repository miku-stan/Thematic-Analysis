"""Multi-Format File Ingestor & Exporter (.csv, .xlsx, .json, .xml, .txt)."""

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


COLUMN_ALIASES: Dict[str, List[str]] = {
    "full_name": ["full_name", "fullname", "name", "respondent", "student_name", "author", "id", "respondent_id", "participant", "user_id"],
    "headline": ["headline", "title", "profile_headline", "tagline", "summary_title"],
    "current_position": ["current_position", "position", "job_title", "role", "current_role", "occupation", "designation"],
    "current_company": ["current_company", "company", "employer", "organization", "institution", "workplace"],
    "degree_program": ["degree_program", "degree", "course", "program", "major", "specialization", "education"],
    "skills": ["skills", "technical_skills", "competencies", "technologies", "tags", "keywords"],
    "about": [
        "about", "bio", "response", "feedback", "comment", "narrative", "text",
        "description", "open_ended", "reflection", "experience_summary", "input_text", "statement",
    ],
}


def auto_detect_mapping(columns: List[str]) -> Dict[str, str]:
    """Guess standard field mappings from a list of file column names."""
    mapping: Dict[str, str] = {}
    lower_to_orig = {col.strip().lower(): col for col in columns if col}

    for target_field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_orig:
                mapping[target_field] = lower_to_orig[alias]
                break

    # If no 'about' or 'headline' was matched, pick the first text-like column as 'about'
    if not any(k in mapping for k in ("headline", "current_position", "about", "skills")) and columns:
        best_col = columns[-1] if len(columns) > 1 else columns[0]
        mapping["about"] = best_col
        if len(columns) > 1 and "full_name" not in mapping:
            mapping["full_name"] = columns[0]

    return mapping


def parse_uploaded_file(filename: str, content: bytes) -> Dict[str, Any]:
    """Parse uploaded file bytes (.csv, .xlsx, .json, .xml, .txt) into structured rows."""
    ext = (filename or "").lower().rsplit(".", 1)[-1] if "." in (filename or "") else ""

    if ext == "csv":
        records, columns = _parse_csv(content)
        file_type = "CSV"
    elif ext == "xlsx":
        records, columns = _parse_excel(content)
        file_type = "Excel (.xlsx)"
    elif ext == "json":
        records, columns = _parse_json(content)
        file_type = "JSON"
    elif ext == "xml":
        records, columns = _parse_xml(content)
        file_type = "XML"
    elif ext in ("txt", "md", "log"):
        records, columns = _parse_txt(content)
        file_type = "Plain Text (.txt)"
    else:
        raise ValueError(
            f"Unsupported file extension '.{ext}'. Supported formats: .csv, .xlsx, .json, .xml, .txt"
        )

    suggested_mapping = auto_detect_mapping(columns)
    return {
        "filename": filename,
        "file_type": file_type,
        "total_rows": len(records),
        "columns": columns,
        "suggested_mapping": suggested_mapping,
        "preview_rows": records[:10],
        "records": records,
    }


def _clean_val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val).strip()


def _parse_csv(content: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = _decode_bytes(content)
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df = df.dropna(how="all")
    columns = [str(c).strip() for c in df.columns.tolist()]
    df.columns = columns
    records = [
        {col: _clean_val(row[col]) for col in columns}
        for _, row in df.iterrows()
        if any(_clean_val(row[col]) for col in columns)
    ]
    return records, columns


def _parse_excel(content: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    df = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False)
    df = df.dropna(how="all")
    columns = [str(c).strip() for c in df.columns.tolist()]
    df.columns = columns
    records = [
        {col: _clean_val(row[col]) for col in columns}
        for _, row in df.iterrows()
        if any(_clean_val(row[col]) for col in columns)
    ]
    return records, columns


def _parse_json(content: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = _decode_bytes(content)
    parsed = json.loads(text)

    raw_list: List[Any] = []
    if isinstance(parsed, list):
        raw_list = parsed
    elif isinstance(parsed, dict):
        def find_records(value, depth=0):
            if depth > 5:
                return None
            if isinstance(value, list):
                if not value or all(isinstance(item, (dict, str)) for item in value):
                    return value
                for item in value:
                    found = find_records(item, depth + 1)
                    if found is not None:
                        return found
            if isinstance(value, dict):
                for key in ("records", "data", "items", "responses", "rows", "profiles", "participants"):
                    if key in value:
                        found = find_records(value[key], depth + 1)
                        if found is not None:
                            return found
                for nested in value.values():
                    found = find_records(nested, depth + 1)
                    if found is not None:
                        return found
            return None

        raw_list = find_records(parsed) or [parsed]

    records: List[Dict[str, Any]] = []
    col_set: List[str] = []

    for idx, item in enumerate(raw_list):
        if isinstance(item, dict):
            cleaned = {str(k).strip(): _clean_val(v) for k, v in item.items()}
            if any(cleaned.values()):
                records.append(cleaned)
                for k in cleaned.keys():
                    if k not in col_set:
                        col_set.append(k)
        elif isinstance(item, str) and item.strip():
            row = {"id": f"Item #{idx + 1}", "response": item.strip()}
            records.append(row)
            for k in ("id", "response"):
                if k not in col_set:
                    col_set.append(k)

    return records, col_set


def _parse_xml(content: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = _decode_bytes(content)
    root = ET.fromstring(text)

    def strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    # Find repeating record nodes: direct children of root or second-level children
    children = list(root)
    record_nodes = children

    # If root has wrapper nodes like <dataset><records><record>...</record></records></dataset>
    if len(children) == 1 and len(list(children[0])) > 0:
        record_nodes = list(children[0])

    records: List[Dict[str, Any]] = []
    col_set: List[str] = []

    for idx, node in enumerate(record_nodes):
        row: Dict[str, Any] = {}
        # Include XML attributes on the record node
        for attr_k, attr_v in node.attrib.items():
            clean_k = strip_ns(attr_k)
            row[clean_k] = _clean_val(attr_v)
            if clean_k not in col_set:
                col_set.append(clean_k)

        sub_elements = list(node)
        if sub_elements:
            for child in sub_elements:
                tag_name = strip_ns(child.tag)
                # Handle nested lists like <skills><skill>Python</skill><skill>SQL</skill></skills>
                grandchildren = list(child)
                if grandchildren:
                    vals = [
                        _clean_val("".join(gc.itertext()))
                        for gc in grandchildren
                        if _clean_val("".join(gc.itertext()))
                    ]
                    val_str = ", ".join(vals)
                else:
                    val_str = _clean_val("".join(child.itertext()))

                if tag_name in row and row[tag_name]:
                    row[tag_name] = f"{row[tag_name]}, {val_str}"
                else:
                    row[tag_name] = val_str

                if tag_name not in col_set:
                    col_set.append(tag_name)
        else:
            # Leaf text node e.g. <response id="1">Text here</response>
            body = _clean_val("".join(node.itertext()))
            if body:
                key = strip_ns(node.tag) or "response"
                row[key] = body
                if key not in col_set:
                    col_set.append(key)

        if any(row.values()):
            if not any(k in row for k in ("id", "full_name", "name")):
                row["record_id"] = f"XML-{idx + 1}"
                if "record_id" not in col_set:
                    col_set.insert(0, "record_id")
            records.append(row)

    return records, col_set


def _parse_txt(content: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = _decode_bytes(content)
    # Blank lines mark multi-line transcript responses; otherwise each line is a record.
    if "\n\n" in text.replace("\r\n", "\n"):
        lines = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [line for line in lines if not line.startswith("#")]

    records: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        # Check if line uses "Name :: Text" or "Headline | Position | ..."
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            records.append({
                "full_name": parts[0] if len(parts) > 1 else f"Line #{idx + 1}",
                "response": " | ".join(parts[1:]) if len(parts) > 1 else parts[0],
            })
        else:
            records.append({
                "full_name": f"Entry #{idx + 1}",
                "response": line,
            })

    return records, ["full_name", "response"]


def _decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def apply_column_mapping(
    records: List[Dict[str, Any]],
    mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Map user-selected file columns onto standard DistilBERT + Thematic Analysis fields."""
    if not mapping:
        return records

    normalized_records: List[Dict[str, Any]] = []
    for row in records:
        new_row = dict(row)
        for target_field, source_col in mapping.items():
            if source_col and source_col in row:
                new_row[target_field] = row[source_col]
        normalized_records.append(new_row)
    return normalized_records


def export_results(analysis_result: Dict[str, Any], fmt: str = "csv") -> Tuple[bytes, str, str]:
    """Export analyzed records and thematic summary into CSV, XLSX, JSON, or XML."""
    fmt_clean = (fmt or "csv").lower().strip()
    records = analysis_result.get("records", [])

    flat_rows = []
    for r in records:
        probs = r.get("probabilities") or {}
        flat_rows.append({
            "Row": r.get("row_index"),
            "Identifier": r.get("identifier"),
            "Primary_Theme": r.get("primary_theme"),
            "Secondary_Theme": r.get("secondary_theme") or "",
            "Theme_Confidence": r.get("theme_confidence"),
            "Matched_Codes": ", ".join(r.get("matched_codes") or []),
            "Emergent_Cluster": r.get("emergent_cluster") or "",
            "Trajectory_Tone": r.get("trajectory_tone") or "",
            "DistilBERT_Prediction": r.get("distilbert_label"),
            "Final_Outcome_Status": r.get("final_status"),
            "DistilBERT_Confidence": r.get("confidence"),
            "Prob_Employed": probs.get("employed", 0.0),
            "Prob_Underemployed": probs.get("underemployed", 0.0),
            "Prob_Unemployed": probs.get("unemployed", 0.0),
            "Degree_Aligned": r.get("alignment", {}).get("aligned", False),
            "Alignment_Reason": r.get("alignment", {}).get("reason", ""),
            "Token_Count": r.get("token_count"),
            "Latency_ms": r.get("latency_ms"),
            "Standardized_Input": r.get("input_text"),
        })

    if fmt_clean == "json":
        payload = json.dumps(analysis_result, indent=2, ensure_ascii=False).encode("utf-8")
        return payload, "application/json", "distilbert_thematic_analysis.json"

    if fmt_clean == "xml":
        root = ET.Element("ThematicAnalysisReport")
        summary_el = ET.SubElement(root, "Summary")
        for k, v in (analysis_result.get("summary") or {}).items():
            child = ET.SubElement(summary_el, str(k))
            child.text = str(v)

        themes_el = ET.SubElement(root, "ThemeDistribution")
        for t in analysis_result.get("theme_distribution", []):
            t_el = ET.SubElement(themes_el, "Theme")
            for k in ("theme", "count", "percentage", "employed", "underemployed", "unemployed"):
                c = ET.SubElement(t_el, k)
                c.text = str(t.get(k, ""))

        records_el = ET.SubElement(root, "CodedRecords")
        for row in flat_rows:
            rec_el = ET.SubElement(records_el, "Record")
            for k, v in row.items():
                field_el = ET.SubElement(rec_el, k)
                field_el.text = str(v if v is not None else "")

        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        return xml_bytes, "application/xml", "distilbert_thematic_analysis.xml"

    if fmt_clean in ("xlsx", "excel"):
        output = io.BytesIO()
        df_records = pd.DataFrame(flat_rows)
        df_themes = pd.DataFrame(analysis_result.get("theme_distribution", []))
        df_summary = pd.DataFrame([analysis_result.get("summary", {})])
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_records.to_excel(writer, index=False, sheet_name="Coded_Records")
            df_themes.to_excel(writer, index=False, sheet_name="Theme_Distribution")
            df_summary.to_excel(writer, index=False, sheet_name="Executive_Summary")
        return (
            output.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "distilbert_thematic_analysis.xlsx",
        )

    # Default: CSV
    output_str = io.StringIO()
    if flat_rows:
        writer = csv.DictWriter(output_str, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    return output_str.getvalue().encode("utf-8-sig"), "text/csv", "distilbert_thematic_analysis.csv"
