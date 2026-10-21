"""Multi-Format File Ingestor & Exporter (.csv, .xlsx, .json, .xml, .txt)."""

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


COLUMN_ALIASES: Dict[str, List[str]] = {
    "full_name": [
        "participant_id", "respondent_id", "case_id", "stakeholder_id", "case", "respondent",
        "participant", "lce_code", "full_name", "fullname", "name", "student_name", "author", "id", "user_id"
    ],
    "headline": [
        "subtheme", "theme_heading", "finding", "headline", "title", "profile_headline", "tagline", "summary_title"
    ],
    "current_position": [
        "office_role", "governance_role", "designation", "office", "department", "position",
        "current_position", "job_title", "role", "current_role", "occupation"
    ],
    "current_company": [
        "lgu", "municipality", "city", "province", "agency", "current_company", "company",
        "employer", "organization", "institution", "workplace"
    ],
    "degree_program": [
        "governance_area", "sglg_area", "pillar", "assessment_area", "degree_program",
        "degree", "course", "program", "major", "specialization", "education"
    ],
    "skills": [
        "skills_indicators", "indicators", "operational_issue", "subthemes", "skills",
        "technical_skills", "competencies", "technologies", "tags", "keywords"
    ],
    "about": [
        "narrative_quote", "quotation", "quote", "excerpt", "interview_transcript", "findings",
        "observation", "statement", "about", "bio", "response", "feedback", "comment", "narrative",
        "text", "description", "open_ended", "reflection", "experience_summary", "input_text"
    ],
    "respondent_group": [
        "respondent_group", "stakeholder_group", "group", "stakeholder_category",
        "tripartite_group", "respondent_category"
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
    """Parse uploaded file bytes (.csv, .xlsx, .json, .xml, .txt, .docx) into structured rows."""
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
    elif ext == "docx":
        records, columns = _parse_docx(content, filename=filename)
        file_type = "Word Transcript (.docx)"
    elif ext == "doc":
        records, columns = _parse_doc(content, filename=filename)
        file_type = "Word Transcript (.doc)"
    elif ext == "pdf":
        records, columns = _parse_pdf(content, filename=filename)
        file_type = "PDF Transcript (.pdf)"
    else:
        raise ValueError(
            f"Unsupported file extension '.{ext}'. Supported formats: .csv, .xlsx, .json, .xml, .txt, .docx, .doc, .pdf"
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


def parse_multiple_uploaded_files(file_items: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """Parse multiple uploaded files (e.g. multiple .docx transcripts) and merge into a unified dataset."""
    if not file_items:
        raise ValueError("No files provided for multi-file upload.")

    if len(file_items) == 1:
        return parse_uploaded_file(file_items[0][0], file_items[0][1])

    all_records: List[Dict[str, Any]] = []
    seen_columns: List[str] = []
    file_names: List[str] = []

    for fname, content in file_items:
        parsed = parse_uploaded_file(fname, content)
        file_names.append(fname)
        for col in parsed.get("columns", []):
            if col not in seen_columns:
                seen_columns.append(col)
        all_records.extend(parsed.get("records", []))

    suggested_mapping = auto_detect_mapping(seen_columns)
    combined_name = f"Multi-Case Transcripts ({len(file_items)} files: {', '.join(file_names[:3])}{'...' if len(file_names) > 3 else ''})"

    return {
        "filename": combined_name,
        "file_type": f"Multi-File ({len(file_items)} transcripts)",
        "total_rows": len(all_records),
        "columns": seen_columns,
        "suggested_mapping": suggested_mapping,
        "preview_rows": all_records[:10],
        "records": all_records,
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


def _extract_indicators(text: str) -> str:
    """Extract salient qualitative governance keywords/indicators from an interview excerpt."""
    lower = text.lower()
    candidates = [
        "documentation", "records", "budget", "funding", "accessibility", "ordinance",
        "coordination", "focal person", "dilg", "technical assistance", "institutionalization",
        "biometrics", "monitoring", "capacity", "turnover", "plantilla", "service delivery"
    ]
    found = [c for c in candidates if c in lower]
    return ", ".join(found) if found else "sglg governance"


def _parse_transcript_paragraphs(paras: List[str], filename: str = "") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse raw paragraphs of interview transcripts into structured qualitative records."""
    fname_lower = (filename or "").lower()

    # Determine municipality and case code
    if "ambaguio" in fname_lower:
        lgu_name = "Ambaguio"
        case_code = "LGU-01"
    elif "penablanca" in fname_lower or "peñablanca" in fname_lower:
        lgu_name = "Peñablanca"
        case_code = "LGU-02"
    else:
        # Check in body text
        joined_lower = " ".join(paras[:10]).lower()
        if "ambaguio" in joined_lower:
            lgu_name = "Ambaguio"
            case_code = "LGU-01"
        elif "penablanca" in joined_lower or "peñablanca" in joined_lower:
            lgu_name = "Peñablanca"
            case_code = "LGU-02"
        else:
            lgu_name = "Municipality"
            case_code = "LGU-01"

    lgu_display = f"{lgu_name} ({case_code})"

    if "mayor" in fname_lower:
        default_role = "Local Chief Executive / Mayor"
    elif "mlgoo" in fname_lower:
        default_role = "MLGOO"
    elif "focal" in fname_lower:
        default_role = "SGLG Focal Person"
    else:
        default_role = "LGU Participant"

    speaker_re = re.compile(
        r"^(INTERVIEWER|Interviewer|Interviewer\s*\d*|Question\s*no\.?\s*\d+|-Question\s*no\.?\s*\d+|SGLG\s*FOCAL(?:\s*PERSON)?|SECTOR\s*LEADER|MLGOO|DILG\s*FOCAL(?:\s*PERSON)?|MAYOR|MAYORS|Local\s*Chief\s*Executives?|LCE)\s*[:\-]\s*(.*)",
        re.IGNORECASE,
    )

    records: List[Dict[str, Any]] = []
    current_speaker: Optional[str] = None
    current_prompt = ""
    current_response_paras: List[str] = []
    in_question_mode = False
    current_question_paras: List[str] = []

    def flush_turn():
        nonlocal current_response_paras, current_speaker, current_prompt
        text = " ".join(current_response_paras).strip()
        if text:
            words = text.split()
            # Filter brief filler turns (less than 15 words)
            if len(words) >= 15:
                spk_clean = (current_speaker or default_role).strip().title()
                if "Mayor" in spk_clean:
                    spk_clean = "Mayor / LCE"
                elif "Mlgoo" in spk_clean:
                    spk_clean = "MLGOO"
                elif "Focal" in spk_clean:
                    spk_clean = "SGLG Focal Person"
                elif "Chief Executive" in spk_clean:
                    spk_clean = "Local Chief Executive"
                elif "Sector" in spk_clean:
                    spk_clean = "Sector Leader"

                turn_num = len(records) + 1
                records.append({
                    "participant_id": f"{lgu_name} - {spk_clean} (Turn #{turn_num})",
                    "subtheme": current_prompt or f"Interview Excerpt #{turn_num}",
                    "office_role": spk_clean,
                    "lgu": lgu_display,
                    "governance_area": "SGLG Local Governance Study 3",
                    "skills_indicators": _extract_indicators(text),
                    "narrative_quote": text,
                })
        current_response_paras = []

    for p in paras:
        m = speaker_re.match(p)
        if m:
            label = m.group(1).strip()
            content = m.group(2).strip()

            if any(k in label.lower() for k in ["interviewer", "question"]):
                flush_turn()
                in_question_mode = True
                current_question_paras = [content] if content else []
                current_speaker = None
            else:
                if in_question_mode:
                    current_prompt = " ".join(current_question_paras).strip()
                    in_question_mode = False
                    current_question_paras = []
                else:
                    flush_turn()
                current_speaker = label
                current_response_paras = [content] if content else []
        else:
            if in_question_mode:
                is_answer_start = re.match(
                    r"^(I\s+believe|For\s+me|Siguro|Sa\s+tingin|Unang\s+una|Actually|Meron|Wala|Nung|Noong)\b",
                    p,
                    re.IGNORECASE,
                )
                if is_answer_start and len(current_question_paras) > 0:
                    current_prompt = " ".join(current_question_paras).strip()
                    in_question_mode = False
                    current_question_paras = []
                    current_speaker = default_role
                    current_response_paras = [p]
                else:
                    current_question_paras.append(p)
            else:
                if current_speaker is None:
                    if re.match(r"^(Thank\s+you|Maraming\s+salamat|Salamat|Ok\s+po)\b", p, re.IGNORECASE):
                        in_question_mode = True
                        current_question_paras = [p]
                    else:
                        current_speaker = default_role
                        current_response_paras.append(p)
                else:
                    current_response_paras.append(p)

    flush_turn()

    # Fallback if no speaker turns detected (e.g. generic document paragraphs)
    if not records:
        for idx, p in enumerate(paras):
            if len(p.split()) >= 8:
                records.append({
                    "participant_id": f"{lgu_name} - Excerpt #{idx + 1}",
                    "subtheme": "Transcript Excerpt",
                    "office_role": default_role,
                    "lgu": lgu_display,
                    "governance_area": "SGLG Local Governance Study 3",
                    "skills_indicators": _extract_indicators(p),
                    "narrative_quote": p,
                })

    columns = [
        "participant_id",
        "subtheme",
        "office_role",
        "lgu",
        "governance_area",
        "skills_indicators",
        "narrative_quote",
    ]
    return records, columns


def _parse_docx(content: bytes, filename: str = "") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse Microsoft Word (.docx) interview transcript into speaker turns and qualitative records."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            xml_content = z.read("word/document.xml")
    except Exception as exc:
        raise ValueError(f"Unable to read .docx archive: {exc}")

    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        root = ET.fromstring(xml_content)
    except Exception as exc:
        raise ValueError(f"Unable to parse document XML in .docx: {exc}")

    paras: List[str] = []
    for p in root.iterfind(".//w:p", namespaces):
        t = "".join([n.text for n in p.iterfind(".//w:t", namespaces) if n.text]).strip()
        if t:
            paras.append(t)

    return _parse_transcript_paragraphs(paras, filename=filename)


def _parse_doc(content: bytes, filename: str = "") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse Microsoft Word (.doc) document, HTML-based Word, or RTF transcript."""
    # 1. Check if it's a .docx zip file renamed to .doc
    if content.startswith(b"PK\x03\x04"):
        return _parse_docx(content, filename=filename)

    decoded = _decode_bytes(content)

    # 2. Check if it is HTML-formatted Word document
    if "<html" in decoded.lower() or "<w:worddocument" in decoded.lower():
        clean = re.sub(r"(?i)<(p|br|tr|div|h[1-6])[^>]*>", "\n\n", decoded)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = (
            clean.replace("&quot;", '"')
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&rsquo;", "'")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
        )
        paras = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
        return _parse_transcript_paragraphs(paras, filename=filename)

    # 3. Check if it is RTF
    if content.startswith(b"{\\rtf"):
        clean = re.sub(r"\\[a-zA-Z0-9\-]+ ?|\{|\}", " ", decoded)
        paras = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
        return _parse_transcript_paragraphs(paras, filename=filename)

    # 4. Fallback for binary Word 97-2003 (extract textual strings)
    strings = re.findall(r"[\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF]{15,}", decoded)
    clean_paras = [
        s.strip()
        for s in strings
        if not s.strip().startswith("Microsoft") and not s.strip().startswith("Word.Document")
    ]
    if clean_paras:
        return _parse_transcript_paragraphs(clean_paras, filename=filename)

    raise ValueError("Unable to extract text from .doc file. Please save as .docx or .pdf.")


def _parse_pdf(content: bytes, filename: str = "") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse PDF interview transcript into structured records using PyPDF2."""
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages_text: List[str] = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages_text.append(txt)
        full_text = "\n\n".join(pages_text).strip()
    except Exception as exc:
        raise ValueError(f"Unable to read PDF file: {exc}")

    if not full_text:
        raise ValueError("The uploaded PDF file contains no readable text (it may be a scanned image or empty).")

    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
    if len(raw_paras) <= 1:
        raw_paras = [line.strip() for line in full_text.splitlines() if line.strip()]

    return _parse_transcript_paragraphs(raw_paras, filename=filename)


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
            "Respondent_Group": r.get("respondent_group", ""),
            "Taglish_Detected": "Yes" if r.get("is_taglish") else "No",
            "Primary_Theme": r.get("primary_theme"),
            "Secondary_Theme": r.get("secondary_theme") or "",
            "Theme_Confidence": r.get("theme_confidence"),
            "Matched_Codes": ", ".join(r.get("matched_codes") or []),
            "Emergent_Cluster": r.get("emergent_cluster") or "",
            "Trajectory_Tone": r.get("trajectory_tone") or "",
            "DistilBERT_Prediction": r.get("distilbert_label"),
            "Governance_Status": r.get("governance_status") or r.get("final_status"),
            "Governance_Trajectory": r.get("governance_status") or r.get("final_status"),
            "Prob_Sustained": probs.get("sustained", 0.0),
            "Prob_Constrained": probs.get("constrained", 0.0),
            "Prob_Deficient": probs.get("deficient", 0.0),
            "Institutionalized": r.get("alignment", {}).get("aligned", False),
            "Trajectory_Rationale": r.get("alignment", {}).get("reason", ""),
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
            for k in ("theme", "count", "percentage", "sustained", "constrained", "deficient"):
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
            cross_case = analysis_result.get("cross_case_matrix")
            if cross_case:
                matrix_rows = cross_case.get("rows") if isinstance(cross_case, dict) else cross_case
                if matrix_rows:
                    pd.DataFrame(matrix_rows).to_excel(writer, index=False, sheet_name="Cross_Case_Matrix")
            rgb = analysis_result.get("respondent_group_breakdown")
            if rgb and rgb.get("groups") and rgb.get("themes"):
                breakdown_rows = []
                for theme_name in rgb["themes"]:
                    row_dict = {"Theme": theme_name}
                    for group in rgb["groups"]:
                        row_dict[group] = rgb["counts"].get(group, {}).get(theme_name, 0)
                    breakdown_rows.append(row_dict)
                pd.DataFrame(breakdown_rows).to_excel(writer, index=False, sheet_name="Respondent_Group_Breakdown")
        return (
            output.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "distilbert_thematic_analysis.xlsx",
        )

    if fmt_clean in ("doc", "docx"):
        html = _generate_word_export(analysis_result, flat_rows)
        return html.encode("utf-8"), "application/msword", "distilbert_thematic_analysis_report.doc"

    # Default: CSV
    output_str = io.StringIO()
    if flat_rows:
        writer = csv.DictWriter(output_str, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    return output_str.getvalue().encode("utf-8-sig"), "text/csv", "distilbert_thematic_analysis.csv"


def _generate_word_export(analysis_result: Dict[str, Any], flat_rows: List[Dict[str, Any]]) -> str:
    """Generate professional Microsoft Word (.doc) report for Tab 2 Dashboard and Tab 3 Coded Corpus."""
    summary = analysis_result.get("summary") or {}
    themes = analysis_result.get("theme_distribution") or []
    cross_case = analysis_result.get("cross_case_matrix")
    matrix_rows = cross_case.get("rows") if isinstance(cross_case, dict) else (cross_case or [])

    total_records = summary.get("total_records", len(flat_rows))
    dominant_theme = summary.get("dominant_theme", "—")
    inst_rate = summary.get("employment_rate", 0.0)
    mean_conf = summary.get("mean_confidence", 0.0)

    html_parts = [
        """<!DOCTYPE html>
<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
  <meta charset='utf-8'>
  <title>SGLG Thematic Analysis Synthesis Report</title>
  <style>
    body { font-family: 'Calibri', 'Times New Roman', serif; font-size: 11pt; line-height: 1.5; color: #111827; }
    h1 { font-size: 18pt; color: #1e293b; margin-bottom: 4pt; border-bottom: 2pt solid #334155; padding-bottom: 4pt; }
    h2 { font-size: 14pt; color: #334155; margin-top: 18pt; margin-bottom: 8pt; border-bottom: 1pt solid #cbd5e1; padding-bottom: 3pt; }
    p.lead { font-size: 12pt; color: #475569; margin-top: 0; margin-bottom: 14pt; font-style: italic; }
    table { width: 100%; border-collapse: collapse; margin-top: 8pt; margin-bottom: 16pt; font-size: 10pt; }
    th { background-color: #f1f5f9; color: #0f172a; font-weight: bold; padding: 7pt 9pt; border: 1pt solid #cbd5e1; text-align: left; }
    td { padding: 6pt 9pt; border: 1pt solid #cbd5e1; vertical-align: top; }
    tr:nth-child(even) td { background-color: #f8fafc; }
    .kpi-table th { background-color: #e2e8f0; }
    .badge { display: inline-block; padding: 2pt 6pt; font-size: 9pt; font-weight: bold; border-radius: 3pt; }
    .badge-sustained { background-color: #dcfce7; color: #166534; }
    .badge-constrained { background-color: #fef9c3; color: #854d0e; }
    .badge-deficient { background-color: #fee2e2; color: #991b1b; }
    .quote-box { background-color: #f8fafc; border-left: 3pt solid #475569; padding: 8pt 12pt; font-style: italic; margin: 6pt 0; }
  </style>
</head>
<body>
  <h1>SGLG Local Governance Thematic Analysis: Empirical Studio Report</h1>
  <p class="lead">DistilBERT Sequence Inference &amp; Qualitative Thematic Synthesis · Study 3 (Moving Beyond Paper Compliance)</p>

  <h2>1. Executive Analytics &amp; Institutionalization Overview</h2>
  <table class="kpi-table">
    <tr>
      <th>Metric</th>
      <th>Value</th>
      <th>Methodological Description</th>
    </tr>
    <tr>
      <td><strong>Total Analyzed Excerpts</strong></td>
      <td><strong>"""
        + str(total_records)
        + """</strong></td>
      <td>Interview quotes, field observations, and transcripts processed by DistilBERT</td>
    </tr>
    <tr>
      <td><strong>Dominant Emergent Theme</strong></td>
      <td><strong>"""
        + str(dominant_theme)
        + """</strong></td>
      <td>Most frequent governance pattern across participating LGUs</td>
    </tr>
    <tr>
      <td><strong>Governance Institutionalization Rate</strong></td>
      <td><strong>"""
        + str(inst_rate)
        + """%</strong></td>
      <td>Proportion of practices sustained outside the formal DILG assessment cycle</td>
    </tr>
    <tr>
      <td><strong>Mean Softmax Confidence</strong></td>
      <td><strong>"""
        + str(mean_conf)
        + """%</strong></td>
      <td>Average multi-class probability assigned by 6-layer DistilBERT encoder</td>
    </tr>
  </table>

  <h2>2. Theme Frequency &amp; Percentage Distribution</h2>
  <table>
    <tr>
      <th style="width:50%;">Core Governance Theme</th>
      <th style="width:15%; text-align:center;">Count</th>
      <th style="width:15%; text-align:center;">Percentage</th>
      <th style="width:20%;">Institutional Trajectory</th>
    </tr>
"""
    ]

    for t in themes:
        html_parts.append(
            f"""    <tr>
      <td><strong>{t.get('theme', '—')}</strong></td>
      <td style="text-align:center;">{t.get('count', 0)}</td>
      <td style="text-align:center;">{t.get('percentage', 0)}%</td>
      <td>Sustained: {t.get('sustained', 0)} | Constrained: {t.get('constrained', 0)} | Deficient: {t.get('deficient', 0)}</td>
    </tr>"""
        )

    html_parts.append(
        """  </table>
"""
    )

    if matrix_rows:
        html_parts.append(
            """  <h2>3. Theme 3: Cross-Case Operational Barriers Comparison Matrix</h2>
  <table>
    <tr>
      <th>Operational Barrier / Dimension</th>
      <th>LGU-01 (Ambaguio)</th>
      <th>LGU-02 (Peñablanca)</th>
      <th>LGU-03</th>
      <th>Institutional Adaptation</th>
    </tr>
"""
        )
        for r in matrix_rows:
            html_parts.append(
                f"""    <tr>
      <td><strong>{r.get('dimension') or r.get('Barrier') or 'Dimension'}</strong></td>
      <td>{r.get('lgu_01') or r.get('LGU-01') or '—'}</td>
      <td>{r.get('lgu_02') or r.get('LGU-02') or '—'}</td>
      <td>{r.get('lgu_03') or r.get('LGU-03') or '—'}</td>
      <td><em>{r.get('adaptation') or r.get('Institutional Adaptation') or '—'}</em></td>
    </tr>"""
            )
        html_parts.append(
            """  </table>
"""
        )

    html_parts.append(
        """  <h2>4. Coded Corpus &amp; Deep Evidence Database</h2>
  <table>
    <tr>
      <th style="width:12%;">Identifier &amp; Role</th>
      <th style="width:12%;">Case LGU</th>
      <th style="width:22%;">Primary Assigned Theme</th>
      <th style="width:10%; text-align:center;">Confidence</th>
      <th style="width:44%;">Verbatim Qualitative Quote Excerpt</th>
    </tr>
"""
    )

    for r in flat_rows:
        conf = r.get("Theme_Confidence")
        conf_str = f"{conf:.1f}%" if isinstance(conf, (int, float)) else str(conf or "—")
        quote = (r.get("Standardized_Input") or "").replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f"""    <tr>
      <td><strong>{r.get('Identifier', '—')}</strong><br><small style="color:#64748b;">{r.get('Office_Role', '')}</small></td>
      <td>{r.get('LGU_Case', '—')}</td>
      <td><strong>{r.get('Primary_Theme', '—')}</strong><br><small style="color:#475569;">Trajectory: {r.get('Governance_Status', '—')}</small></td>
      <td style="text-align:center;">{conf_str}</td>
      <td><div class="quote-box">“{quote}”</div></td>
    </tr>"""
        )

    html_parts.append(
        """  </table>
</body>
</html>"""
    )
    return "".join(html_parts)
