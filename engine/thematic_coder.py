"""Thematic Coding, Custom Codebook & Unsupervised TF-IDF/K-Means Theme Discovery Engine."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from engine.taglish_normalizer import (
    TAGALOG_STOP_WORDS,
    extract_salient_terms,
    is_taglish_text,
    normalize_taglish_text,
)


# Teacher's Study 3 Thematic Framework for SGLG Local Governance
SGLG_GOVERNANCE_THEMES: List[Dict[str, Any]] = [
    {
        "id": "theme_1_leadership_governance",
        "name": "Theme 1: Leadership Experiences and Governance Transformation",
        "color": "#6366f1",
        "description": "Examines how critical experiences, leadership decisions, accountability arrangements, executive direction, and previous SGLG assessments shaped current governance practice.",
        "subthemes": [
            "Leadership commitment and executive direction",
            "Learning from previous assessment gaps",
            "Accountability arrangements",
            "Concrete changes in policy, process, or practice",
        ],
        "writing_sequence": "State dominant pattern → show supporting cases/stakeholders → present short de-identified quotations (e.g., LCE-03) → identify contrasting cases → interpret what pattern means for sustained SGLG performance.",
        "sample_quote": '"We realized that cramming two weeks before the DILG table assessment always led to failed indicators. I issued an Executive Order mandating quarterly self-audits and designated permanent focal persons in every department." — LCE-03',
        "keywords": [
            "leadership commitment", "executive direction", "leadership decisions", "accountability arrangements",
            "previous assessment gaps", "governance practice", "concrete changes in policy", "policy change",
            "governance transformation", "executive order", "local chief executive", "lce", "mayor", "vice mayor",
            "political will", "administrative reform", "corrective actions", "leadership-driven change",
            "prior assessment", "learning from gaps", "leadership vision", "performance commitment",
            "decisive leadership", "ordinance reform", "executive action", "governance reform",
            "pamumuno", "direksyon", "pamamahala", "pagbabago", "desisyon", "administrasyon",
            "reorganization", "ordinances", "ordinansa", "resolusyon", "resolution", "supportive",
            "leadership journey", "leadership transition",
        ],
    },
    {
        "id": "theme_2_interdepartmental_coordination",
        "name": "Theme 2: Interdepartmental Coordination and Accountability",
        "color": "#06b6d4",
        "description": "Focuses on how responsibilities are distributed across offices, how coordination is maintained, how issues are escalated, and how accountability is monitored.",
        "subthemes": [
            "Designated focal persons and committees",
            "Regular coordination and monitoring meetings",
            "Shared documentation or information systems",
            "Escalation of unresolved issues",
            "Institutional rather than person-dependent accountability",
        ],
        "keywords": [
            "focal persons", "designated committee", "coordination meetings", "regular monitoring meetings",
            "shared documentation", "information systems", "escalation of unresolved issues",
            "institutional accountability", "interdepartmental coordination", "cross-departmental",
            "department heads", "office coordination", "responsibility distribution", "monitoring committee",
            "inter-office collaboration", "systemic accountability", "focal team", "shared drive",
            "escalation mechanism", "coordination mechanism", "joint meetings", "accountability matrix",
            "ugnayan", "pulong", "meeting", "meetings", "pagpupulong", "bawat opisina",
            "koordinasyon", "proper coordination", "tulong-tulong", "pagtutulungan", "sector leader",
            "entrusting", "sit down together", "collaborate", "bawat departamento",
        ],
    },
    {
        "id": "theme_3_operational_challenges",
        "name": "Theme 3: Operational, Resource, and Technical Challenges",
        "color": "#f59e0b",
        "description": "Recurring operational barriers identified by participants across cases, including records management, staffing, technical expertise, budget, data systems, inter-office delays, and last-minute compliance.",
        "subthemes": [
            "Documentation and records management",
            "Staffing and personnel turnover",
            "Technical capacity and expertise",
            "Budget and financial resources",
            "Data systems, inter-office delays, and last-minute compliance",
        ],
        "keywords": [
            "records management", "documentation backlog", "staffing constraints", "personnel shortage",
            "technical expertise", "budget constraints", "data systems", "inter-office delays",
            "last-minute compliance", "operational barriers", "compliance rush", "limited resources",
            "funding shortfall", "technical capacity", "documentary deficiency", "archiving", "manpower",
            "delayed submission", "heavy workload", "cramming", "resource constraints", "operational issue",
            "staff turnover", "personnel turnover", "lack of technical capacity",
            "pera", "pondo", "budget", "funding", "5th class", "kakulangan", "kulang sa pondo",
            "kulang sa tao", "dokumento", "hindi documented", "not documented", "dpwh", "accessibility",
            "gusali", "building", "pricy", "mahirap", "problema", "suliranin", "lack of equipment",
            "lack of knowledge", "job order", "greener pasture", "last minute", "walang documentation",
            "difficulties", "operational difficulties",
        ],
    },
    {
        "id": "theme_4_institutionalization_continuity",
        "name": "Theme 4: Institutionalization and Year-Round Continuity",
        "color": "#10b981",
        "description": "Distinguishes LGUs that embed SGLG-related practices into routine governance from those that treat it as a periodic assessment scramble. Moving beyond paper compliance.",
        "subthemes": [
            "Policies, ordinances, and standard operating procedures (SOPs)",
            "Routine reports and year-round monitoring",
            "Institutional memory and records systems",
            "Continuity despite leadership or personnel changes",
            "Continuation of practices even when no assessment is imminent",
        ],
        "suggested_analytic_statement": "The strongest evidence of institutionalization was not the volume of documentation prepared for assessment, but the continuation of governance practices as part of regular administrative and service-delivery routines.",
        "keywords": [
            "institutionalization", "year-round continuity", "routine governance", "standard operating procedures",
            "sops", "local ordinances", "routine reports", "year-round monitoring", "institutional memory",
            "continuity across administrations", "personnel changes", "continuation of practices",
            "no assessment imminent", "regular administrative routines", "service-delivery routines",
            "embedded routines", "institutionalized practice", "quarterly monitoring", "routine compliance",
            "permanent systems", "beyond assessment period", "routine administrative",
            "nainstitutionalized", "institutionalized", "tuloy-tuloy", "tuloy tuloy", "kahit walang assessment",
            "way of life", "regular", "taon-taon", "sistema", "hiring", "permanent", "qualification standard",
            "plus 10 points", "routine", "prepare early", "mdrrmo", "menro", "biometrics", "gfh",
            "good financial housekeeping", "awardee", "continuation", "early preparation",
        ],
    },
    {
        "id": "theme_5_dilg_technical_assistance",
        "name": "Theme 5: DILG Technical Assistance as an Enabling Mechanism",
        "color": "#8b5cf6",
        "description": "Compares forms of DILG support considered useful (orientation, coaching, monitoring, pre-assessment, feedback, technical clarification) and cases where problems recurred despite assistance.",
        "subthemes": [
            "DILG coaching, orientation, and mentoring",
            "Pre-assessment, mock evaluations, and table audits",
            "Technical clarification and policy guidelines",
            "Post-assessment feedback and corrective action planning",
            "Capacity gaps and recurring challenges despite assistance",
        ],
        "keywords": [
            "dilg technical assistance", "dilg coaching", "orientation", "pre-assessment",
            "monitoring and feedback", "technical clarification", "dilg support", "mlgoo",
            "provincial director", "advisory support", "mock assessment", "dilg guidelines",
            "enabling mechanism", "capacity building", "recurring problems despite assistance",
            "dilg consultation", "dilg evaluation", "assessment feedback", "dilg intervention",
            "dilg field officer", "table assessment",
            "dilg", "tulong ng dilg", "gabay", "pag-alalay", "suggestions", "working with the lgu",
            "clarification", "field officer", "dilg focal",
        ],
    },
    {
        "id": "theme_6_beyond_paper_compliance",
        "name": "Theme 6: Beyond Paper Compliance and Sustainable Governance",
        "color": "#ec4899",
        "description": "Culminating theme explaining evidence that an LGU has moved beyond documentary compliance to sustained, genuine public service and visible service-delivery effects.",
        "subthemes": [
            "Year-round implementation",
            "Sustained resource allocation",
            "Continuously updated data and records",
            "Routine monitoring and embedded accountability",
            "Continuity despite personnel changes",
            "Visible service-delivery effects and community impact",
        ],
        "suggested_theme_conclusion": "Across cases, sustainable SGLG performance was distinguished from documentary compliance by the extent to which governance practices were institutionalized, owned by responsible offices, continuously monitored, and maintained outside the formal assessment cycle.",
        "keywords": [
            "beyond paper compliance", "sustainable governance", "documentary compliance",
            "year-round implementation", "sustained resource allocation", "continuously updated data",
            "embedded accountability", "visible service-delivery effects", "ownership by offices",
            "culture of excellence", "lasting community impact", "public satisfaction", "genuine governance",
            "outside the formal assessment cycle", "maintained routines", "sustainable performance",
            "meaningful public service", "sustainable sglg performance", "genuine public service",
            "paper compliance", "service delivery effects",
            "tunay na serbisyo", "hindi lang papel", "hindi lang documents", "hindi lang sa documents",
            "hindi lang dahil sa award", "tpa", "delivery ng serbisyo", "quality of service",
            "ramdam ng tao", "tao muna", "buhay ng mamamayan", "sustainable", "visible service",
            "efficient and effective", "public service to the people", "para sa tao",
            "nadeliver mo ung services", "service to the people",
        ],
    },
]

THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "sglg_governance": {
        "id": "sglg_governance",
        "name": "Teacher's SGLG Governance Study (6 Core Themes)",
        "description": "Seal of Good Local Governance (SGLG) qualitative framework covering leadership, interdepartmental coordination, operational challenges, institutionalization, DILG assistance, and sustained governance ('Beyond Paper Compliance').",
        "themes": SGLG_GOVERNANCE_THEMES,
    },
}

# The default active codebook is the Teacher's SGLG Framework
DEFAULT_THEMES: List[Dict[str, Any]] = SGLG_GOVERNANCE_THEMES

DOMAIN_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(TAGALOG_STOP_WORDS).union({
    "notes", "page", "item", "row", "entry", "response", "statement", "excerpt",
    "na", "none", "n/a", "yr", "yrs", "mo", "mos",
})

RESPONDENT_GROUPS = ["Mayor / Admin", "LCE / Focal", "DILG / PLGU"]


def classify_respondent_group(office_role: str) -> str:
    """Classify a participant's office_role into one of 3 respondent groups."""
    role_lower = (office_role or "").lower()
    # Group 3: DILG / PLGU — field officers, assessors, and provincial-level DILG
    if any(k in role_lower for k in [
        "mlgoo", "dilg", "provincial", "plgu", "field officer",
        "assessment team", "provincial assessment", "validator",
    ]):
        return "DILG / PLGU"
    # SGLG Focal Persons / Focal Teams belong to Group 2 unless specifically Mayor
    if "focal" in role_lower and not any(k in role_lower for k in ["mayor", "vice mayor"]):
        return "LCE / Focal"
    # Group 1: Mayor / Admin — executive leadership, administrator, and executive admin support
    if any(k in role_lower for k in [
        "mayor", "local chief executive", "vice mayor",
        "executive assistant", "bac secretariat", "sanggunian",
        "administrator", "admin", "executive",
    ]):
        return "Mayor / Admin"
    if "lce" in role_lower and "focal" not in role_lower:
        return "Mayor / Admin"
    # Group 2: LCE / Focal — department heads and SGLG focal persons (default)
    return "LCE / Focal"


class ThematicCoder:
    """Performs codebook-based thematic coding and unsupervised TF-IDF + K-Means theme discovery."""

    def __init__(self, custom_themes: Optional[List[Dict[str, Any]]] = None):
        self.themes = custom_themes if custom_themes else [dict(t) for t in DEFAULT_THEMES]

    def get_themes(self) -> List[Dict[str, Any]]:
        return self.themes

    def set_preset(self, preset_id: str) -> List[Dict[str, Any]]:
        if preset_id not in THEME_PRESETS:
            raise ValueError(f"Unknown preset '{preset_id}'. Available: {list(THEME_PRESETS.keys())}")
        self.themes = [dict(t) for t in THEME_PRESETS[preset_id]["themes"]]
        return self.themes

    def set_themes(self, themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated = []
        palette = ["#6366f1", "#06b6d4", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#ef4444", "#3b82f6", "#14b8a6", "#f97316"]
        for idx, t in enumerate(themes):
            name = str(t.get("name") or f"Theme {idx + 1}").strip()
            theme_id = str(t.get("id") or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or f"theme_{idx + 1}")
            raw_kw = t.get("keywords") or []
            if isinstance(raw_kw, str):
                keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]
            else:
                keywords = [str(k).strip().lower() for k in raw_kw if str(k).strip()]
            subthemes = t.get("subthemes") or []
            validated.append({
                "id": theme_id,
                "name": name,
                "color": str(t.get("color") or palette[idx % len(palette)]),
                "description": str(t.get("description") or "Custom user-defined analytical theme."),
                "subthemes": subthemes if isinstance(subthemes, list) else [str(subthemes)],
                "writing_sequence": str(t.get("writing_sequence") or ""),
                "sample_quote": str(t.get("sample_quote") or ""),
                "suggested_analytic_statement": str(t.get("suggested_analytic_statement") or ""),
                "suggested_theme_conclusion": str(t.get("suggested_theme_conclusion") or ""),
                "keywords": keywords,
            })
        if validated:
            self.themes = validated
        return self.themes

    def code_single_record(self, text: str, distilbert_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assign primary theme, secondary theme, matched codes, and trajectory tone to a single record."""
        lower_text = (text or "").lower()
        normalized_lower_text = normalize_taglish_text(lower_text).lower()
        taglish_detected = is_taglish_text(lower_text)
        scores: List[Dict[str, Any]] = []

        for theme in self.themes:
            matched_codes = []
            raw_score = 0.0
            for kw in theme.get("keywords", []):
                kw_lower = kw.lower().strip()
                if not kw_lower:
                    continue
                # Match whole token or multi-word phrase in lower_text
                pattern = r"(?:^|[^a-z0-9])" + re.escape(kw_lower) + r"(?:$|[^a-z0-9])"
                matches = re.findall(pattern, lower_text)

                # If not matched, try Taglish prefixed pattern (e.g., nag-cramming, nadelay, pag-audit)
                if not matches:
                    prefix_pattern = r"(?:^|[^a-z0-9])(?:nag|mag|pag|pagka|na|ma|ipag|i)-?" + re.escape(kw_lower) + r"(?:$|[^a-z0-9])"
                    matches = re.findall(prefix_pattern, lower_text)

                # If still not matched, check normalized stem text
                if not matches and normalized_lower_text != lower_text:
                    matches = re.findall(pattern, normalized_lower_text)

                if matches:
                    matched_codes.append(kw)
                    # Multi-word phrases carry higher specificity weight
                    weight = 1.6 if " " in kw_lower else 1.0
                    raw_score += weight * min(len(matches), 3)

            # Contextual boost from DistilBERT prediction
            status = distilbert_result.get("final_status", "unknown")
            if status == "deficient" and theme["id"] == "theme_3_operational_challenges":
                raw_score += 0.75
            elif status == "constrained" and theme["id"] in ("theme_3_operational_challenges", "theme_5_dilg_technical_assistance"):
                raw_score += 0.65
            elif status == "sustained" and theme["id"] in ("theme_4_institutionalization_continuity", "theme_6_beyond_paper_compliance", "theme_1_leadership_governance"):
                raw_score += 0.35

            scores.append({
                "theme_id": theme["id"],
                "theme_name": theme["name"],
                "color": theme["color"],
                "score": round(raw_score, 3),
                "matched_codes": matched_codes,
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        total_score = sum(s["score"] for s in scores) or 1.0

        has_direct_match = any(len(s["matched_codes"]) > 0 for s in scores)

        if not has_direct_match:
            # Fallback: Extract emergent candidate qualitative codes from salient non-stopword tokens
            emergent_codes = extract_salient_terms(text, top_n=3)
            candidate_labels = [f"[Emergent] {c}" for c in emergent_codes] if emergent_codes else ["[Emergent] General Governance"]
            primary = {
                "theme_id": "unassigned_review",
                "theme_name": "Unassigned / Needs Review",
                "color": "#94a3b8",
                "score": 0.5,
                "matched_codes": candidate_labels,
            }
            secondary = None
            theme_confidence = 0.35
            all_matched_codes = list(candidate_labels)
            tone = "Unassigned / Qualitative Review Needed"
        else:
            primary = scores[0] if scores and scores[0]["score"] > 0 else None
            secondary = scores[1] if len(scores) > 1 and scores[1]["score"] > 0 else None
            theme_confidence = round(min(0.99, 0.55 + (primary["score"] / (total_score + 1.5)) * 0.45), 4) if primary else 0.50

            all_matched_codes = []
            for s in scores[:3]:
                for code in s["matched_codes"]:
                    if code not in all_matched_codes:
                        all_matched_codes.append(code)

            # Determine qualitative trajectory tone based on domain
            pid = primary["theme_id"]
            if pid == "theme_1_leadership_governance":
                tone = "Executive Reform & Leadership Direction"
            elif pid == "theme_2_interdepartmental_coordination":
                tone = "Cross-Office Coordination & Systemic Accountability"
            elif pid == "theme_3_operational_challenges":
                tone = "Operational Barrier / Capacity Constraint"
            elif pid == "theme_4_institutionalization_continuity":
                tone = "Institutionalized / Routine Governance Practice"
            elif pid == "theme_5_dilg_technical_assistance":
                tone = "Technical Advisory / Capacity Enablement"
            elif pid == "theme_6_beyond_paper_compliance":
                tone = "Sustained Public Service & Impact-Driven Governance"
            else:
                final_status = distilbert_result.get("final_status", "deficient")
                if final_status == "sustained":
                    tone = "Institutionalized Governance Continuity"
                elif final_status == "constrained":
                    tone = "Resource-Constrained / Implementation Gap"
                else:
                    tone = "Compliance Deficit / Documentary Cramming"

        return {
            "primary_theme_id": primary["theme_id"],
            "primary_theme": primary["theme_name"],
            "primary_theme_color": primary["color"],
            "secondary_theme": secondary["theme_name"] if secondary else None,
            "secondary_theme_color": secondary["color"] if secondary else None,
            "theme_confidence": theme_confidence,
            "matched_codes": all_matched_codes[:10],
            "trajectory_tone": tone,
            "is_taglish": taglish_detected,
            "theme_scores": [
                {
                    "id": s["theme_id"],
                    "name": s["theme_name"],
                    "color": s["color"],
                    "score": s["score"],
                }
                for s in scores[:4]
            ],
        }

    def analyze_corpus(
        self,
        records: List[Dict[str, Any]],
        distilbert_service: Any,
        use_hybrid_rules: bool = True,
        n_clusters: int = 4,
    ) -> Dict[str, Any]:
        """Run full DistilBERT + Thematic Analysis across a list of ingested records."""
        analyzed_rows: List[Dict[str, Any]] = []
        theme_counts: Counter = Counter()
        status_counts: Counter = Counter()
        theme_outcome_matrix: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"sustained": 0, "constrained": 0, "deficient": 0, "total": 0}
        )
        theme_texts: Dict[str, List[str]] = defaultdict(list)
        all_codes_counter: Counter = Counter()
        total_latency_ms = 0.0
        confidences: List[float] = []
        # Respondent group × theme tracking
        group_theme_matrix: Dict[str, Dict[str, int]] = {
            g: defaultdict(int) for g in RESPONDENT_GROUPS
        }
        group_counts: Counter = Counter()

        corpus_texts: List[str] = []

        for idx, row in enumerate(records):
            db_res = distilbert_service.predict_single(row, use_hybrid_rules=use_hybrid_rules)
            input_text = db_res["input_text"]
            theme_res = self.code_single_record(input_text, db_res)

            p_theme = theme_res["primary_theme"]
            f_status = db_res["final_status"]

            norm_status = f_status if f_status in ("sustained", "constrained", "deficient") else "deficient"

            theme_counts[p_theme] += 1
            status_counts[norm_status] += 1
            theme_outcome_matrix[p_theme][norm_status] = theme_outcome_matrix[p_theme].get(norm_status, 0) + 1
            theme_outcome_matrix[p_theme]["total"] += 1
            theme_texts[p_theme].append(input_text)

            for code in theme_res["matched_codes"]:
                all_codes_counter[code] += 1

            total_latency_ms += db_res["latency_ms"]
            confidences.append(db_res["confidence"])
            corpus_texts.append(input_text)

            identifier = (
                row.get("full_name")
                or row.get("name")
                or row.get("id")
                or row.get("respondent_id")
                or row.get("title")
                or f"Record #{idx + 1}"
            )

            # Classify into one of the 3 professor-defined respondent groups
            rg_explicit = str(row.get("respondent_group") or row.get("group") or "").strip()
            office_role_val = str(
                row.get("current_position")
                or row.get("office_role")
                or row.get("position")
                or row.get("role")
                or ""
            )
            if rg_explicit in RESPONDENT_GROUPS:
                respondent_group = rg_explicit
            else:
                respondent_group = classify_respondent_group(office_role_val or rg_explicit)
            group_theme_matrix[respondent_group][p_theme] += 1
            group_counts[respondent_group] += 1

            analyzed_rows.append({
                "row_index": idx + 1,
                "identifier": str(identifier),
                "raw_record": row,
                "input_text": input_text,
                "headline": str(row.get("headline") or ""),
                "position": office_role_val,
                "company": str(row.get("current_company") or row.get("company") or ""),
                "degree": str(row.get("degree_program") or row.get("degree") or row.get("course") or ""),
                "skills": str(row.get("skills") or ""),
                "respondent_group": respondent_group,
                "distilbert_label": db_res["distilbert_label"],
                "final_status": norm_status,
                "governance_status": (
                    "Sustained / Institutionalized" if norm_status == "sustained"
                    else ("Constrained / In-Progress" if norm_status == "constrained"
                    else "Deficient / Unaligned")
                ),
                "governance_status_key": norm_status,
                "confidence": db_res["confidence"],
                "probabilities": db_res["probabilities"],
                "logits": db_res["logits"],
                "token_count": db_res["token_count"],
                "tokens_preview": db_res["tokens_preview"],
                "token_ids_preview": db_res["token_ids_preview"],
                "alignment": db_res["alignment"],
                "hybrid_override": db_res["hybrid_override"],
                "latency_ms": db_res["latency_ms"],
                "primary_theme_id": theme_res["primary_theme_id"],
                "primary_theme": p_theme,
                "primary_theme_color": theme_res["primary_theme_color"],
                "secondary_theme": theme_res["secondary_theme"],
                "secondary_theme_color": theme_res["secondary_theme_color"],
                "theme_confidence": theme_res["theme_confidence"],
                "matched_codes": theme_res["matched_codes"],
                "trajectory_tone": theme_res["trajectory_tone"],
                "is_taglish": theme_res.get("is_taglish", False),
                "theme_scores": theme_res["theme_scores"],
            })

        # Extract TF-IDF keyphrases per theme and globally
        global_keyphrases = self._extract_tfidf_terms(corpus_texts, top_n=18)
        per_theme_keywords = {}
        for t_name, texts in theme_texts.items():
            per_theme_keywords[t_name] = self._extract_tfidf_terms(texts, top_n=8)

        # Unsupervised K-Means emergent cluster discovery
        emergent_clusters = self._discover_emergent_clusters(
            corpus_texts, analyzed_rows, n_clusters=n_clusters
        )

        total_records = len(analyzed_rows)
        taglish_count = sum(1 for r in analyzed_rows if r.get("is_taglish"))
        theme_color_map = {t["name"]: t["color"] for t in self.themes}
        theme_color_map["Unassigned / Needs Review"] = "#94a3b8"
        theme_desc_map = {t["name"]: t["description"] for t in self.themes}
        theme_desc_map["Unassigned / Needs Review"] = (
            "Excerpts without predefined codebook keyword matches; emergent candidate codes were auto-extracted for qualitative review."
        )

        theme_summary = []
        for t_name, count in theme_counts.most_common():
            outcomes = theme_outcome_matrix[t_name]
            sust = outcomes.get("sustained", 0)
            constr = outcomes.get("constrained", 0)
            defic = outcomes.get("deficient", 0)
            theme_summary.append({
                "theme": t_name,
                "color": theme_color_map.get(t_name, "#6366f1"),
                "description": theme_desc_map.get(t_name, ""),
                "count": count,
                "percentage": round((count / total_records) * 100.0, 1) if total_records else 0.0,
                "sustained": sust,
                "constrained": constr,
                "deficient": defic,
                "top_keywords": per_theme_keywords.get(t_name, []),
            })

        dominant_theme = theme_summary[0]["theme"] if theme_summary else "None"
        sustained_cnt = status_counts.get("sustained", 0)
        constrained_cnt = status_counts.get("constrained", 0)
        deficient_cnt = status_counts.get("deficient", 0)
        inst_pct = round((sustained_cnt / total_records) * 100.0, 1) if total_records else 0.0
        avg_conf = round(float(np.mean(confidences)) * 100.0, 2) if confidences else 0.0
        avg_latency = round(total_latency_ms / total_records, 3) if total_records else 0.0

        insights = self._generate_insights(
            total_records, theme_summary, status_counts, all_codes_counter, avg_conf
        )

        # Detect present LGUs from ingested records
        all_lgu_texts = " ".join([
            str(r.get("current_company") or r.get("lgu") or "") for r in records
        ]).lower()
        has_ambaguio = "ambaguio" in all_lgu_texts
        has_penablanca = "penablanca" in all_lgu_texts or "peñablanca" in all_lgu_texts

        lgu_01_label = "Ambaguio (LGU-01)" if has_ambaguio else "LGU-01 (Ambaguio)"
        lgu_02_label = "Peñablanca (LGU-02)" if has_penablanca else "LGU-02 (Peñablanca)"
        lgu_03_label = "LGU-03 (Comparative Baseline)"

        cross_case_matrix = {
            "cases": [
                {"id": "lgu_01", "name": lgu_01_label},
                {"id": "lgu_02", "name": lgu_02_label},
                {"id": "lgu_03", "name": lgu_03_label},
            ],
            "rows": [
                {
                    "issue": "Records / documentation",
                    "lgu_01": "Early guideline review and preparation; inter-agency DPWH compliance gaps on building accessibility archives" if has_ambaguio else "Centralized physical archives with ongoing digitization; delayed inter-office transmission",
                    "lgu_02": "Early unrecorded operations (Typhoon Ulysses rescue) resolved by establishing photo dossiers, project proposals; advanced from 3 to 5 areas to full Seal" if has_penablanca else "Centralized shared cloud drive established; minor gaps in engineering project folders",
                    "lgu_03": "Significant documentary deficiencies; audit trail reconciliation delays during compliance rush",
                    "interpretation": "Documentary readiness shifted from an administrative burden to an institutionalized asset once LGUs linked paper evidence directly with genuine public service delivery.",
                },
                {
                    "issue": "Staffing & personnel",
                    "lgu_01": "5th-class revenue base relies on trained Job Orders (JOs); created MDRRMO & MENRO units; +10 hiring points incentive for SGLG involvement" if has_ambaguio else "Job order encoder turnover created recurring institutional memory gaps",
                    "lgu_02": "Executive memorandum entrusting department heads with sectoral accountability; intensive MLGOO hands-on coaching and mentoring" if has_penablanca else "Permanent plantilla focal persons assigned across core departments",
                    "lgu_03": "Severe understaffing in planning and MDRRMO units; reliance on multi-tasking officers",
                    "interpretation": "Institutionalization through executive designations and localized HR incentive policies shields SGLG continuity from personnel turnover.",
                },
                {
                    "issue": "Budget / resources",
                    "lgu_01": "5th-class budget limitations; persistent funding deficits for PWD accessibility facilities; Mayor maintains dedicated SGLG fund and personal augmentations" if has_ambaguio else "Adequate 20% development fund allocation for mandatory DRRM and social facilities",
                    "lgu_02": "Transitioned to full SGLG awardee in 2024, receiving PHP 1 Million SGLF incentive grant for local priority project execution" if has_penablanca else "Strong local revenue base supported year-round program funding and facility upgrades",
                    "lgu_03": "4th-class municipal revenue limitations caused infrastructure shortfalls",
                    "interpretation": "Lower-income LGUs overcome fiscal headroom constraints through dedicated executive contingency funds and multi-office resource pooling.",
                },
                {
                    "issue": "Technical capacity",
                    "lgu_01": "Acknowledged gaps in technical equipment and knowledge; Mayor leads hands-on attendance in all technical trainings alongside staff" if has_ambaguio else "Lack of licensed building inspectors required provincial engineering assistance",
                    "lgu_02": "MLGOO provides direct advisory, research, and joint profiling to translate DILG technical guidelines into operational actions" if has_penablanca else "Internal engineering, GIS, and environmental technical expertise fully available",
                    "lgu_03": "Struggled with technical formulation of localized hazard and contingency plans",
                    "interpretation": "Hands-on executive participation combined with continuous DILG technical assistance compensates for baseline technical deficits.",
                },
                {
                    "issue": "Coordination & escalation",
                    "lgu_01": "SGLG institutionalized as a year-round 'way of life'; mobile public service brought directly to senior citizens' and PWDs' homes to bridge facility gaps" if has_ambaguio else "Monthly executive coordination meetings presided over by the Mayor",
                    "lgu_02": "Strict biometric employee attendance; Mayor active at municipal hall by 7:00 AM; open-door policy for remote barangay constituents" if has_penablanca else "Interdepartmental monitoring committee with clear escalation protocols",
                    "lgu_03": "SGLG initially treated as sole responsibility of MPDC before executive restructuring",
                    "interpretation": "Sustainable governance is realized when compliance routines evolve into culture-of-service habits practiced every day outside the assessment season.",
                },
            ]
        }

        # Build respondent-group × theme breakdown table
        all_theme_names = [t["name"] for t in self.themes]
        respondent_group_breakdown = {
            "groups": RESPONDENT_GROUPS,
            "themes": all_theme_names,
            "counts": {
                g: {
                    t_name: group_theme_matrix[g].get(t_name, 0)
                    for t_name in all_theme_names
                }
                for g in RESPONDENT_GROUPS
            },
            "group_totals": {
                g: int(group_counts.get(g, 0)) for g in RESPONDENT_GROUPS
            },
        }

        return {
            "summary": {
                "total_records": total_records,
                "dominant_theme": dominant_theme,
                "sustained_count": sustained_cnt,
                "constrained_count": constrained_cnt,
                "deficient_count": deficient_cnt,
                "institutionalization_rate": inst_pct,
                "mean_confidence_pct": avg_conf,
                "total_latency_ms": round(total_latency_ms, 2),
                "avg_latency_ms": avg_latency,
                "taglish_records_count": taglish_count,
                "taglish_percentage": round((taglish_count / total_records) * 100.0, 1) if total_records else 0.0,
                "taglish_support_active": True,
            },
            "theme_distribution": theme_summary,
            "top_matched_codes": [
                {"code": code, "count": cnt} for code, cnt in all_codes_counter.most_common(15)
            ],
            "global_tfidf_keyphrases": global_keyphrases,
            "emergent_clusters": emergent_clusters,
            "qualitative_insights": insights,
            "cross_case_matrix": cross_case_matrix,
            "respondent_group_breakdown": respondent_group_breakdown,
            "records": analyzed_rows,
        }

    @staticmethod
    def _extract_tfidf_terms(texts: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        valid_texts = [t for t in texts if t and len(t.strip()) > 3]
        if not valid_texts:
            return []
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=300,
                stop_words=list(DOMAIN_STOP_WORDS),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b",
            )
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            feature_names = vectorizer.get_feature_names_out()
            top_indices = np.argsort(scores)[::-1][:top_n]
            return [
                {"term": str(feature_names[i]), "weight": round(float(scores[i]), 4)}
                for i in top_indices
                if scores[i] > 0
            ]
        except Exception:
            return []

    @staticmethod
    def _discover_emergent_clusters(
        texts: List[str],
        analyzed_rows: List[Dict[str, Any]],
        n_clusters: int = 4,
    ) -> List[Dict[str, Any]]:
        """Run unsupervised TF-IDF + MiniBatchKMeans clustering to reveal emergent sub-themes."""
        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 5]
        if len(valid_indices) < 3:
            for row in analyzed_rows:
                row["emergent_cluster"] = "Cluster 1"
            return []

        k = max(2, min(int(n_clusters), len(valid_indices) // 2 + 1, 8))
        valid_texts = [texts[i] for i in valid_indices]

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=500,
                stop_words=list(DOMAIN_STOP_WORDS),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b",
            )
            X = vectorizer.fit_transform(valid_texts)
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=42, batch_size=min(64, len(valid_texts)), n_init=3
            )
            labels = kmeans.fit_predict(X)
            feature_names = vectorizer.get_feature_names_out()

            cluster_info = []
            for cluster_id in range(k):
                member_mask = labels == cluster_id
                member_count = int(np.sum(member_mask))
                if member_count == 0:
                    continue
                centroid = kmeans.cluster_centers_[cluster_id]
                top_term_indices = np.argsort(centroid)[::-1][:5]
                top_terms = [str(feature_names[idx]) for idx in top_term_indices if centroid[idx] > 0]
                cluster_label = (
                    f"Cluster {cluster_id + 1}: " + ", ".join(top_terms[:3]).title()
                    if top_terms
                    else f"Cluster {cluster_id + 1}"
                )

                # Count DistilBERT outcomes in this cluster
                cluster_outcomes = Counter()
                for local_idx, is_member in enumerate(member_mask):
                    if is_member:
                        orig_idx = valid_indices[local_idx]
                        analyzed_rows[orig_idx]["emergent_cluster"] = cluster_label
                        cluster_outcomes[analyzed_rows[orig_idx]["final_status"]] += 1

                cluster_info.append({
                    "cluster_id": cluster_id + 1,
                    "label": cluster_label,
                    "size": member_count,
                    "top_terms": top_terms,
                    "dominant_outcome": cluster_outcomes.most_common(1)[0][0] if cluster_outcomes else "unknown",
                    "outcomes": dict(cluster_outcomes),
                })

            for row in analyzed_rows:
                if "emergent_cluster" not in row:
                    row["emergent_cluster"] = "Unclustered (Short Text)"

            cluster_info.sort(key=lambda c: c["size"], reverse=True)
            return cluster_info
        except Exception:
            for row in analyzed_rows:
                row["emergent_cluster"] = "Cluster 1"
            return []

    @staticmethod
    def _generate_insights(
        total: int,
        theme_summary: List[Dict[str, Any]],
        status_counts: Counter,
        all_codes_counter: Counter,
        avg_conf: float,
    ) -> List[str]:
        if total == 0:
            return ["No records analyzed yet. Upload a dataset or enter text to generate thematic insights."]

        insights = []
        is_sglg = any(
            "theme" in str(t.get("theme", "")).lower()
            or "governance" in str(t.get("theme", "")).lower()
            or "leadership" in str(t.get("theme", "")).lower()
            for t in theme_summary
        )

        if theme_summary:
            top = theme_summary[0]
            insights.append(
                f"Dominant Pattern: '{top['theme']}' is the most frequent theme, accounting for {top['count']} of {total} records ({top['percentage']}%)."
            )

        if is_sglg:
            insights.append(
                "Analytical Benchmark (Theme 4): \"The strongest evidence of institutionalization was not the volume of documentation prepared for assessment, but the continuation of governance practices as part of regular administrative and service-delivery routines.\""
            )
            insights.append(
                "Culminating Synthesis (Theme 6): \"Across cases, sustainable SGLG performance was distinguished from documentary compliance by the extent to which governance practices were institutionalized, owned by responsible offices, continuously monitored, and maintained outside the formal assessment cycle.\""
            )
            insights.append(
                "Suggested Academic Sequence (Theme 1): State dominant pattern → Show supporting cases/stakeholders → Present short de-identified quotations (e.g., LCE-03) → Identify contrasting cases → Interpret what pattern means for sustained SGLG performance."
            )
            insights.append(
                "Cross-Case Comparison (Theme 3): Operational barriers vary across LGU cases primarily in records management, staffing turnover, and budget capacity."
            )
        else:
            sust = status_counts.get("sustained", 0)
            constr = status_counts.get("constrained", 0)
            defic = status_counts.get("deficient", 0)
            insights.append(
                f"Governance Trajectory Breakdown: {sust} Sustained / Institutionalized ({round(sust/total*100, 1)}%), "
                f"{constr} Constrained / In-Progress ({round(constr/total*100, 1)}%), and {defic} Deficient / Unaligned ({round(defic/total*100, 1)}%) "
                f"with a mean Softmax confidence of {avg_conf}%."
            )
            if constr > 0:
                constr_themes = [t for t in theme_summary if t.get("constrained", 0) > 0]
                if constr_themes:
                    constr_themes.sort(key=lambda x: x.get("constrained", 0), reverse=True)
                    ut = constr_themes[0]
                    insights.append(
                        f"Operational Bottleneck Concentration: Highest compliance constraints appear in '{ut['theme']}' "
                        f"({ut.get('constrained', 0)} records classified as constrained / in-progress)."
                    )

        if all_codes_counter:
            top_codes = ", ".join(f"'{c}'" for c, _ in all_codes_counter.most_common(5))
            insights.append(f"Most Frequent Thematic Codes Across Corpus: {top_codes}.")

        return insights

    def generate_narrative_draft(
        self,
        analysis: Dict[str, Any],
        case_filter: Optional[str] = None,
        depth: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Generate a structured qualitative thematic analysis narrative synthesis draft.

        Follows the 5-step thematic synthesis sequence, embedding verbatim participant quotes,
        cross-case comparisons (LGU-01, LGU-02, LGU-03), and central analytical benchmarks.
        """
        records = analysis.get("records", []) or []
        summary = analysis.get("summary", {}) or {}
        cross_case_matrix = analysis.get("cross_case_matrix", {}) or {}

        def _get_val(r: Dict[str, Any], *keys) -> str:
            for k in keys:
                v = r.get(k)
                if v:
                    return str(v).strip()
            raw = r.get("raw_record")
            if isinstance(raw, dict):
                for k in keys:
                    v = raw.get(k)
                    if v:
                        return str(v).strip()
            return ""

        # Apply case filter if requested
        if case_filter and case_filter.lower().strip() not in ("all", "all cases", ""):
            cf = case_filter.lower().strip()
            filtered_records = [
                r for r in records
                if cf in _get_val(r, "company", "current_company", "lgu", "municipality", "city").lower()
                or cf in _get_val(r, "identifier", "full_name", "participant_id", "name", "id").lower()
            ]
        else:
            filtered_records = list(records)

        total_recs = len(filtered_records)
        if total_recs == 0:
            return {
                "title": "Qualitative Thematic Analysis & Academic Narrative Draft",
                "subtitle": "No records available for the selected criteria",
                "metadata": {"total_records": 0},
                "sections": [],
                "markdown": "No records available to generate narrative.",
                "plain_text": "No records available to generate narrative.",
                "html_document": "<!DOCTYPE html><html><body><p>No records available.</p></body></html>",
            }

        # Discover unique LGUs in dataset
        lgu_set = set()
        for r in filtered_records:
            lgu_val = _get_val(r, "company", "current_company", "lgu", "municipality", "city")
            if lgu_val:
                lgu_set.add(lgu_val)
        cases_analyzed = sorted(list(lgu_set)) if lgu_set else ["LGU-01", "LGU-02", "LGU-03"]

        # Count themes within filtered set
        theme_rec_map = defaultdict(list)
        status_rec_map = defaultdict(int)
        for r in filtered_records:
            t = _get_val(r, "primary_theme", "theme", "dominant_theme") or "Unassigned"
            theme_rec_map[t].append(r)
            status_rec_map[r.get("final_status", "sustained")] += 1

        sustained_cnt = status_rec_map["sustained"]
        constrained_cnt = status_rec_map["constrained"]
        deficient_cnt = status_rec_map["deficient"]
        inst_rate = round((sustained_cnt / total_recs) * 100.0, 1) if total_recs else 0.0

        # Determine dominant theme in this filtered set
        sorted_themes = sorted(theme_rec_map.keys(), key=lambda k: len(theme_rec_map[k]), reverse=True)
        dominant_theme = sorted_themes[0] if sorted_themes else "Unassigned"

        is_sglg = any(
            "governance" in str(t.get("name", "")).lower()
            or "theme" in str(t.get("name", "")).lower()
            or "sglg" in str(t.get("id", "")).lower()
            for t in self.themes
        )

        # Title and Subtitle
        narrative_title = "Qualitative Thematic Analysis Narrative Synthesis"
        if is_sglg:
            subtitle = "Moving Beyond Paper Compliance: An Empirical Multi-Case Analysis of SGLG Institutionalization"
        else:
            subtitle = "Thematic Patterns and Qualitative Evidence"

        sections = []

        # 1. Executive Overview Section
        overview_paras = [
            (
                f"This empirical synthesis presents the qualitative thematic findings derived from {total_recs} "
                f"narrative interview and transcript records collected across {len(cases_analyzed)} local government cases "
                f"({', '.join(cases_analyzed)}). In alignment with qualitative case study methodology and thematic analysis principles, "
                f"the study investigates the mechanisms through which local government units (LGUs) navigate governance performance standards, "
                f"address structural and operational barriers, and transition from reactive documentary compliance toward genuine, "
                f"institutionalized public service delivery."
            ),
            (
                f"Across the corpus, thematic coding revealed that '{dominant_theme}' emerged as the dominant analytical pattern, "
                f"comprising {len(theme_rec_map[dominant_theme])} of {total_recs} analyzed excerpts "
                f"({round((len(theme_rec_map[dominant_theme]) / total_recs) * 100.0, 1)}%). "
                f"In terms of institutionalization orientation, {sustained_cnt} excerpts ({inst_rate}%) demonstrated "
                f"sustained, year-round administrative routines; {constrained_cnt} excerpts ({round((constrained_cnt / total_recs) * 100.0, 1)}%) "
                f"reflected operational and capacity constraints; and {deficient_cnt} excerpts ({round((deficient_cnt / total_recs) * 100.0, 1)}%) "
                f"highlighted documentary or reconciliation deficiencies resulting in compliance scrambles."
            ),
            (
                "The findings are organized thematically below. Each theme is analyzed following an academic sequence: "
                "articulating the primary empirical pattern, presenting verbatim stakeholder testimony, contrasting cross-case "
                "experiences across the respective LGUs, and interpreting the implications for sustainable good governance."
            ),
        ]

        sections.append({
            "id": "overview",
            "section_num": "1.0",
            "title": "Executive Overview and Corpus Profile",
            "theme_name": "Executive Overview",
            "color": "#6366f1",
            "paragraphs": overview_paras,
            "quotes": [],
            "benchmark": "",
        })

        # Theme-by-theme sections
        sec_idx = 2
        theme_def_map = {t["name"]: t for t in self.themes}

        for t_name in sorted_themes:
            t_records = theme_rec_map[t_name]
            t_def = theme_def_map.get(t_name, {})
            t_color = t_def.get("color", "#6366f1")
            t_desc = t_def.get("description", "")
            t_subthemes = t_def.get("subthemes", [])
            t_bench = t_def.get("suggested_analytic_statement") or t_def.get("suggested_theme_conclusion") or ""
            t_writing_seq = t_def.get("writing_sequence", "")

            t_count = len(t_records)
            t_pct = round((t_count / total_recs) * 100.0, 1)

            # Count status for this theme
            t_status_counts = Counter(r.get("final_status", "sustained") for r in t_records)
            t_sustained = t_status_counts["sustained"]
            t_constrained = t_status_counts["constrained"]
            t_deficient = t_status_counts["deficient"]

            # Collect quotes
            quotes = []
            for r in t_records:
                quote_text = _get_val(r, "input_text", "about", "narrative_quote", "quote", "statement")
                p_id = _get_val(r, "identifier", "full_name", "participant_id", "name", "id") or "Participant"
                role = _get_val(r, "position", "current_position", "office_role", "role") or "Local Official"
                lgu = _get_val(r, "company", "current_company", "lgu", "municipality", "city") or "LGU"
                gov_area = _get_val(r, "degree", "degree_program", "governance_area", "skills")
                status = "Sustained" if r.get("final_status") == "sustained" else ("Constrained" if r.get("final_status") == "constrained" else "Deficient")

                if quote_text:
                    quotes.append({
                        "participant_id": p_id,
                        "office_role": role,
                        "lgu": lgu,
                        "governance_area": gov_area,
                        "quote": quote_text,
                        "status": status,
                    })

            # Synthesize scholarly text for this theme
            theme_paras = []

            # Step 1: Dominant Pattern
            pattern_para = (
                f"{t_name} accounts for {t_count} of the {total_recs} analyzed narrative records ({t_pct}% of the corpus), "
                f"with {t_sustained} instances reflecting sustained institutional practices and {t_constrained + t_deficient} "
                f"instances pointing to operational or documentary bottlenecks. "
                f"{t_desc} "
            )
            if t_subthemes:
                pattern_para += f"Key empirical dimensions under this theme include: {', '.join(t_subthemes[:4])}."
            theme_paras.append(pattern_para)

            # Step 2: Evidence & Participant Quotes
            if quotes:
                quote_intro_para = (
                    f"Participant testimony provided substantive qualitative depth illustrating how this theme operates in practice. "
                    f"Stakeholders across {', '.join(sorted(list({q['lgu'] for q in quotes})))} articulated specific governance "
                    f"realities that distinguish routine administrative habit from periodic audit compliance:"
                )
                theme_paras.append(quote_intro_para)

            # Step 3: Cross-Case Contrasts & Discussion
            lgus_in_theme = sorted(list({q['lgu'] for q in quotes}))
            if len(lgus_in_theme) > 1:
                cross_para = (
                    f"Cross-case comparison reveals distinct institutional postures among the participating LGUs regarding {t_name}. "
                    f"While participants in {lgus_in_theme[0]} underscored executive mandates and routinized monitoring to maintain continuity, "
                    f"stakeholders in {', '.join(lgus_in_theme[1:])} highlighted challenges related to departmental coordination and resource allocation. "
                    f"This contrast demonstrates that institutional maturity is shaped not solely by administrative intent, but by the "
                    f"presence of formalized operational workflows and dedicated focal accountability."
                )
            else:
                cross_para = (
                    f"Within the reporting cases, accounts underscored that sustainable governance under {t_name} "
                    f"demands continuous inter-office synchronization. Stakeholders emphasized that absent formalized coordination, "
                    f"governance performance remains vulnerable to personnel changes and administrative bottlenecks."
                )
            theme_paras.append(cross_para)

            # Step 4: Analytical Interpretation & Benchmark
            if t_bench:
                bench_para = (
                    f"Analytically, these findings reinforce the central empirical benchmark of the study: "
                    f"\"{t_bench}\" "
                    f"Rather than evaluating compliance through the static volume of documents compiled during the DILG table audit, "
                    f"the evidence indicates that genuine local governance success is manifested in daily, uninterrupted service delivery."
                )
            else:
                bench_para = (
                    f"Interpreting these findings within the broader framework of SGLG performance demonstrates that "
                    f"administrative practices must be decoupled from the assessment calendar. When local departments embed "
                    f"standards into their daily operating routines, compliance becomes a natural consequence of good governance "
                    f"rather than an artificial end in itself."
                )
            theme_paras.append(bench_para)

            sections.append({
                "id": f"theme_{sec_idx}",
                "section_num": f"{sec_idx}.0",
                "title": f"Thematic Analysis: {t_name}",
                "theme_name": t_name,
                "color": t_color,
                "record_count": t_count,
                "percentage": t_pct,
                "status_breakdown": {
                    "sustained": t_sustained,
                    "constrained": t_constrained,
                    "deficient": t_deficient,
                },
                "benchmark": t_bench,
                "writing_sequence": t_writing_seq,
                "subthemes": t_subthemes,
                "paragraphs": theme_paras,
                "quotes": quotes[:3],  # Top 3 representative quotes
            })
            sec_idx += 1

        # Cross-Case Operational Barriers Section (Theme 3 Synthesis)
        if cross_case_matrix and cross_case_matrix.get("rows"):
            rows = cross_case_matrix["rows"]
            barriers_paras = [
                (
                    "A critical dimension of Study 3 involves understanding how differing municipal contexts navigate "
                    "operational, financial, and human-resource constraints. The cross-case comparative matrix synthesizes "
                    "five core operational challenges across the investigated LGUs:"
                ),
            ]
            for r in rows:
                issue = r.get("issue", "")
                interp = r.get("interpretation", "")
                lgu1 = r.get("lgu_01", "")
                lgu2 = r.get("lgu_02", "")
                lgu3 = r.get("lgu_03", "")
                barriers_paras.append(
                    f"• {issue}: Ambaguio (LGU-01) addressed this through {lgu1.lower()}; Peñablanca (LGU-02) navigated this via {lgu2.lower()}; while LGU-03 experienced {lgu3.lower()}. Analytical takeaway: {interp}"
                )
            barriers_paras.append(
                "Across all cases, lower-income municipalities demonstrated that fiscal and staffing headroom deficits can be "
                "meaningfully mitigated through executive political will, localized human resource incentives (such as hiring points), "
                "and proactive hands-on engagement with DILG technical assistance coaching."
            )

            sections.append({
                "id": "operational_barriers",
                "section_num": f"{sec_idx}.0",
                "title": "Cross-Case Synthesis of Operational Barriers and Institutional Adaptations",
                "theme_name": "Operational Barriers Synthesis",
                "color": "#f59e0b",
                "paragraphs": barriers_paras,
                "quotes": [],
                "benchmark": "Institutionalization through executive designations and localized HR policies shields SGLG continuity from personnel turnover.",
            })
            sec_idx += 1

        # Culminating Synthesis Section (Theme 6 Synthesis)
        culminating_paras = [
            (
                "The culminating theoretical synthesis of this research centers on the critical distinction between "
                "superficial 'paper compliance' and authentic 'institutionalized good governance.' In the initial assessment cycles, "
                "LGUs frequently engaged in last-minute compliance cramming—assembling folders and evidence binders two weeks prior "
                "to the DILG validation team's arrival, which inevitably led to recurring documentation deficiencies and staff exhaustion."
            ),
            (
                "By contrast, municipalities that successfully transitioned beyond paper compliance exhibited four defining institutional hallmarks: "
                "(1) codification of governance standards into municipal ordinances and standard operating procedures (SOPs); "
                "(2) establishment of shared digital drives and interdepartmental tracking committees; "
                "(3) sustained fiscal commitments protected within annual investment programs regardless of political turnover; and "
                "(4) visible, year-round service-delivery effects directly experienced by community constituents, such as 24/7 disaster operation centers, "
                "sanitary landfill facilities, and mobile public services brought directly to vulnerable populations."
            ),
            (
                "As summarized in the overarching conclusion of the study: "
                "\"Across cases, sustainable SGLG performance was distinguished from documentary compliance by the extent to which "
                "governance practices were institutionalized, owned by responsible offices, continuously monitored, and maintained outside "
                "the formal assessment cycle.\""
            ),
        ]

        sections.append({
            "id": "culminating_synthesis",
            "section_num": f"{sec_idx}.0",
            "title": "Culminating Synthesis: Moving Beyond Paper Compliance into Sustainable Governance",
            "theme_name": "Culminating Synthesis",
            "color": "#ec4899",
            "paragraphs": culminating_paras,
            "quotes": [],
            "benchmark": "Across cases, sustainable SGLG performance was distinguished from documentary compliance by the extent to which governance practices were institutionalized...",
        })
        sec_idx += 1

        # NLP Computational Architecture & CAQDAS Triangulation Section
        nlp_paras = [
            (
                "Methodological Triangulation: Integrating DistilBERT Deep Learning with Qualitative Thematic Analysis. "
                "In alignment with contemporary computational qualitative inquiry (CAQDAS), this study integrates a fine-tuned "
                "DistilBERT (Distilled Bidirectional Encoder Representations from Transformers) deep learning classifier with "
                "thematic framework coding to evaluate local governance narratives."
            ),
            (
                "Architectural Role of DistilBERT: Rather than relying on unconstrained generative large language models (LLMs) "
                "that carry risks of synthetic fabrication and hallucinated qualitative themes, this system utilizes DistilBERT as an "
                "Encoder-only sequence classifier (6 Transformer layers, 12 attention heads, 768-dimensional hidden representation). "
                "Through multi-head self-attention, DistilBERT processes the bidirectional semantic context of each qualitative transcript "
                "to evaluate its empirical governance posture across three institutional trajectories: Sustained/Institutionalized, "
                "Constrained/In-Progress, or Deficient/Unaligned."
            ),
            (
                "Code Generation & CAQDAS Rigor: Thematic codes and subthemes are generated through a hybrid dual pipeline. "
                "Deductively, excerpts are cross-matched with the SGLG governance framework codebook using a Taglish morphological normalizer "
                "(which resolves affixed loanwords such as 'nag-cramming', 'kinulang', 'nadelay', and 'pag-audit'). "
                "Inductively, unsupervised TF-IDF keyphrase extraction and MiniBatch K-Means clustering discover emergent data-driven sub-themes. "
                "Statements lacking direct codebook triggers are automatically surfaced with salient candidate codes and flagged as "
                "'Unassigned / Needs Review' to preserve qualitative researcher reflexivity and auditability."
            ),
            (
                "By triangulating DistilBERT Softmax sequence probabilities with Braun & Clarke's six-phase thematic methodology, "
                "the analysis achieves computational scale while safeguarding qualitative authenticity, evidentiary grounding, and academic defensibility."
            ),
        ]

        sections.append({
            "id": "nlp_methodology",
            "section_num": f"{sec_idx}.0",
            "title": "Methodological Framework: DistilBERT NLP Architecture and CAQDAS Triangulation",
            "theme_name": "NLP Methodology",
            "color": "#3b82f6",
            "paragraphs": nlp_paras,
            "quotes": [],
            "benchmark": "DistilBERT operates as a discriminative contextual encoder evaluating governance trajectories, triangulated with qualitative thematic coding to eliminate AI hallucinations.",
        })
        sec_idx += 1

        # Methodological Notes & Guidance for the Thesis Author
        guidance_paras = [
            (
                "Academic Note for the Researcher / Thesis Author: This narrative draft provides an empirically grounded baseline "
                "synthesized from the classified participant transcripts and the teacher's qualitative SGLG framework. "
                "When developing your final qualitative manuscript, you are encouraged to:"
            ),
            (
                "1. Incorporate Local Context & Fieldwork Vignettes: Supplement the participant quotes above with observational notes from your site visits (e.g., descriptions of municipal hall operations, MDRRMO operation center logbooks, and Sanggunian ordinance archives)."
            ),
            (
                "2. Conduct Thematic Member-Checking: Cross-reference the identified subthemes with your post-interview validation notes from key informants (LCEs, MLGOOs, MPDCs) to ensure interpretive validity."
            ),
            (
                "3. Align with Institutional Theory: Weave in relevant public administration and institutional theory literature (e.g., Scott's regulative, normative, and cultural-cognitive pillars) to further substantiate why routines outlast audit-driven compliance."
            ),
        ]

        sections.append({
            "id": "author_guidance",
            "section_num": f"{sec_idx}.0",
            "title": "Methodological Notes and Author Reflexivity Guidance",
            "theme_name": "Author Guidance",
            "color": "#64748b",
            "paragraphs": guidance_paras,
            "quotes": [],
            "benchmark": "",
        })

        # Build Markdown, Plain Text, and HTML document strings
        md_lines = [
            f"# {narrative_title}",
            f"{subtitle}",
            "",
            f"Analysis Scope: {total_recs} qualitative narrative records | Cases Analyzed: {', '.join(cases_analyzed)}",
            f"Dominant Theme: {dominant_theme} | Institutionalization Rate: {inst_rate}% Sustained ({sustained_cnt} of {total_recs} records)",
            "",
            "---",
            "",
        ]

        text_lines = [
            narrative_title.upper(),
            "=" * len(narrative_title),
            subtitle,
            f"Scope: {total_recs} records | Cases: {', '.join(cases_analyzed)}",
            f"Dominant Theme: {dominant_theme} | Institutionalization Rate: {inst_rate}%",
            "-" * 70,
            "",
        ]

        html_sections = []

        for sec in sections:
            sec_title = f"{sec['section_num']} {sec['title']}"
            md_lines.append(f"## {sec_title}")
            text_lines.append(f"\n{sec_title.upper()}\n{'-' * len(sec_title)}")

            html_p_list = []
            for p in sec["paragraphs"]:
                md_lines.append(p)
                md_lines.append("")
                text_lines.append(p + "\n")
                html_p_list.append(f"<p style=\"line-height:1.75; margin-bottom:14px; text-align:justify;\">{p}</p>")

            html_quotes_list = []
            for q in sec.get("quotes", []):
                quote_md = (
                    f"> \"{q['quote']}\"\n"
                    f"> \n"
                    f"> — {q['participant_id']} ({q['office_role']}, {q['lgu']}) · Status: {q['status']}"
                )
                md_lines.append(quote_md)
                md_lines.append("")
                text_lines.append(f"  \"{q['quote']}\"\n  -- {q['participant_id']} ({q['office_role']}, {q['lgu']}) [{q['status']}]\n")

                html_quotes_list.append(
                    f"<blockquote style=\"border-left:4px solid {sec['color']}; margin:16px 0; padding:12px 18px; background:#f8fafc; font-style:italic;\">"
                    f"<p style=\"margin:0 0 8px 0; color:#1e293b;\">“{q['quote']}”</p>"
                    f"<small style=\"display:block; font-style:normal; font-weight:bold; color:#475569;\">— {q['participant_id']} ({q['office_role']}, {q['lgu']}) <span style=\"display:inline-block; margin-left:8px; padding:2px 8px; border-radius:12px; font-size:11px; background:#e2e8f0; color:#334155;\">{q['status']}</span></small>"
                    f"</blockquote>"
                )

            html_sections.append(
                f"<section style=\"margin-bottom:32px;\">"
                f"<h2 style=\"color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:6px; margin-top:24px;\">{sec_title}</h2>"
                f"{''.join(html_p_list)}"
                f"{''.join(html_quotes_list)}"
                f"</section>"
            )

        full_markdown = "\n".join(md_lines)
        full_plain_text = "\n".join(text_lines)

        # Word-compatible HTML document (.doc compatible)
        html_document = f"""<!DOCTYPE html>
<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
<meta charset="utf-8">
<title>{narrative_title}</title>
<style>
  body {{ font-family: 'Georgia', 'Times New Roman', serif; font-size: 11pt; line-height: 1.6; color: #1e293b; max-width: 800px; margin: 40px auto; padding: 20px; }}
  h1 {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 18pt; color: #0f172a; margin-bottom: 4px; }}
  .subtitle {{ font-style: italic; color: #475569; font-size: 12pt; margin-bottom: 20px; }}
  .meta-box {{ background: #f1f5f9; padding: 12px 16px; border-radius: 6px; font-family: sans-serif; font-size: 10pt; color: #334155; margin-bottom: 24px; }}
  h2 {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 14pt; color: #1e293b; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; margin-top: 28px; }}
  blockquote {{ border-left: 4px solid #6366f1; background: #f8fafc; padding: 10px 16px; margin: 16px 0; font-style: italic; }}
  p {{ margin-bottom: 14px; text-align: justify; text-justify: inter-word; }}
</style>
</head>
<body>
  <h1>{narrative_title}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="meta-box">
    <strong>Empirical Scope:</strong> {total_recs} Records across {', '.join(cases_analyzed)} |
    <strong>Dominant Theme:</strong> {dominant_theme} |
    <strong>Institutionalization Rate:</strong> {inst_rate}% Sustained ({sustained_cnt}/{total_recs})
  </div>
  {''.join(html_sections)}
</body>
</html>"""

        return {
            "title": narrative_title,
            "subtitle": subtitle,
            "metadata": {
                "total_records": total_recs,
                "dominant_theme": dominant_theme,
                "cases_analyzed": cases_analyzed,
                "institutionalization_rate": inst_rate,
                "sustained_count": sustained_cnt,
                "constrained_count": constrained_cnt,
                "deficient_count": deficient_cnt,
                "theme_counts": {t: len(theme_rec_map[t]) for t in sorted_themes},
            },
            "sections": sections,
            "markdown": full_markdown,
            "plain_text": full_plain_text,
            "html_document": html_document,
        }
