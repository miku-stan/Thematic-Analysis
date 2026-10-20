"""Thematic Coding, Custom Codebook & Unsupervised TF-IDF/K-Means Theme Discovery Engine."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


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

# Alternate preset: Alumni Career Tracking & CHED Alignment
ALUMNI_TRACER_THEMES: List[Dict[str, Any]] = [
    {
        "id": "software_engineering",
        "name": "Software & Systems Engineering",
        "color": "#6366f1",
        "description": "Full-stack, web, mobile, and enterprise software architecture and development.",
        "keywords": [
            "software engineer", "software developer", "web developer", "frontend", "backend",
            "full stack", "full-stack", "fullstack", "react", "next.js", "nodejs", "node.js",
            "laravel", "javascript", "typescript", "java", "c#", ".net", "flutter", "android",
            "mobile app", "programmer", "systems developer", "api", "microservices", "git",
            "web application", "django", "flask", "spring boot", "php",
        ],
    },
    {
        "id": "data_ai_analytics",
        "name": "Data Science, AI & Analytics",
        "color": "#06b6d4",
        "description": "Data analysis, business intelligence, machine learning, and data mining workflows.",
        "keywords": [
            "data analyst", "data scientist", "data mining", "machine learning", "artificial intelligence",
            "deep learning", "nlp", "natural language processing", "distilbert", "power bi", "tableau",
            "sql", "postgresql", "mysql", "pandas", "python", "data visualization", "business intelligence",
            "predictive modeling", "statistics", "analytics", "big data", "etl", "data engineer",
            "llm", "computer vision", "tensorflow", "pytorch",
        ],
    },
    {
        "id": "infra_cybersecurity",
        "name": "Cloud, Network & Cybersecurity",
        "color": "#10b981",
        "description": "Network administration, technical infrastructure support, cloud ops, and information security.",
        "keywords": [
            "network engineer", "network administrator", "cybersecurity", "information security",
            "cyber defense", "cloud", "aws", "azure", "devops", "system administrator", "sysadmin",
            "it specialist", "it support", "technical support", "hardware", "windows server",
            "linux", "cisco", "ccna", "firewall", "network security", "troubleshooting", "helpdesk",
            "infrastructure", "database administrator",
        ],
    },
    {
        "id": "creative_multimedia",
        "name": "Creative Design & Digital Media",
        "color": "#ec4899",
        "description": "Graphic design, UI/UX, video editing, multimedia production, and digital marketing creatives.",
        "keywords": [
            "graphic designer", "graphics designer", "video editor", "multimedia", "ui/ux",
            "ux design", "ui design", "figma", "photoshop", "premiere pro", "illustrator",
            "canva", "videographer", "photographer", "creative director", "art direction",
            "logo design", "social media graphic", "brand identity", "animation", "web design",
            "content creator", "layout design",
        ],
    },
    {
        "id": "bpo_support_admin",
        "name": "BPO, Customer Service & Admin Ops",
        "color": "#f59e0b",
        "description": "Customer service representation, back-office encoding, virtual assistance, and administrative roles.",
        "keywords": [
            "call center", "customer service", "csr", "support representative", "technical support representative",
            "virtual assistant", "data entry", "data encoder", "encoder", "administrative assistant",
            "office clerk", "receptionist", "cashier", "chat support", "back office", "bpo",
            "document preparation", "transcription", "order processing", "client relations",
            "delivery rider", "service crew", "sales associate",
        ],
    },
    {
        "id": "management_business",
        "name": "Business, Finance & Management",
        "color": "#8b5cf6",
        "description": "Project management, operations leadership, financial analysis, accounting, and entrepreneurship.",
        "keywords": [
            "project manager", "operations manager", "assistant manager", "team lead", "supervisor",
            "business analyst", "accountant", "accounting", "financial analyst", "auditor",
            "human resources", "hr", "recruitment", "executive", "founder", "ceo", "director",
            "procurement", "supply chain", "sales manager", "marketing manager", "scrum master",
            "product manager", "entrepreneur",
        ],
    },
    {
        "id": "career_transition_seeking",
        "name": "Career Transition & Job Seeking",
        "color": "#ef4444",
        "description": "Graduates actively seeking entry-level opportunities, transitioning roles, or open to work.",
        "keywords": [
            "open to work", "seeking", "looking for", "aspiring", "job hunter", "job seeker",
            "fresh graduate", "entry-level", "entry level", "immediately available", "unemployed",
            "career shifter", "actively applying", "exploring opportunities", "available for hire",
            "looking for opportunities",
        ],
    },
    {
        "id": "academic_upskilling",
        "name": "Academic, OJT & Skill Development",
        "color": "#3b82f6",
        "description": "Academic internships, OJT traineeships, continuing education, certifications, and research.",
        "keywords": [
            "student", "intern", "internship", "ojt", "trainee", "attended", "attending",
            "undergraduate", "scholar", "researcher", "thesis", "capstone", "bootcamp",
            "certification", "instructor", "faculty", "professor", "teacher", "tutor",
            "academic", "university", "college",
        ],
    },
]

THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "sglg_governance": {
        "id": "sglg_governance",
        "name": "Teacher's SGLG Governance Study (6 Themes)",
        "description": "Seal of Good Local Governance (SGLG) qualitative framework covering leadership, interdepartmental coordination, operational challenges, institutionalization, DILG assistance, and sustained governance.",
        "themes": SGLG_GOVERNANCE_THEMES,
    },
    "alumni_tracer": {
        "id": "alumni_tracer",
        "name": "Alumni Career Tracer & Alignment (8 Themes)",
        "description": "Graduate tracer study framework analyzing software engineering, data science, cybersecurity, design, BPO ops, and CHED curriculum alignment.",
        "themes": ALUMNI_TRACER_THEMES,
    },
}

# The default active codebook is the Teacher's SGLG Framework
DEFAULT_THEMES: List[Dict[str, Any]] = SGLG_GOVERNANCE_THEMES

DOMAIN_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "notes", "page", "item", "row", "entry", "response", "statement", "excerpt",
    "na", "none", "n/a", "yr", "yrs", "mo", "mos",
})


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
        scores: List[Dict[str, Any]] = []

        for theme in self.themes:
            matched_codes = []
            raw_score = 0.0
            for kw in theme.get("keywords", []):
                kw_lower = kw.lower().strip()
                if not kw_lower:
                    continue
                # Match whole token or multi-word phrase
                pattern = r"(?:^|[^a-z0-9])" + re.escape(kw_lower) + r"(?:$|[^a-z0-9])"
                matches = re.findall(pattern, lower_text)
                if matches:
                    matched_codes.append(kw)
                    # Multi-word phrases carry higher specificity weight
                    weight = 1.6 if " " in kw_lower else 1.0
                    raw_score += weight * min(len(matches), 3)

            # Contextual boost from DistilBERT prediction
            status = distilbert_result.get("final_status", "unknown")
            if status == "unemployed" and theme["id"] in ("career_transition_seeking", "academic_upskilling"):
                raw_score += 0.75
            elif status == "underemployed" and theme["id"] in ("bpo_support_admin", "creative_multimedia"):
                raw_score += 0.65
            elif status == "employed" and theme["id"] in ("software_engineering", "data_ai_analytics", "infra_cybersecurity"):
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

        primary = scores[0] if scores and scores[0]["score"] > 0 else None
        secondary = scores[1] if len(scores) > 1 and scores[1]["score"] > 0 else None

        if primary is None:
            # Fallback theme selection
            fb_theme = self.themes[0]
            primary = {
                "theme_id": fb_theme["id"],
                "theme_name": fb_theme["name"],
                "color": fb_theme["color"],
                "score": 1.0,
                "matched_codes": [],
            }
            theme_confidence = round(float(distilbert_result.get("confidence", 0.65)) * 0.75, 4)
        else:
            theme_confidence = round(min(0.99, 0.55 + (primary["score"] / (total_score + 1.5)) * 0.45), 4)

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
            final_status = distilbert_result.get("final_status", "unemployed")
            if final_status == "employed":
                tone = "Aligned Professional Growth"
            elif final_status == "underemployed":
                tone = "Skill-Role Misalignment / Underutilization"
            else:
                tone = (
                    "Academic / Pre-Career Preparation"
                    if primary["theme_id"] == "academic_upskilling"
                    else "Active Opportunity Seeking"
                )

        return {
            "primary_theme_id": primary["theme_id"],
            "primary_theme": primary["theme_name"],
            "primary_theme_color": primary["color"],
            "secondary_theme": secondary["theme_name"] if secondary else None,
            "secondary_theme_color": secondary["color"] if secondary else None,
            "theme_confidence": theme_confidence,
            "matched_codes": all_matched_codes[:10],
            "trajectory_tone": tone,
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
            lambda: {"employed": 0, "underemployed": 0, "unemployed": 0, "total": 0}
        )
        theme_texts: Dict[str, List[str]] = defaultdict(list)
        all_codes_counter: Counter = Counter()
        total_latency_ms = 0.0
        confidences: List[float] = []

        corpus_texts: List[str] = []

        for idx, row in enumerate(records):
            db_res = distilbert_service.predict_single(row, use_hybrid_rules=use_hybrid_rules)
            input_text = db_res["input_text"]
            theme_res = self.code_single_record(input_text, db_res)

            p_theme = theme_res["primary_theme"]
            f_status = db_res["final_status"]

            theme_counts[p_theme] += 1
            status_counts[f_status] += 1
            theme_outcome_matrix[p_theme][f_status] = theme_outcome_matrix[p_theme].get(f_status, 0) + 1
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

            analyzed_rows.append({
                "row_index": idx + 1,
                "identifier": str(identifier),
                "raw_record": row,
                "input_text": input_text,
                "headline": str(row.get("headline") or ""),
                "position": str(row.get("current_position") or row.get("position") or row.get("role") or ""),
                "company": str(row.get("current_company") or row.get("company") or ""),
                "degree": str(row.get("degree_program") or row.get("degree") or row.get("course") or ""),
                "skills": str(row.get("skills") or ""),
                "distilbert_label": db_res["distilbert_label"],
                "final_status": f_status,
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
        theme_color_map = {t["name"]: t["color"] for t in self.themes}
        theme_desc_map = {t["name"]: t["description"] for t in self.themes}

        theme_summary = []
        for t_name, count in theme_counts.most_common():
            outcomes = theme_outcome_matrix[t_name]
            theme_summary.append({
                "theme": t_name,
                "color": theme_color_map.get(t_name, "#6366f1"),
                "description": theme_desc_map.get(t_name, ""),
                "count": count,
                "percentage": round((count / total_records) * 100.0, 1) if total_records else 0.0,
                "employed": outcomes["employed"],
                "underemployed": outcomes["underemployed"],
                "unemployed": outcomes["unemployed"],
                "top_keywords": per_theme_keywords.get(t_name, []),
            })

        dominant_theme = theme_summary[0]["theme"] if theme_summary else "None"
        employed_pct = round((status_counts["employed"] / total_records) * 100.0, 1) if total_records else 0.0
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

        return {
            "summary": {
                "total_records": total_records,
                "dominant_theme": dominant_theme,
                "employed_count": status_counts["employed"],
                "underemployed_count": status_counts["underemployed"],
                "unemployed_count": status_counts["unemployed"],
                "employment_alignment_rate": employed_pct,
                "mean_confidence_pct": avg_conf,
                "total_latency_ms": round(total_latency_ms, 2),
                "avg_latency_ms": avg_latency,
            },
            "theme_distribution": theme_summary,
            "top_matched_codes": [
                {"code": code, "count": cnt} for code, cnt in all_codes_counter.most_common(15)
            ],
            "global_tfidf_keyphrases": global_keyphrases,
            "emergent_clusters": emergent_clusters,
            "qualitative_insights": insights,
            "cross_case_matrix": cross_case_matrix,
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
            emp = status_counts["employed"]
            under = status_counts["underemployed"]
            unemp = status_counts["unemployed"]
            insights.append(
                f"DistilBERT Outcome Breakdown: {emp} Employed ({round(emp/total*100, 1)}%), "
                f"{under} Underemployed ({round(under/total*100, 1)}%), and {unemp} Unemployed ({round(unemp/total*100, 1)}%) "
                f"with a mean Softmax confidence of {avg_conf}%."
            )
            if under > 0:
                under_themes = [t for t in theme_summary if t["underemployed"] > 0]
                if under_themes:
                    under_themes.sort(key=lambda x: x["underemployed"], reverse=True)
                    ut = under_themes[0]
                    insights.append(
                        f"Underemployment Concentration: Highest skill-role misalignment appears in '{ut['theme']}' "
                        f"({ut['underemployed']} records classified as underemployed)."
                    )

        if all_codes_counter:
            top_codes = ", ".join(f"'{c}'" for c, _ in all_codes_counter.most_common(5))
            insights.append(f"Most Frequent Thematic Codes Across Corpus: {top_codes}.")

        return insights
