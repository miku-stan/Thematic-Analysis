"""Thematic Coding, Custom Codebook & Unsupervised TF-IDF/K-Means Theme Discovery Engine."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


DEFAULT_THEMES: List[Dict[str, Any]] = [
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

DOMAIN_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "headline", "position", "company", "about", "skills", "past", "roles", "degree",
    "notes", "bs", "bachelor", "science", "isabela", "state", "university", "isu",
    "philippines", "present", "yr", "yrs", "mo", "mos", "na", "none", "work", "using",
})


class ThematicCoder:
    """Performs codebook-based thematic coding and unsupervised TF-IDF + K-Means theme discovery."""

    def __init__(self, custom_themes: Optional[List[Dict[str, Any]]] = None):
        self.themes = custom_themes if custom_themes else [dict(t) for t in DEFAULT_THEMES]

    def get_themes(self) -> List[Dict[str, Any]]:
        return self.themes

    def set_themes(self, themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated = []
        palette = ["#6366f1", "#06b6d4", "#10b981", "#ec4899", "#f59e0b", "#8b5cf6", "#ef4444", "#3b82f6", "#14b8a6", "#f97316"]
        for idx, t in enumerate(themes):
            name = str(t.get("name") or f"Theme {idx + 1}").strip()
            theme_id = str(t.get("id") or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or f"theme_{idx + 1}")
            raw_kw = t.get("keywords") or []
            if isinstance(raw_kw, str):
                keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]
            else:
                keywords = [str(k).strip().lower() for k in raw_kw if str(k).strip()]
            validated.append({
                "id": theme_id,
                "name": name,
                "color": str(t.get("color") or palette[idx % len(palette)]),
                "description": str(t.get("description") or "Custom user-defined analytical theme."),
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
            # Fallback based on DistilBERT status
            status = distilbert_result.get("final_status", "unemployed")
            fallback_id = (
                "career_transition_seeking"
                if status == "unemployed"
                else "bpo_support_admin"
                if status == "underemployed"
                else "software_engineering"
            )
            fb_theme = next((t for t in self.themes if t["id"] == fallback_id), self.themes[0])
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

        # Determine qualitative trajectory tone
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
        if theme_summary:
            top = theme_summary[0]
            insights.append(
                f"Dominant Theme: '{top['theme']}' accounts for {top['count']} of {total} records ({top['percentage']}%), "
                f"with {top['employed']} employed, {top['underemployed']} underemployed, and {top['unemployed']} unemployed."
            )

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
