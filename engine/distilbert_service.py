"""DistilBERT ONNX + WordPiece Tokenizer Inference Service for Thematic Analysis."""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from tokenizers import Tokenizer as FastTokenizer
except ImportError:
    FastTokenizer = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from engine.taglish_normalizer import is_taglish_text, normalize_taglish_text


DEFAULT_ID2LABEL = {
    0: "sustained",
    1: "constrained",
    2: "deficient",
}

DEFICIENT_KEYWORDS = [
    "cramming",
    "nag-cramming",
    "nagcram",
    "paper compliance",
    "rush compliance",
    "retroactive",
    "backdated",
    "inspection only",
    "folder preparation",
    "show for assessors",
    "last minute",
    "huling sandali",
    "failed indicator",
    "failed minimum",
    "non-functional",
    "ghost committee",
    "ghost council",
    "tabletop compliance",
    "for compliance only",
    "para sa compliance lang",
    "window dressing",
    "unaligned",
    "audit observation memorandum",
    "unresolved aom",
    "disallowance",
    "walang dokumento",
    "hindi documented",
    "walang documentation",
    "kulang sa documents",
    "nagmamadali",
    "bagsak sa assessment",
    "audit findings",
]

CONSTRAINED_KEYWORDS = [
    "budget constraint",
    "lack of funds",
    "fiscal limitation",
    "ira cut",
    "nta reduction",
    "unfilled plantilla",
    "understaffed",
    "staffing shortage",
    "overburdened",
    "equipment shortage",
    "logistical delay",
    "delayed download",
    "capacity gap",
    "partial implementation",
    "in progress",
    "job order turnover",
    "ad hoc",
    "manual tracking",
    "backlog",
    # Tagalog/Taglish resource constraints
    "kulang sa pondo",
    "walang pondo",
    "kinulang sa pondo",
    "kulang sa tao",
    "walang tao",
    "kinulang sa tao",
    "kulang sa gamit",
    "walang equipment",
    "na-delay",
    "nadelay",
    "pagkaantala",
    "mabagal ang downloading",
    "job order umalis",
    "nag-alisan ang jo",
    "manual lang",
    "nahihirapan",
    "problema sa budget",
    "kakulangan",
]

SUSTAINED_KEYWORDS = [
    "institutionalized",
    "ordinance enacted",
    "ordinance passed",
    "allocated budget",
    "year-round",
    "24/7 operations",
    "sop in place",
    "automated",
    "digitized",
    "fully functional",
    "citizen charter",
    "sustained delivery",
    "continuous monitoring",
    "permanent personnel",
    "seal recipient",
    "sglf incentive",
    "culture of service",
    "beyond compliance",
    # Tagalog/Taglish sustained governance
    "nainstitutionalized",
    "na-institutionalize",
    "tuloy-tuloy",
    "tuloy tuloy",
    "araw-araw",
    "kahit walang assessment",
    "kahit walang validation",
    "parating ginagawa",
    "may ordinansa",
    "naipasa ang ordinansa",
    "may sops",
    "ramdam ng tao",
    "tunay na serbisyo",
    "hindi lang papel",
    "para sa mamamayan",
    "taon-taon",
]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


class DistilBertService:
    """Manages DistilBERT ONNX runtime session, WordPiece tokenization, and inference."""

    def __init__(self, custom_model_dir: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.workspace_dir = self.base_dir.parent
        self.model_dir = self._resolve_model_dir(custom_model_dir)
        self.metrics_path = self._resolve_metrics_path()

        self.session: Optional[ Any ] = None
        self.torch_model: Optional[Any] = None
        self.torch: Optional[Any] = None
        self.fast_tokenizer: Optional[ Any ] = None
        self.hf_tokenizer: Optional[ Any ] = None
        self.config: Dict[str, Any] = {}
        self.id2label: Dict[int, str] = dict(DEFAULT_ID2LABEL)
        self.max_length: int = 256
        self.loaded: bool = False
        self.load_error: Optional[str] = None
        self.load_time_ms: float = 0.0

        self._load_model()

    def _resolve_model_dir(self, custom_dir: Optional[str] = None) -> Path:
        candidates = []
        if custom_dir:
            candidates.append(Path(custom_dir))
        env_dir = os.environ.get("DISTILBERT_MODEL_DIR")
        if env_dir:
            candidates.append(Path(env_dir))

        candidates.extend([
            self.base_dir / "model" / "distilbert_onnx",
            self.base_dir / "ml" / "distilbert_onnx",
        ])

        for candidate in candidates:
            if candidate.exists() and (
                (candidate / "onnx" / "model.onnx").exists()
                or (candidate / "model.onnx").exists()
                or (candidate / "model.safetensors").exists()
                or (candidate / "pytorch_model.bin").exists()
            ):
                return candidate
        return candidates[-1]

    def _resolve_metrics_path(self) -> Optional[Path]:
        candidates = [
            self.base_dir / "model" / "metrics.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_model(self) -> None:
        start = time.perf_counter()
        try:
            config_file = self.model_dir / "config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                raw_map = self.config.get("id2label", {})
                if raw_map:
                    self.id2label = {int(k): str(v).lower() for k, v in raw_map.items()}

            onnx_path = self.model_dir / "onnx" / "model.onnx"
            if not onnx_path.exists():
                onnx_path = self.model_dir / "model.onnx"

            if onnx_path.exists() and ort is not None:
                try:
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.session = ort.InferenceSession(
                        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
                    )
                except Exception as exc:
                    self.load_error = f"ONNX model could not load: {exc}"

            if self.session is None and any((self.model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
                try:
                    import torch
                    from transformers import AutoModelForSequenceClassification

                    self.torch = torch
                    self.torch_model = AutoModelForSequenceClassification.from_pretrained(
                        str(self.model_dir), local_files_only=True
                    ).eval()
                    self.load_error = None
                except Exception as exc:
                    self.load_error = f"PyTorch model could not load: {exc}"

            if self.session is None and self.torch_model is None:
                self.load_error = self.load_error or (
                    "onnxruntime is not installed" if onnx_path.exists() else
                    f"No ONNX or PyTorch model weights found in {self.model_dir}"
                )
                self.loaded = False
                return

            tokenizer_json = self.model_dir / "tokenizer.json"
            if FastTokenizer is not None and tokenizer_json.exists():
                self.fast_tokenizer = FastTokenizer.from_file(str(tokenizer_json))
            elif AutoTokenizer is not None:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_dir), local_files_only=True
                )
            else:
                self.load_error = "Neither tokenizers nor transformers library is available"
                self.loaded = False
                return

            self.loaded = True
            self.load_error = None
            self.load_time_ms = round((time.perf_counter() - start) * 1000.0, 2)
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)

    def get_model_status(self) -> Dict[str, Any]:
        """Return detailed metadata about the loaded DistilBERT model and training metrics."""
        onnx_path = self.model_dir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = self.model_dir / "model.onnx"
        onnx_data_path = self.model_dir / "onnx" / "model.onnx.data"

        size_bytes = 0
        if onnx_path.exists():
            size_bytes += onnx_path.stat().st_size
        if onnx_data_path.exists():
            size_bytes += onnx_data_path.stat().st_size
        if self.torch_model is not None:
            for name in ("model.safetensors", "pytorch_model.bin"):
                weights = self.model_dir / name
                if weights.exists():
                    size_bytes += weights.stat().st_size

        benchmark_metrics = None
        if self.metrics_path and self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    full_metrics = json.load(f)
                benchmark_metrics = {
                    "generated_at": full_metrics.get("generated_at"),
                    "evaluation_label": full_metrics.get("evaluation_label"),
                    "distilbert": full_metrics.get("distilbert"),
                    "tfidf_kmeans": (
                        full_metrics.get("experimental_baselines", {}).get("tfidf_kmeans")
                        or full_metrics.get("tfidf_kmeans")
                    ),
                }
            except Exception:
                benchmark_metrics = None

        return {
            "loaded": self.loaded,
            "load_error": self.load_error,
            "load_time_ms": self.load_time_ms,
            "model_dir": str(self.model_dir),
            "model_size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else 0,
            "architecture": (self.config.get("architectures") or ["DistilBertForSequenceClassification"])[0],
            "model_type": self.config.get("model_type", "distilbert"),
            "num_layers": self.config.get("n_layers", 6),
            "num_attention_heads": self.config.get("n_heads", 12),
            "hidden_dim": self.config.get("dim", 768),
            "ffn_hidden_dim": self.config.get("hidden_dim", 3072),
            "vocab_size": self.config.get("vocab_size", 30522),
            "max_position_embeddings": self.config.get("max_position_embeddings", 512),
            "max_sequence_length": self.max_length,
            "labels": self.id2label,
            "runtime": (
                "ONNX Runtime (CPUExecutionProvider)" if self.session is not None else
                "PyTorch Transformers (CPU)" if self.torch_model is not None else
                "Fallback Heuristic Engine"
            ),
            "taglish_support": {
                "active": True,
                "morphological_normalizer": True,
                "affix_handling": ["nag-", "mag-", "pag-", "pagka-", "na-", "ma-", "i-", "in-", "um-", "-an", "-in"],
                "bilingual_governance_glossary": True,
            },
            "benchmark_metrics": benchmark_metrics,
        }

    @staticmethod
    def build_standardized_input(record: Dict[str, Any]) -> str:
        """Construct standardized DistilBERT input sequence from structured governance fields."""
        governance_area = str(
            record.get("governance_area")
            or record.get("degree_program")
            or record.get("area")
            or record.get("pillar")
            or ""
        ).strip()
        office_role = str(
            record.get("office_role")
            or record.get("current_position")
            or record.get("position")
            or record.get("role")
            or ""
        ).strip()
        lgu = str(
            record.get("lgu")
            or record.get("current_company")
            or record.get("company")
            or record.get("municipality")
            or record.get("city")
            or ""
        ).strip()
        indicators = str(
            record.get("skills_indicators")
            or record.get("skills")
            or record.get("indicators")
            or record.get("subtheme")
            or ""
        ).strip()
        narrative = str(
            record.get("narrative_quote")
            or record.get("about")
            or record.get("bio")
            or record.get("response")
            or record.get("feedback")
            or record.get("narrative")
            or ""
        ).strip()
        headline = str(record.get("headline") or record.get("interview_topic") or "").strip()
        raw_text = str(record.get("text") or record.get("input_text") or "").strip()

        parts = []
        if lgu:
            parts.append(f"LGU: {lgu}")
        if governance_area:
            parts.append(f"Governance Area: {governance_area}")
        if office_role:
            parts.append(f"Office/Role: {office_role}")
        if headline:
            parts.append(f"Topic: {headline}")
        if indicators:
            parts.append(f"Indicators/Criteria: {indicators}")
        if narrative:
            parts.append(f"Narrative: {narrative}")

        if parts:
            if raw_text and raw_text not in " | ".join(parts):
                parts.append(f"Notes: {raw_text}")
            return " | ".join(parts)

        return raw_text

    def tokenize_text(self, text: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """Tokenize text into WordPiece input_ids, attention_mask, token strings, and raw IDs."""
        clean_text = text.strip() if text else ""
        if self.fast_tokenizer is not None:
            encoding = self.fast_tokenizer.encode(clean_text)
            tokens = encoding.tokens[: self.max_length]
            ids = encoding.ids[: self.max_length]
            if len(ids) == self.max_length and ids[-1] != 102:
                ids[-1] = 102
                tokens[-1] = "[SEP]"
            attn = [1] * len(ids)
            input_ids = np.array([ids], dtype=np.int64)
            attention_mask = np.array([attn], dtype=np.int64)
            return input_ids, attention_mask, tokens, list(ids)

        if self.hf_tokenizer is not None:
            encoded = self.hf_tokenizer(
                clean_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            raw_ids = input_ids[0].tolist()
            tokens = self.hf_tokenizer.convert_ids_to_tokens(raw_ids)
            return input_ids, attention_mask, tokens, raw_ids

        # Basic fallback tokenization if tokenizer is unavailable
        words = ["[CLS]"] + re.findall(r"\w+|[^\w\s]", clean_text.lower())[: self.max_length - 2] + ["[SEP]"]
        ids = [101] + [1000 + (hash(w) % 28000) for w in words[1:-1]] + [102]
        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        return input_ids, attention_mask, words, ids

    @staticmethod
    def classify_governance_institutionalization(record: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """Evaluate local governance indicators and operational trajectories alongside DistilBERT."""
        office_role = str(
            record.get("office_role")
            or record.get("current_position")
            or record.get("position")
            or record.get("role")
            or ""
        ).strip()
        lgu = str(
            record.get("lgu")
            or record.get("current_company")
            or record.get("company")
            or ""
        ).strip()
        area = str(
            record.get("governance_area")
            or record.get("degree_program")
            or record.get("area")
            or ""
        ).strip()
        indicators = str(
            record.get("skills_indicators")
            or record.get("skills")
            or record.get("indicators")
            or ""
        ).strip()
        narrative = str(
            record.get("narrative_quote")
            or record.get("about")
            or record.get("bio")
            or record.get("response")
            or ""
        ).strip()

        combined = " ".join(p for p in [office_role, lgu, area, indicators, narrative, input_text] if p).lower()
        taglish_active = is_taglish_text(combined)
        # Augment with normalized Taglish stems so 'nag-cramming' -> 'cramming', 'kinulang' -> 'kulang', etc.
        search_corpus = normalize_taglish_text(combined).lower()

        if not combined.strip():
            return {
                "status": "deficient",
                "aligned": False,
                "confidence": 0.95,
                "reason": "Empty governance profile / transcript record",
                "taglish_detected": False,
            }

        # Check Deficient / Cramming indicators
        for kw in DEFICIENT_KEYWORDS:
            if kw in search_corpus:
                return {
                    "status": "deficient",
                    "aligned": False,
                    "confidence": 0.92,
                    "reason": f"Detected compliance cramming or documentary deficiency indicator ('{kw}')",
                    "taglish_detected": taglish_active,
                }

        # Check Sustained / Institutionalized indicators
        for kw in SUSTAINED_KEYWORDS:
            if kw in search_corpus:
                return {
                    "status": "sustained",
                    "aligned": True,
                    "confidence": 0.92,
                    "reason": f"Detected institutionalized governance indicator ('{kw}')",
                    "taglish_detected": taglish_active,
                }

        # Check Constrained / In-Progress indicators
        for kw in CONSTRAINED_KEYWORDS:
            if kw in search_corpus:
                return {
                    "status": "constrained",
                    "aligned": False,
                    "confidence": 0.88,
                    "reason": f"Detected structural capacity or resource limitation ('{kw}')",
                    "taglish_detected": taglish_active,
                }

        # Default trajectory evaluation based on role/mandate presence
        if office_role or area:
            return {
                "status": "sustained",
                "aligned": True,
                "confidence": 0.78,
                "reason": "Active governance mandate / focal office specified",
                "taglish_detected": taglish_active,
            }

        return {
            "status": "constrained",
            "aligned": False,
            "confidence": 0.75,
            "reason": "General governance record without explicit institutionalization markers",
            "taglish_detected": taglish_active,
        }

    # Backward-compatible alias
    classify_degree_alignment = classify_governance_institutionalization

    def predict_single(self, record: Dict[str, Any], use_hybrid_rules: bool = True) -> Dict[str, Any]:
        """Run DistilBERT sequence inference and return logits, probabilities, tokens, and governance trajectory."""
        input_text = self.build_standardized_input(record)
        alignment = self.classify_governance_institutionalization(record, input_text)

        if not input_text.strip():
            return {
                "input_text": "",
                "distilbert_label": "deficient",
                "final_status": "deficient",
                "confidence": 1.0,
                "probabilities": {"sustained": 0.0, "constrained": 0.0, "deficient": 1.0},
                "logits": [-2.5, -2.5, 3.5],
                "token_count": 2,
                "tokens_preview": ["[CLS]", "[SEP]"],
                "token_ids_preview": [101, 102],
                "alignment": alignment,
                "latency_ms": 0.05,
                "hybrid_override": False,
            }

        start_time = time.perf_counter()
        input_ids, attention_mask, tokens, raw_ids = self.tokenize_text(input_text)

        if self.loaded and self.session is not None:
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            ort_outputs = self.session.run(["logits"], ort_inputs)
            raw_logits = ort_outputs[0][0].astype(np.float64)
            probs = softmax(raw_logits)
        elif self.loaded and self.torch_model is not None:
            with self.torch.no_grad():
                output = self.torch_model(
                    input_ids=self.torch.tensor(input_ids),
                    attention_mask=self.torch.tensor(attention_mask),
                )
            raw_logits = output.logits[0].detach().cpu().numpy().astype(np.float64)
            probs = softmax(raw_logits)
        else:
            # Deterministic fallback if ONNX session could not load
            status_map = {
                "sustained": [2.4, -0.8, -1.6],
                "constrained": [-0.5, 2.2, -1.2],
                "deficient": [-1.5, -0.9, 2.5],
            }
            raw_logits = np.array(status_map.get(alignment["status"], [0.0, 0.0, 0.0]), dtype=np.float64)
            probs = softmax(raw_logits)

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        prob_dict = {}
        for idx, prob in enumerate(probs):
            label_name = self.id2label.get(idx, f"label_{idx}")
            prob_dict[label_name] = round(float(prob), 4)

        best_idx = int(np.argmax(probs))
        distilbert_label = self.id2label.get(best_idx, "unknown")
        distilbert_conf = round(float(probs[best_idx]), 4)

        # Hybrid resolution combining DistilBERT sequence confidence with SGLG institutionalization rules
        final_status = distilbert_label
        hybrid_override = False
        if use_hybrid_rules:
            if distilbert_label == "sustained":
                if alignment["status"] == "deficient" and distilbert_conf < 0.90:
                    final_status = "deficient"
                    hybrid_override = True
                elif alignment["status"] == "constrained" and not alignment["aligned"]:
                    final_status = "constrained"
                    hybrid_override = True
            elif distilbert_label == "constrained":
                if alignment["status"] == "deficient":
                    final_status = "deficient"
                    hybrid_override = True

        return {
            "input_text": input_text,
            "distilbert_label": distilbert_label,
            "final_status": final_status,
            "confidence": distilbert_conf,
            "probabilities": prob_dict,
            "logits": [round(float(x), 4) for x in raw_logits.tolist()],
            "token_count": len(tokens),
            "tokens_preview": tokens[:64],
            "token_ids_preview": raw_ids[:64],
            "alignment": alignment,
            "latency_ms": latency_ms,
            "hybrid_override": hybrid_override,
        }
