# Chapter 3 Methodology & Defense Blueprint: DistilBERT NLP & Qualitative Thematic Analysis

> **Academic Guide & Oral Defense Preparation**  
> *Study Title:* Multi-Case Qualitative Analysis of Seal of Good Local Governance (SGLG) Institutionalization  
> *Methodology:* Computer-Assisted Qualitative Data Analysis (CAQDAS) Triangulated with DistilBERT Deep Learning  

---

## 1. Chapter 3 Methodology Text (Ready for Thesis Manuscript)

### 3.X Natural Language Processing & Computational Thematic Analysis Pipeline

To process, code, and synthesize the multi-case qualitative interview transcripts collected across participating Local Government Units (LGUs), this study utilized a hybrid **Computer-Assisted Qualitative Data Analysis (CAQDAS)** architecture integrating **DistilBERT** (*Distilled Bidirectional Encoder Representations from Transformers*) deep learning sequence classification with **Braun & Clarke's (2006, 2019, 2021) six-phase thematic analysis framework**.

Unlike traditional qualitative studies that rely purely on manual line-by-line annotation or unconstrained generative artificial intelligence, this study implemented a dual **deductive-inductive computational pipeline** designed to preserve evidentiary auditability, eliminate synthetic hallucinations, and maintain strict academic trustworthiness (*Lincoln & Guba, 1985*).

```
                      QUALITATIVE TRANSCRIPT (Taglish / English)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
    ┌──────────────────────────────┐        ┌──────────────────────────────┐
    │     DistilBERT Transformer   │        │   Taglish Linguistic Adapter │
    │       (Encoder-Classifier)   │        │     (Morphology & Lexicon)   │
    └───────────────┬──────────────┘        └───────────────┬──────────────┘
                    │                                       │
     Contextual Trajectory Logits            Affix Stripping & Root Matching
     [Sustained/Constrained/Deficient]       ['nag-cramming' -> 'cramming']
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   Thematic Coding Engine      │
                        │   • Deductive SGLG Codebook   │
                        │   • Unsupervised TF-IDF       │
                        │   • MiniBatch K-Means Clusters│
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │  Scholarly Narrative Synthesis│
                        │  (100% Verbatim Human Quotes) │
                        └───────────────────────────────┘
```

---

### 3.X.1 Deep Learning Model Specification: DistilBERT

The deep learning model deployed is an optimized **DistilBERT sequence classification model** (*Sanh et al., 2019*) exported in ONNX (*Open Neural Network Exchange*) runtime format. DistilBERT retains 97% of BERT's language understanding capabilities while offering a 40% reduction in parameter footprint and 60% faster inference latency.

1. **Architecture Specifications:**
   * **Model Architecture:** `DistilBertForSequenceClassification`
   * **Layers ($N$):** 6 Transformer encoder blocks (distilled from BERT's 12 layers)
   * **Multi-Head Attention:** 12 attention heads per layer ($d_k = 64$)
   * **Hidden Dimension ($d$):** 768-dimensional contextual representation
   * **Feed-Forward Dimension ($d_{ff}$):** 3,072 dimensions with Gaussian Error Linear Unit (GELU) activation
   * **Vocabulary Size:** 30,522 WordPiece subword tokens
   * **Max Sequence Length ($L$):** 256 tokens

2. **Mathematical Formulation of Trajectory Classification:**
   Given a standardized qualitative transcript sequence $\mathbf{x} = (x_1, x_2, \dots, x_L)$, the model computes WordPiece embeddings and passes them through 6 bidirectional Transformer layers:
   $$\mathbf{H} = \text{TransformerEncoder}(\mathbf{E}_{\text{token}} + \mathbf{E}_{\text{pos}})$$
   The summary representation $\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{768}$ is extracted at index 0 and projected through a classification head:
   $$\mathbf{z} = \mathbf{W} \mathbf{h}_{[\text{CLS}]} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{3 \times 768}$$
   Softmax probabilities for governance trajectories $y \in \{\text{Sustained}, \text{Constrained}, \text{Deficient}\}$ are computed via:
   $$P(y = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^{3} \exp(z_j)}$$

---

### 3.X.2 Thematic Coding and Code Generation

The system implements a dual **Deductive Codebook** and **Inductive Discovery** engine:

1. **Deductive Framework Coding:**  
   Excerpts are matched against the 6 core SGLG governance themes (Leadership Direction, Interdepartmental Coordination, Operational Challenges, Institutionalization, DILG Assistance, and Sustainable Governance). Keywords carrying multi-word specificity receive differential analytical weighting.

2. **Inductive Statistical Discovery:**  
   * **TF-IDF Keyphrase Extraction:** Extracts unigrams and bigrams across the corpus, pruning functional stop words.
   * **MiniBatch K-Means Clustering:** Groups semantically adjacent responses into $k=4$ emergent clusters without researcher preconception.

3. **Emergent Code Fallback & Reflexive Safeguard:**  
   Participant statements that do not match predefined codebook keywords are automatically processed by an emergent term extractor, assigned candidate qualitative codes (labeled `[Emergent]`), and categorized under `"Unassigned / Needs Review"`. This ensures **no record is forced into an incorrect theme** and preserves the researcher's reflexive review.

---

### 3.X.3 Bilingual Taglish Processing & Morphological Normalizer

In Philippine local governance, stakeholders express administrative experiences in **Taglish** (Tagalog-English code-switching). To bridge the gap between English WordPiece tokenization and Philippine vernacular discourse, the system incorporates an integrated **Taglish Morphological Normalizer**:

1. **Morphological Affix Stripping:**  
   Tagalog verbal and nominal affixes attached to English loanwords are normalized in real-time:
   * *Prefixes:* `nag-cramming` $\rightarrow$ `cramming`; `nadelay` $\rightarrow$ `delay`; `pag-audit` $\rightarrow$ `audit`
   * *Infixes:* `kinulang` $\rightarrow$ `kulang`; `ginawa` $\rightarrow$ `gawa`
   * *Circumfixes & Suffixes:* `kakulangan` $\rightarrow$ `kulang`; `ordinansahan` $\rightarrow$ `ordinansa`
2. **Anchor Term Recognition:**  
   Taglish statements are heavily anchored by technical governance nouns (*"budget"*, *"ordinance"*, *"assessment"*, *"focal person"*, *"cramming"*). DistilBERT's 12 attention heads attend to these anchor terms across the sentence context.
3. **Bilingual Stop-Word Filtration:**  
   Over 80 Tagalog functional markers (`ang`, `mga`, `sa`, `ng`, `dahil`, `para`, `pero`, `naman`, `lang`, `po`, `natin`) are purged during TF-IDF vectorization, ensuring data-driven clusters represent substantive governance issues rather than grammatical noise.

---

## 2. Oral Defense Preparation: Panelist Questions & Answers

### Q1: "What is DistilBERT's exact role in your Natural Language Processing (NLP) pipeline? Did it write the narrative?"

> **Recommended Answer:**  
> *"No, sir/ma'am. DistilBERT did not write the narrative, and that was a deliberate methodological choice. DistilBERT is an **Encoder model** (specifically `DistilBertForSequenceClassification`), which means its function is **discriminative, not generative**. It uses 12 bidirectional self-attention heads to read the semantic context of each participant's statement and classify its institutional governance trajectory into one of three states: **Sustained/Institutionalized**, **Constrained/In-Progress**, or **Deficient/Unaligned**.  
> The codes and academic narrative synthesis are assembled by the qualitative CAQDAS engine using DistilBERT's contextual trajectory to contextualize verified, verbatim participant quotes. This ensures complete auditability and academic rigor."*

---

### Q2: "Why didn't you just use ChatGPT, Gemini, or a Generative LLM to do the coding and write the findings?"

> **Recommended Answer:**  
> *"In academic qualitative research—especially for graduate theses—unconstrained generative LLMs pose a severe threat to **evidentiary trustworthiness** (*Lincoln & Guba, 1985*). Generative models suffer from **hallucinations**: they can invent quotes that no participant ever uttered, fabricate nonexistent municipal initiatives, or alter the meaning of stakeholder testimony.  
> By deploying DistilBERT as a trajectory classifier combined with a deterministic CAQDAS framework, **100% of the quotations in our study are genuine, verbatim excerpts directly traceable to specific informant IDs** (e.g., LCE-03, MPDC-02, MLGOO-01). The AI serves as an analytical assistant, not a ghostwriter."*

---

### Q3: "DistilBERT was trained on English. How can it validly understand Taglish (Tagalog-English) interview transcripts?"

> **Recommended Answer:**  
> *"Our system addresses Taglish through a **two-layer linguistic adapter**:  
> First, Philippine local governance discourse is inherently anchored in English technical terminology—terms like 'cramming', 'budget', 'ordinance', 'assessment', 'focal team', and 'audit'. DistilBERT's attention mechanism easily latches onto these anchor nouns.  
> Second, our system implements a **Taglish Morphological Normalizer** in Python (`engine/taglish_normalizer.py`). When a participant says 'Nag-cramming kami dahil kinulang sa pondo', the normalizer strips the Tagalog prefix 'nag-' to expose 'cramming' and the infix '-in-' from 'kinulang' to expose 'kulang'. Furthermore, 80+ Tagalog stop words (`ang`, `mga`, `sa`, `ng`, `dahil`) were filtered out so our statistical TF-IDF clusters remain pristine."*

---

### Q4: "How does your system conform to Braun & Clarke's Thematic Analysis standards?"

> **Recommended Answer:**  
> *"Our system aligns with Braun & Clarke's (2006, 2019) 6 phases:  
> 1. **Familiarization:** Through dataset file ingestion, field mapping, and previewing raw interview text.  
> 2. **Initial Coding:** The system extracts specific empirical code labels and candidate emergent codes.  
> 3. **Searching for Themes:** It maps codes into the 6 SGLG core themes while discovering emergent clusters via K-Means.  
> 4. **Reviewing Themes:** Through cross-tabulation against respondent groups (Mayors, Focal Persons, DILG Officers) and trajectory statuses.  
> 5. **Defining Themes:** Using the customizable codebook editor with analytic statements and benchmarks.  
> 6. **Producing the Report:** Generating the scholarly narrative synthesis draft following the 5-step academic sequence with embedded informant voices."*

---

### Q5: "What was your role as the researcher? Doesn't the computer do everything?"

> **Recommended Answer:**  
> *"Qualitative research requires **reflexivity**. The system acts strictly as **Computer-Assisted Qualitative Data Analysis Software (CAQDAS)**.  
> When an excerpt contains novel ideas not in the codebook, the system does not force it into an arbitrary theme; it flags it as **'Unassigned / Needs Review'** with candidate emergent codes. I, as the researcher, have the final authority to inspect the raw transcripts, apply manual theme overrides in the deep inspector, modify the codebook, and contextualize the findings with field observations from municipal visits."*
