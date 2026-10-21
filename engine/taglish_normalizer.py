"""Taglish Morphological Normalizer & Linguistic Processing Engine.

Provides Tagalog affix stripping (prefixes, infixes, suffixes), Taglish loanword normalization
(e.g., 'nag-cramming' -> 'cramming', 'nadelay' -> 'delay', 'kinulang' -> 'kulang'),
and comprehensive Tagalog functional stop words for qualitative thematic coding.
"""

import re
from typing import List, Set

# Comprehensive Tagalog functional particles, ligatures, pronouns, and grammatical markers
TAGALOG_STOP_WORDS: Set[str] = {
    # Functional markers & ligatures
    "ang", "mga", "sa", "ng", "ni", "nina", "kay", "kina", "si", "sina", "ay", "na", "nang",
    # Conjunctions & transition words
    "at", "o", "pero", "ngunit", "subalit", "dapat", "kasi", "dahil", "para", "kung", "kapag",
    "pag", "upang", "habang", "bago", "pagkatapos", "kaya", "sakaling", "samantala", "bagaman",
    # Particles & enclitics
    "naman", "lang", "lamang", "din", "rin", "daw", "raw", "po", "opo", "ho", "oho", "ba",
    "pa", "pala", "nga", "man", "sana", "talaga", "yata", "mismo", "tuloy", "kahit",
    # Pronouns
    "ko", "mo", "niya", "namin", "natin", "ninyo", "nila",
    "ako", "ikaw", "ka", "siya", "kami", "tayo", "kayo", "sila",
    "akin", "iyo", "kaniya", "atin", "inyo", "kanila",
    # Demonstratives
    "ito", "iyan", "iyon", "dito", "diyan", "doon", "nito", "niyan", "niyon",
    "ganito", "ganiyan", "ganoon",
    # Question words
    "ano", "sino", "saan", "kailan", "bakit", "paano", "ilan", "alin",
    # Existentials & quantifiers
    "may", "mayroon", "wala", "meron", "lahat", "bawat", "iba", "sarili",
    "isa", "dalawa", "tatlo", "una", "huli", "mas", "sobra", "masyado", "muli",
    # Temporal & relational adverbs
    "agad", "palagi", "lagi", "minsan", "noon", "ngayon", "bukas", "kahapon", "dati",
    "parang", "tulad", "katulad", "mukhang", "medyo", "halos", "basta",
}

# Tagalog/Taglish marker indicators to test if a passage contains Tagalog
TAGALOG_MARKERS: Set[str] = {
    "ang", "mga", "sa", "ng", "dahil", "para", "pero", "hindi", "namin", "natin",
    "kami", "sila", "nila", "po", "opo", "lang", "naman", "talaga", "kahit",
    "wala", "meron", "mayroon", "kulang", "pondo", "pamahalaan", "ordinansa",
    "ugnayan", "pulong", "bawat", "opisina", "tulong", "gabay", "serbisyo",
    "mamamayan", "tao", "ramdam",
}

# Common Tagalog/Taglish prefix patterns (with optional hyphen for English loanwords)
PREFIX_RE = re.compile(
    r"^(?:nag|mag|pag|pagka|ipag|ipinag|i|na|ma|ka|pang|pan|pam|taga)-?(?=[a-z]{3,})",
    re.IGNORECASE,
)

# Infix patterns (e.g. -in- or -um- inside a root word like kinulang -> kulang, pumasa -> pasa)
INFIX_IN_RE = re.compile(r"^([bcdfghjklmnpqrstvwxyz])in([aeiou][a-z]{2,})$", re.IGNORECASE)
INFIX_UM_RE = re.compile(r"^([bcdfghjklmnpqrstvwxyz])um([aeiou][a-z]{2,})$", re.IGNORECASE)

# Common suffix patterns
SUFFIX_RE = re.compile(r"(?:an|in|han|hin)$", re.IGNORECASE)


def is_taglish_text(text: str) -> bool:
    """Detect if an excerpt contains Tagalog or Taglish markers."""
    if not text:
        return False
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    matches = words.intersection(TAGALOG_MARKERS)
    return len(matches) >= 2 or any(
        w.startswith(("nag-", "mag-", "pag-", "na-", "i-", "nainstitutional"))
        for w in words
    )


def extract_taglish_stem_variants(word: str) -> List[str]:
    """Given a single word, return the word plus possible Tagalog/Taglish stem variations."""
    w = word.strip().lower()
    if len(w) < 4:
        return [w]

    variants = {w}

    # 1. Hyphenated loanwords: e.g. 'nag-cramming' -> 'cramming', 'na-delay' -> 'delay'
    if "-" in w:
        parts = w.split("-")
        if len(parts) == 2 and parts[1]:
            variants.add(parts[1])
            # e.g., 'nag-co-coordinate'
            if len(parts[1]) > 3:
                variants.add(re.sub(r"^([a-z]{2,3})\1", r"\1", parts[1]))

    # 2. Tagalog prefix stripping: e.g., 'nagcramming' -> 'cramming', 'nadelay' -> 'delay'
    prefix_match = PREFIX_RE.match(w)
    if prefix_match:
        stem = w[prefix_match.end():]
        if len(stem) >= 3:
            variants.add(stem)
            # Tagalog reduplication: e.g., 'pagko-coordinate' or 'nagcocomply' -> 'comply'
            if len(stem) >= 4 and stem[:2] == stem[2:4]:
                variants.add(stem[2:])

    # 3. Infix stripping: e.g., 'kinulang' -> 'kulang', 'ginawa' -> 'gawa'
    m_in = INFIX_IN_RE.match(w)
    if m_in:
        variants.add(m_in.group(1) + m_in.group(2))

    m_um = INFIX_UM_RE.match(w)
    if m_um:
        variants.add(m_um.group(1) + m_um.group(2))

    # 4. Circumfix/Suffix stripping: e.g., 'kakulangan' -> 'kulang', 'ordinansahan' -> 'ordinansa'
    if w.startswith("ka") and w.endswith("an") and len(w) >= 6:
        stem = w[2:-2]
        if len(stem) >= 3:
            variants.add(stem)

    return list(variants)


def normalize_taglish_text(text: str) -> str:
    """Return text augmented with stripped Taglish stems so regex/token matching succeeds."""
    if not text:
        return ""

    tokens = re.findall(r"\b[a-zA-Z0-9\-]+\b", text)
    additional_stems = []

    for t in tokens:
        variants = extract_taglish_stem_variants(t)
        for v in variants:
            if v != t.lower() and len(v) >= 3:
                additional_stems.append(v)

    if additional_stems:
        return text + " " + " ".join(additional_stems)
    return text


def extract_salient_terms(text: str, top_n: int = 4) -> List[str]:
    """Extract candidate qualitative codes from text when no predefined codebook keywords match."""
    if not text or len(text.strip()) < 4:
        return []

    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower())
    filtered = [
        t for t in tokens
        if t not in TAGALOG_STOP_WORDS and len(t) > 3 and not t.isdigit()
    ]

    if not filtered:
        return []

    from collections import Counter
    counts = Counter(filtered)

    # Also build bigrams
    bigrams = []
    for i in range(len(filtered) - 1):
        bigram = f"{filtered[i]} {filtered[i + 1]}"
        bigrams.append(bigram)

    candidate_codes = []
    for bg, _ in Counter(bigrams).most_common(2):
        candidate_codes.append(bg)

    for word, _ in counts.most_common(top_n):
        if not any(word in c for c in candidate_codes):
            candidate_codes.append(word)
        if len(candidate_codes) >= top_n:
            break

    return candidate_codes[:top_n]
