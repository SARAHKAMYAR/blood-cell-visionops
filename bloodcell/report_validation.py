"""Deterministic verification of LLM-written report narratives.

The LLM is only allowed to *explain* measurements; every number it writes must
be traceable to the structured payload produced by the detector and QC layer.
No second model is involved: verification is rule-based and reproducible.

Checks
------
1. Traceability: every numeric token in the narrative must equal a payload
   value (or a value derived from it, e.g. a class share in %) at the
   precision it is written with. Untraceable numbers are fabrications.
2. Contextual consistency: numbers attached to a cell class ("17 RBCs",
   "WBC count: 2") or to the total ("detected 19 cells") must equal *that*
   specific count, not merely some number in the payload.
3. Completeness: the summary must state the total detection count and the QC
   status.
4. Scope: clinical/diagnostic terms are rejected (the system is not a
   diagnostic device).
5. Markup: links, images, URLs and HTML are rejected. The narrative is
   rendered as Markdown in the dashboard, and the filename (user input) is
   part of the prompt, so this blocks prompt-injected tracking images,
   phishing links and script URLs.

Known limitations: numbers written as words ("sixteen") are not checked, and
contextual patterns cover common phrasings rather than all of English. A
rejected narrative never reaches the user; the deterministic template is used.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from bloodcell.config import CLASS_NAMES
from bloodcell.schemas import ReportValidation


class LLMNarrative(BaseModel):
    """Structured output requested from the LLM: prose only, no tables."""

    summary: str = Field(min_length=20, max_length=900, description="2-3 sentence overview")
    quality_assessment: str = Field(
        min_length=10, max_length=1200, description="Plain-language explanation of the QC result"
    )
    reviewer_actions: list[str] = Field(
        min_length=1, max_length=5, description="Concrete checks for a human reviewer"
    )


CLASS_TERMS = {
    "RBC": r"(?:RBCs?|red\s+(?:blood\s+)?cells?|erythrocytes?)",
    "WBC": r"(?:WBCs?|white\s+(?:blood\s+)?cells?|leu[ck]ocytes?)",
    "Platelets": r"(?:platelets?|thrombocytes?)",
}

CLINICAL_TERMS = re.compile(
    r"\b(?:an(?:a)?emi[ac]\w*|leuk(?:a)?emi\w*|lymphoma\w*|thrombocytop\w*|thrombocytosis|"
    r"leukocytosis|leukopenia|neutropeni\w*|sepsis|septic|infection\w*|malaria\w*|"
    r"sickle[- ]cell|cancer\w*|malignan\w*|prognos\w*)\b",
    re.IGNORECASE,
)

MARKUP = re.compile(
    r"https?://|\bwww\.|\b(?:javascript|data|vbscript|file):|!\[|\]\s*\(|<\s*/?\s*[A-Za-z!]",
    re.IGNORECASE,
)

NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s?%|\s?percent\b)?")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
DIMENSIONS = re.compile(r"(\d)\s*[x×]\s*(\d)")


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
def _walk_numbers(value) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_numbers(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _walk_numbers(v)
    elif isinstance(value, str):
        # Numbers inside explanatory strings we supplied (e.g. QC rule text).
        for m in NUMBER.finditer(value):
            yield float(m.group(1))


def collect_facts(payload: dict) -> list[float]:
    """All numbers the narrative may legitimately cite."""
    facts = set(_walk_numbers({k: v for k, v in payload.items() if k not in {"filename", "model"}}))
    total = payload.get("total_detections") or 0
    if total:
        for n in payload.get("counts", {}).values():
            facts.add(100.0 * n / total)  # class share in %
        qc = payload.get("quality_control", {})
        facts.add(100.0 * qc.get("low_confidence_count", 0) / total)
    return sorted(facts)


def _matches(value: float, decimals: int, is_percent: bool, facts: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9  # consistent with rounding to the shown precision
    for f in facts:
        candidates = [f]
        if is_percent and 0.0 <= f <= 1.0:
            candidates.append(f * 100.0)
        if any(abs(value - c) <= tol for c in candidates):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _normalise(text: str, payload: dict) -> str:
    for literal in (payload.get("filename"), payload.get("model")):
        if literal:
            text = text.replace(str(literal), " ")
    text = LIST_MARKER.sub("", text)
    return DIMENSIONS.sub(r"\1 x \2", text)


def _contextual_issues(text: str, payload: dict) -> list[str]:
    issues: list[str] = []
    counts = payload.get("counts", {})
    for cls, term in CLASS_TERMS.items():
        expected = counts.get(cls, 0)
        patterns = (
            rf"(?<![\w.])(\d+)\s+{term}\b",
            rf"\b{term}\s*(?:counts?\s*)?(?::|=|\(|\bof\b|\bwas\b|\bwere\b|\bis\b|\bare\b|\btotal(?:l)?ing\b)\s*(\d+)(?![\d%]|\.\d)",
        )
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                claimed = int(m.group(1))
                if claimed != expected:
                    issues.append(f"inconsistent {cls} count: narrative says {claimed}, detector reports {expected}")
    total = payload.get("total_detections", 0)
    total_patterns = (
        r"(?<![\w.])(\d+)\s+(?:cells|objects|detections)\s+(?:were\s+|in\s+total\s+)?(?:detected|identified|found)",
        r"\b(?:detected|identified|found)\s+(?:a\s+total\s+of\s+)?(\d+)\s+(?:cells|objects|detections)\b",
        r"\btotal(?:\s+of)?\s+(\d+)(?![\d%]|\.\d)",
        r"(?<![\w.])(\d+)\s+total\b",
    )
    for pattern in total_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            claimed = int(m.group(1))
            if claimed != total:
                issues.append(f"inconsistent total: narrative says {claimed}, detector reports {total}")
    return issues


def verify_narrative(narrative: LLMNarrative, payload: dict, llm_model: str | None = None) -> ReportValidation:
    raw_text = "\n".join([narrative.summary, narrative.quality_assessment, *narrative.reviewer_actions])
    text = _normalise(raw_text, payload)
    facts = collect_facts(payload)
    issues: list[str] = []

    checked = 0
    for m in NUMBER.finditer(text):
        token = m.group(1)
        decimals = len(token.split(".")[1]) if "." in token else 0
        checked += 1
        if not _matches(float(token), decimals, bool(m.group(2)), facts):
            issues.append(f"untraceable number '{m.group(0).strip()}' (not in detector/QC output)")

    issues += _contextual_issues(text, payload)

    summary = _normalise(narrative.summary, payload)
    total = payload.get("total_detections", 0)
    if total and not re.search(rf"(?<![\w.]){total}(?![\d%]|\.\d)", summary):
        issues.append(f"summary omits the total detection count ({total})")
    status = str(payload.get("quality_control", {}).get("status", ""))
    if status and status.lower() not in summary.lower():
        issues.append(f"summary omits the QC status ({status})")

    for m in CLINICAL_TERMS.finditer(raw_text):
        issues.append(f"out-of-scope clinical term '{m.group(0)}'")
    if MARKUP.search(raw_text):
        issues.append("links, images, URLs or HTML are not allowed in the narrative")

    # De-duplicate while keeping order.
    issues = list(dict.fromkeys(issues))
    return ReportValidation(
        status="rejected" if issues else "verified",
        numbers_checked=checked,
        issues=issues,
        llm_model=llm_model,
    )


__all__ = ["LLMNarrative", "verify_narrative", "collect_facts", "CLASS_NAMES"]
