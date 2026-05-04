"""Section-aware extraction for 10-K / 10-Q filings.

EDGAR filings are notoriously messy HTML: tables, nested divs, inline XBRL,
boilerplate header/footers. We don't try to be perfect — the goal is to
isolate the high-signal sections (Item 1A Risk Factors, Item 7 MD&A, Item 8
Financial Statements) so retrieval doesn't dilute against table-of-contents
boilerplate.

Strategy:
    1. Strip via selectolax (fast, lenient HTML5).
    2. Linearize text, keeping anchor positions.
    3. Regex-locate "Item 1A.", "Item 7.", "Item 8." headings; the FIRST
       occurrence is the TOC, the SECOND is the section body. We use the
       second-or-later match per item, then run to the next "Item N" boundary.
    4. Trim, collapse whitespace, return Sections with char_start/char_end so
       chunker output can be traced back to source offsets.

This is intentionally string-based, not DOM-based: it's robust across the
filing styles (modern HTML, legacy HTML, EDGAR-converted DOCX) and avoids
the DOM-traversal hellscape unique to each filer's template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from selectolax.parser import HTMLParser

# Map section code → friendly title + heading regex.
# Heading regex tolerates italics, decorative spaces, line breaks.
_ITEM_PATTERNS: dict[str, tuple[str, re.Pattern[str]]] = {
    "1A": (
        "Risk Factors",
        re.compile(r"\bItem\s*1A\.?\s*Risk\s*Factors\b", re.IGNORECASE),
    ),
    "7": (
        "Management's Discussion and Analysis",
        re.compile(r"\bItem\s*7\.?\s*Management['’]s\s+Discussion\s+and\s+Analysis", re.IGNORECASE),
    ),
    "7A": (
        "Quantitative and Qualitative Disclosures About Market Risk",
        re.compile(r"\bItem\s*7A\.?\s*Quantitative", re.IGNORECASE),
    ),
    "8": (
        "Financial Statements",
        re.compile(r"\bItem\s*8\.?\s*Financial\s+Statements", re.IGNORECASE),
    ),
}

# Outer "Item N" boundary used to terminate a section. Permissive on purpose.
_ITEM_BOUNDARY = re.compile(r"\bItem\s*\d+[A-Z]?\.\s+[A-Z]", re.IGNORECASE)


@dataclass(slots=True)
class Section:
    item_code: str
    title: str
    text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class ParsedFiling:
    sections: list[Section]
    full_text: str


def _html_to_text(html: str) -> str:
    tree = HTMLParser(html)
    # Standard CSS selectors (selectolax does not accept `:` in tag names).
    for sel in ("script", "style", "noscript", "head"):
        for n in tree.css(sel):
            n.decompose()
    # XBRL inline tags (ix:header, ix:hidden, etc.) — strip via tag-name match.
    if tree.body is not None:
        for n in tree.body.iter():
            tag = n.tag or ""
            if tag.startswith("ix:") or tag in ("ix:header", "ix:hidden"):
                n.decompose()
    text = tree.body.text(separator=" ") if tree.body else tree.text(separator=" ")
    # Collapse whitespace; keep newlines as boundaries for chunker.
    text = re.sub(r" ", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_section(
    text: str,
    pattern: re.Pattern[str],
    item_code: str,
) -> tuple[int, int] | None:
    """Locate (start, end) char offsets for an item.

    Heuristic: the FIRST pattern match is usually inside the table of contents
    (heading appears as a hyperlink with no body following). The SECOND match
    is the section body. If only one match exists, treat it as body.

    The end is set to the next "Item N." boundary occurring at least
    `min_body_len` chars after start (to skip immediately-adjacent items in
    the TOC), or end-of-text.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    body_match = matches[1] if len(matches) >= 2 else matches[0]
    start = body_match.start()
    min_body_len = 800
    boundary = _ITEM_BOUNDARY.search(text, pos=start + min_body_len)
    end = boundary.start() if boundary else len(text)
    if end - start < 200:
        return None
    return start, end


def parse_10k_html(html: str, *, items: tuple[str, ...] = ("1A", "7", "7A", "8")) -> ParsedFiling:
    full = _html_to_text(html)
    sections: list[Section] = []
    for code in items:
        title, pattern = _ITEM_PATTERNS[code]
        span = _find_section(full, pattern, code)
        if span is None:
            continue
        start, end = span
        sections.append(
            Section(
                item_code=code,
                title=title,
                text=full[start:end].strip(),
                char_start=start,
                char_end=end,
            )
        )
    return ParsedFiling(sections=sections, full_text=full)
