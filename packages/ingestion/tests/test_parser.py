from equityiq_ingestion import parse_10k_html


# Minimal 10-K-shaped HTML: TOC pointing at items, then bodies.
_FIXTURE = """
<html><body>
<h1>Annual Report on Form 10-K</h1>
<p>Table of Contents</p>
<ul>
  <li>Item 1A. Risk Factors</li>
  <li>Item 7. Management's Discussion and Analysis of Financial Condition</li>
  <li>Item 8. Financial Statements and Supplementary Data</li>
</ul>

<h2>Item 1A. Risk Factors</h2>
<p>{risks}</p>

<h2>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>{mda}</p>

<h2>Item 8. Financial Statements and Supplementary Data</h2>
<p>{fin}</p>
</body></html>
""".format(
    risks=("Our business faces material risks. " * 60),
    mda=("Net revenue increased 12 percent year over year. " * 60),
    fin=("Consolidated balance sheet and statement of operations. " * 60),
)


def test_extracts_three_canonical_sections():
    parsed = parse_10k_html(_FIXTURE)
    codes = {s.item_code for s in parsed.sections}
    assert {"1A", "7", "8"} <= codes


def test_section_text_contains_body_not_toc():
    parsed = parse_10k_html(_FIXTURE)
    risks = next(s for s in parsed.sections if s.item_code == "1A")
    assert "material risks" in risks.text


def test_returns_empty_when_no_items_present():
    parsed = parse_10k_html("<html><body><p>nothing here</p></body></html>")
    assert parsed.sections == []


def test_offsets_index_into_full_text():
    parsed = parse_10k_html(_FIXTURE)
    for s in parsed.sections:
        assert parsed.full_text[s.char_start : s.char_end].startswith(s.text[:30])
