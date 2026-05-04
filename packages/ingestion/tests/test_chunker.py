from equityiq_ingestion import SemanticChunker


def test_empty_input_returns_no_chunks():
    c = SemanticChunker(target_tokens=50, overlap_tokens=8, max_tokens=80)
    assert c.chunk("") == []
    assert c.chunk("   \n  ") == []


def test_short_text_yields_one_chunk():
    c = SemanticChunker(target_tokens=200, overlap_tokens=20, max_tokens=300)
    chunks = c.chunk("Apple reported earnings of $94B. Margins expanded.")
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert "Apple" in chunks[0].text
    assert chunks[0].tokens > 0


def test_long_text_splits_with_overlap():
    sentence = "This is a moderately long financial sentence about revenue. "
    text = sentence * 60  # ~600+ tokens
    c = SemanticChunker(target_tokens=80, overlap_tokens=16, max_tokens=120)
    chunks = c.chunk(text)
    assert len(chunks) >= 4
    assert all(ch.tokens <= 120 for ch in chunks)
    # Overlap → adjacent chunks share at least one sentence prefix.
    assert any(chunks[i].text.split(".")[0] in chunks[i + 1].text for i in range(len(chunks) - 1))


def test_oversized_single_sentence_emits_alone():
    c = SemanticChunker(target_tokens=20, overlap_tokens=4, max_tokens=40)
    huge = "word " * 200  # one "sentence" by our regex (no terminal punct)
    chunks = c.chunk(huge.strip() + ".")
    assert len(chunks) >= 1


def test_overlap_must_be_smaller_than_target():
    import pytest

    with pytest.raises(ValueError):
        SemanticChunker(target_tokens=50, overlap_tokens=50, max_tokens=80)
