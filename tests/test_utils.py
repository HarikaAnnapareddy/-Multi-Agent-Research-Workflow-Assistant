import pytest

from app.utils import chunk_text, rrf, tokenize


def test_chunk_overlap_and_coverage():
    text = "word " * 1000
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_empty_and_bad_overlap():
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_rrf_prefers_docs_in_both_rankings():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    ids = [d for d, _ in fused]
    assert ids[0] == "a" and set(ids) == {"a", "b", "c", "d"}


def test_tokenize():
    assert tokenize("Hello, World-2!") == ["hello", "world", "2"]
