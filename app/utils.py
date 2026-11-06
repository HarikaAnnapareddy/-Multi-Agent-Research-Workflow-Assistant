"""Pure helpers (no heavy deps) so they are easy to unit test."""
import re


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: merges several ranked id lists into one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
