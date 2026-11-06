import chromadb
from rank_bm25 import BM25Okapi

from app import config
from app.llm import get_embeddings
from app.schemas import Chunk, Evidence
from app.utils import rrf, tokenize


class HybridRetriever:
    """Dense (Chroma + Ollama embeddings) + sparse (BM25), fused with RRF."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.col = self.client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._meta: dict[str, tuple[str, str]] = {}
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        data = self.col.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._meta = {i: (d, m["source"]) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        corpus = [tokenize(d) for d in data["documents"]]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = get_embeddings().embed_documents([c.text for c in chunks])
        self.col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[{"source": c.source} for c in chunks],
        )
        self._rebuild_bm25()
        return len(chunks)

    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Evidence]:
        if not self._ids:
            return []
        rankings: list[list[str]] = []
        if mode in ("hybrid", "dense"):
            qv = get_embeddings().embed_query(query)
            res = self.col.query(query_embeddings=[qv], n_results=min(k * 2, len(self._ids)))
            rankings.append(res["ids"][0])
        if mode in ("hybrid", "sparse") and self._bm25:
            scores = self._bm25.get_scores(tokenize(query))
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k * 2]
            rankings.append([self._ids[i] for i in top if scores[i] > 0])
        fused = rrf(rankings)[:k]
        return [Evidence(ref="", source=self._meta[i][1], text=self._meta[i][0]) for i, _ in fused]

    def count(self) -> int:
        return self.col.count()
