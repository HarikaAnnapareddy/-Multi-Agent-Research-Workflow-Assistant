import time

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import ingestion
from app.agents import build_graph
from app.retrieval import HybridRetriever
from app.schemas import (IngestResponse, ResearchRequest, ResearchResponse,
                         TraceStep, UrlIngestRequest)

app = FastAPI(title="Multi-Agent Research Assistant")
retriever = HybridRetriever()
graph = build_graph(retriever)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": retriever.count()}


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    chunks = ingestion.load_pdf(await file.read(), file.filename)
    if not chunks:
        raise HTTPException(422, "No extractable text found (scanned PDF?)")
    return IngestResponse(source=file.filename, chunks_added=retriever.add(chunks))


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(req: UrlIngestRequest):
    try:
        chunks = ingestion.load_url(str(req.url))
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch URL: {exc}")
    return IngestResponse(source=str(req.url), chunks_added=retriever.add(chunks))


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    if retriever.count() == 0:
        raise HTTPException(400, "Index is empty. Ingest a PDF or URL first.")
    t0 = time.perf_counter()
    state = await graph.ainvoke({"question": req.question, "trace": []})
    return ResearchResponse(
        question=req.question,
        sub_questions=state["sub_questions"],
        answer=state["answer"],
        evidence=state["evidence"],
        review=state["review"],
        review_loops=state["loops"],
        trace=[TraceStep(**t) for t in state["trace"]],
        total_seconds=round(time.perf_counter() - t0, 2),
    )
