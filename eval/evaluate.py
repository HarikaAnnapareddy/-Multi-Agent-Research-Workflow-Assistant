"""Baseline (retrieve + 1 LLM call) vs multi-agent graph.

Metrics per pipeline: mean latency and groundedness (share of answer sentences an
LLM judge says are supported by the retrieved evidence).
Run:  python -m eval.evaluate
Caveat: the judge is the same local model, so treat numbers as relative, not absolute.
"""
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from app.agents import build_graph, format_evidence
from app.llm import get_llm
from app.retrieval import HybridRetriever
from app.schemas import Evidence


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 4]


async def groundedness(answer: str, evidence: list[Evidence]) -> float:
    sents = sentences(answer)
    if not sents:
        return 0.0
    llm, ctx, ok = get_llm(), format_evidence(evidence), 0
    for s in sents:
        msg = await llm.ainvoke(
            f"EVIDENCE:\n{ctx}\n\nCLAIM: {s}\n\nIs the claim fully supported by the evidence? Answer only YES or NO."
        )
        ok += msg.content.strip().upper().startswith("YES")
    return ok / len(sents)


async def baseline(q: str, retriever: HybridRetriever):
    ev = [Evidence(ref=f"S{i+1}", source=e.source, text=e.text) for i, e in enumerate(retriever.search(q))]
    msg = await get_llm().ainvoke(f"Answer using the evidence.\n\nQuestion: {q}\n\nEvidence:\n{format_evidence(ev)}")
    return msg.content, ev


async def main():
    retriever = HybridRetriever()
    graph = build_graph(retriever)
    qs = [x["question"] for x in json.loads(Path("eval/questions.json").read_text())]
    rows = {"baseline": [], "multi-agent": []}
    for q in qs:
        t = time.perf_counter()
        ans, ev = await baseline(q, retriever)
        lat = time.perf_counter() - t
        rows["baseline"].append((lat, await groundedness(ans, ev)))

        t = time.perf_counter()
        st = await graph.ainvoke({"question": q, "trace": []})
        lat = time.perf_counter() - t
        rows["multi-agent"].append((lat, await groundedness(st["answer"], st["evidence"])))
        print(f"done: {q[:60]}")

    lines = ["| Pipeline | Avg latency (s) | Avg groundedness |", "|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {statistics.mean(x[0] for x in r):.1f} | {statistics.mean(x[1] for x in r):.0%} |")
    Path("eval/results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
