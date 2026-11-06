"""LangGraph workflow: planner -> researcher -> summarizer -> reviewer -(retry)-> END."""
import asyncio
import time
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from app import config
from app.llm import get_llm
from app.retrieval import HybridRetriever
from app.schemas import Evidence, Plan, Review, ReviewOut


class State(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    evidence: list[Evidence]
    answer: str
    review: Review
    loops: int
    trace: list[dict]


def _timed(name: str, fn: Callable):
    """Wrap an async node so its duration is appended to state['trace']."""
    async def wrapper(state: State) -> State:
        t0 = time.perf_counter()
        out = await fn(state)
        trace = list(state.get("trace", [])) + [{"node": name, "seconds": round(time.perf_counter() - t0, 2)}]
        return {**out, "trace": trace}
    return wrapper


def format_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e.ref}] (source: {e.source})\n{e.text}" for e in evidence)


def build_graph(retriever: HybridRetriever):
    llm = get_llm()

    async def planner(state: State) -> State:
        prompt = (
            "Break this research question into 1-3 focused sub-questions that can be "
            f"answered from documents.\nQuestion: {state['question']}"
        )
        try:
            plan = await llm.with_structured_output(Plan).ainvoke(prompt)
            subs = [s for s in plan.sub_questions if s.strip()][:3] or [state["question"]]
        except Exception:  # small local models sometimes break JSON mode
            subs = [state["question"]]
        return {"sub_questions": subs, "loops": 0}

    async def researcher(state: State) -> State:
        # Retrieve for every sub-question concurrently
        results = await asyncio.gather(
            *[asyncio.to_thread(retriever.search, q) for q in state["sub_questions"]]
        )
        seen, evidence = set(), []
        for hits in results:
            for h in hits:
                if h.text not in seen:
                    seen.add(h.text)
                    evidence.append(Evidence(ref=f"S{len(evidence) + 1}", source=h.source, text=h.text))
        return {"evidence": evidence}

    async def summarizer(state: State) -> State:
        feedback = ""
        if state.get("review") and not state["review"].approved:
            feedback = "\nA reviewer found these problems, fix them:\n- " + "\n- ".join(state["review"].issues)
        prompt = (
            "Answer the question using ONLY the evidence below. Cite sources inline like [S1]. "
            "If the evidence is insufficient, say so instead of guessing.\n\n"
            f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state['evidence'])}{feedback}"
        )
        msg = await llm.ainvoke(prompt)
        return {"answer": msg.content}

    async def reviewer(state: State) -> State:
        prompt = (
            "You are a strict fact-checker. Check every claim in the ANSWER against the EVIDENCE. "
            "approved=true only if all claims are supported and cited.\n\n"
            f"EVIDENCE:\n{format_evidence(state['evidence'])}\n\nANSWER:\n{state['answer']}"
        )
        try:
            out = await llm.with_structured_output(ReviewOut).ainvoke(prompt)
            review = Review(approved=out.approved, issues=out.issues)
        except Exception:
            review = Review(approved=True, issues=["reviewer output unparseable; accepted by default"])
        return {"review": review, "loops": state.get("loops", 0) + 1}

    def route(state: State) -> str:
        if state["review"].approved or state["loops"] > config.MAX_REVIEW_LOOPS:
            return "done"
        return "retry"

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("researcher", researcher),
                     ("summarizer", summarizer), ("reviewer", reviewer)]:
        g.add_node(name, _timed(name, fn))
    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "summarizer")
    g.add_edge("summarizer", "reviewer")
    g.add_conditional_edges("reviewer", route, {"retry": "summarizer", "done": END})
    return g.compile()
