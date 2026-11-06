import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption("Planner → Researcher → Summarizer → Reviewer (LangGraph + Ollama, fully local)")

with st.sidebar:
    st.header("1. Add knowledge")
    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf and st.button("Ingest PDF"):
        r = requests.post(f"{API}/ingest/pdf", files={"file": (pdf.name, pdf.getvalue())}, timeout=300)
        st.success(r.json()) if r.ok else st.error(r.text)
    url = st.text_input("Or a web page URL")
    if url and st.button("Ingest URL"):
        r = requests.post(f"{API}/ingest/url", json={"url": url}, timeout=300)
        st.success(r.json()) if r.ok else st.error(r.text)
    try:
        st.metric("Chunks indexed", requests.get(f"{API}/health", timeout=5).json()["chunks_indexed"])
    except Exception:
        st.warning("API not reachable. Start it with uvicorn.")

st.header("2. Ask")
q = st.text_area("Research question", height=80)
if st.button("Run research", type="primary") and q:
    with st.spinner("Agents working..."):
        r = requests.post(f"{API}/research", json={"question": q}, timeout=900)
    if not r.ok:
        st.error(r.text)
    else:
        d = r.json()
        st.subheader("Answer")
        st.write(d["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total time (s)", d["total_seconds"])
        c2.metric("Review loops", d["review_loops"])
        c3.metric("Reviewer approved", "Yes" if d["review"]["approved"] else "No")
        if d["review"]["issues"]:
            st.warning("Reviewer notes:\n- " + "\n- ".join(d["review"]["issues"]))
        with st.expander("Sub-questions"):
            st.write(d["sub_questions"])
        with st.expander("Evidence"):
            for e in d["evidence"]:
                st.markdown(f"**[{e['ref']}]** `{e['source']}`")
                st.caption(e["text"])
        with st.expander("Agent trace"):
            st.table(d["trace"])
