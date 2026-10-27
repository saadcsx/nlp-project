"""Streamlit UI for Student Document Q&A (Tier 2 - Intermediate)."""

import hashlib
import time

import streamlit as st

import rag

st.set_page_config(page_title="Student Document Q&A", page_icon="📚")
st.title("📚 Student Document Q&A")
st.caption("Upload a PDF or .txt file, then ask questions. Answers cite the source text.")

# Embeddings are local, so only answer generation needs an LLM key.
try:
    client = rag.get_client()
except RuntimeError as e:
    client = None
    st.warning(f"{e} Retrieval will still work, but answers won't be generated.")

with st.sidebar:
    st.header("Settings")
    chunk_tokens = st.slider("Chunk size (tokens)", 50, 250, rag.DEFAULT_CHUNK_TOKENS, step=10)
    overlap = st.slider(
        "Chunk overlap (tokens)",
        0,
        min(100, chunk_tokens - 10),
        min(rag.DEFAULT_OVERLAP, chunk_tokens - 10),
        step=10,
    )
    k = st.slider("Chunks to retrieve (top-k)", 1, 8, rag.DEFAULT_TOP_K)
    compact = st.checkbox("Remove near-duplicate chunks", value=True)
    threshold = st.slider(
        "Duplicate threshold (cosine)",
        0.80,
        0.99,
        rag.DEFAULT_DEDUPE_THRESHOLD,
        step=0.01,
        disabled=not compact,
    )

uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded is not None:
    data = uploaded.getvalue()
    # Embed once per (document, chunk settings); reuse the index for every question.
    key = (hashlib.md5(data).hexdigest(), chunk_tokens, overlap)
    if st.session_state.get("index_key") != key:
        with st.spinner("Loading embedding model, then chunking and embedding the document..."):
            t0 = time.perf_counter()
            text = rag.load_document(uploaded.name, data)
            try:
                st.session_state["index"] = rag.build_index(text, chunk_tokens, overlap)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            st.session_state["embed_seconds"] = time.perf_counter() - t0
            st.session_state["index_key"] = key
    index = st.session_state["index"]
    st.success(
        f"Indexed **{uploaded.name}** into {len(index.chunks)} chunks "
        f"in {st.session_state['embed_seconds']:.2f}s "
        "(the first run includes loading the embedding model)."
    )

    question = st.text_input("Ask a question about the document")

    if question:
        t0 = time.perf_counter()
        retrieved, dropped = rag.retrieve(index, question, k=k, compact=compact, threshold=threshold)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        st.subheader("Answer")
        llm_seconds = None
        if client is None:
            st.info("Set LLM_API_KEY in your .env file to get a generated answer.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    t0 = time.perf_counter()
                    answer = rag.answer_question(client, question, retrieved)
                    llm_seconds = time.perf_counter() - t0
                st.write(answer)
            except Exception as e:  # keep the sources visible even if the LLM call fails
                st.error(f"The LLM request failed: {e}")

        timing = f"Retrieval: {retrieve_ms:.0f} ms"
        if llm_seconds is not None:
            timing += f" · Answer generation: {llm_seconds:.2f} s"
        if compact:
            timing += f" · Near-duplicate chunks removed: {dropped}"
        st.caption(timing)

        st.subheader("Sources")
        for n, (chunk_id, score, text) in enumerate(retrieved, start=1):
            with st.expander(f"[{n}] chunk #{chunk_id} — similarity {score:.3f}"):
                st.write(text)