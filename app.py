"""Streamlit UI for Student Document Q&A (Tier 1 - Basic)."""

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

uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded is not None:
    # Embed once per uploaded file; reuse the index for every question.
    if st.session_state.get("doc_name") != uploaded.name:
        with st.spinner("Loading embedding model, then chunking and embedding the document..."):
            text = rag.load_document(uploaded.name, uploaded.getvalue())
            try:
                st.session_state["index"] = rag.build_index(text)
                st.session_state["doc_name"] = uploaded.name
            except ValueError as e:
                st.error(str(e))
                st.stop()
    index = st.session_state["index"]
    st.success(f"Indexed **{uploaded.name}** into {len(index.chunks)} chunks.")

    k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)
    question = st.text_input("Ask a question about the document")

    if question:
        retrieved = rag.retrieve(index, question, k=k)

        st.subheader("Answer")
        if client is None:
            st.info("Set LLM_API_KEY in your .env file to get a generated answer.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    st.write(rag.answer_question(client, question, retrieved))
            except Exception as e:  # keep the sources visible even if the LLM call fails
                st.error(f"The LLM request failed: {e}")

        st.subheader("Sources")
        for n, (chunk_id, score, text) in enumerate(retrieved, start=1):
            with st.expander(f"[{n}] chunk #{chunk_id} — similarity {score:.3f}"):
                st.write(text)