"""Core RAG pipeline for Student Document Q&A (Tier 1 - Basic).

Pipeline: load -> chunk (embedding-model tokenizer, overlapping) -> embed
(local pretrained model) -> numpy store -> cosine top-k retrieval ->
LLM answer (any OpenAI-compatible endpoint) with citations.
"""

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Read settings from a .env file next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Embeddings run locally (no API key, no cost).
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The answer-generating LLM: any OpenAI-compatible endpoint (Groq, Ollama, OpenAI, Together...).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


@dataclass
class Index:
    chunks: list[str]
    embeddings: np.ndarray  # shape (n_chunks, dim), L2-normalized


def get_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Add it to your .env file "
            "(see .env.example) to enable answer generation."
        )
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


@lru_cache(maxsize=1)
def get_embedder():
    """Load the local embedding model once and reuse it."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    model.tokenizer.model_max_length = 10**9  # we chunk ourselves; silence length warning
    return model


# ---------- 1. Load ----------
def load_document(name: str, data: bytes) -> str:
    """Extract plain text from a .pdf or .txt upload."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


# ---------- 2. Chunk (L1 tokenization, overlapping windows) ----------
def chunk_text(text: str, chunk_tokens: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping windows measured in the embedding model's tokens.

    Chunks keep the original text (sliced by character offsets), so the
    citations shown to the user are readable and unmodified.
    """
    model = get_embedder()
    limit = model.max_seq_length - 2  # room for [CLS] and [SEP]
    if chunk_tokens > limit:
        raise ValueError(f"chunk_tokens must be <= {limit} for this embedding model")
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk_tokens")

    enc = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    step = chunk_tokens - overlap
    chunks = []
    for start in range(0, len(offsets), step):
        end = min(start + chunk_tokens, len(offsets))
        piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if end == len(offsets):
            break
    return chunks


# ---------- 3. Embed (L1 embeddings, pretrained, no training) ----------
def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    # normalize_embeddings=True makes cosine similarity a plain dot product.
    return get_embedder().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


# ---------- 4. Vector store (numpy) ----------
def build_index(text: str, chunk_tokens: int = 200, overlap: int = 40) -> Index:
    chunks = chunk_text(text, chunk_tokens, overlap)
    if not chunks:
        raise ValueError("No text could be extracted from the document.")
    return Index(chunks=chunks, embeddings=embed_texts(chunks))


# ---------- 5. Retrieve (cosine similarity top-k) ----------
def retrieve(index: Index, question: str, k: int = 4):
    q = embed_texts([question])[0]
    scores = index.embeddings @ q
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), index.chunks[i]) for i in top]


# ---------- 6. Generate (L2 LLM call, low temperature) ----------
SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the provided "
    "context passages. Cite the passages you used like [1], [2]. If the "
    "context does not contain the answer, say you could not find it in the document."
)


def answer_question(client: OpenAI, question: str, retrieved) -> str:
    context = "\n\n".join(
        f"[{n}] {text}" for n, (_, _, text) in enumerate(retrieved, start=1)
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content