"""Core RAG pipeline for Student Document Q&A (Tier 2 - Intermediate).

Pipeline: load -> chunk (embedding-model tokenizer, overlapping) -> embed
(local pretrained model, batched) -> numpy store -> cosine top-k retrieval ->
context compaction (drop near-duplicates, respect a token budget) ->
LLM answer (any OpenAI-compatible endpoint, temperature 0, fixed seed) with citations.
"""

import io
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pypdf import PdfReader

# Read settings from a .env file next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------- Reproducibility (fixed seeds) ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- Models ----------
# Embeddings run locally (no API key, no cost).
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The answer-generating LLM: any OpenAI-compatible endpoint (Groq, Ollama, OpenAI, Together...).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Fixed sampling seed for reproducible answers (set LLM_SEED= (empty) to disable).
LLM_SEED = os.getenv("LLM_SEED", str(SEED))

# ---------- Tunable defaults (update these after running `python eval/run_eval.py --tune`) ----------
DEFAULT_CHUNK_TOKENS = 200
DEFAULT_OVERLAP = 40
DEFAULT_TOP_K = 4
DEFAULT_DEDUPE_THRESHOLD = 0.92  # cosine similarity above which a chunk counts as a near-duplicate
FETCH_FACTOR = 2  # fetch k * FETCH_FACTOR candidates so compaction can still return k chunks
MAX_CONTEXT_TOKENS = 1500  # context budget sent to the LLM (L2: respect context length)


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
    """Load the local embedding model once and reuse it (L5: never reload per question)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    model.tokenizer.model_max_length = 10**9  # we chunk ourselves; silence length warning
    return model


def count_tokens(text: str) -> int:
    """Number of tokens in the embedding model's tokenizer."""
    return len(get_embedder().tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------- 1. Load ----------
def load_document(name: str, data: bytes) -> str:
    """Extract plain text from a .pdf or .txt upload."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


# ---------- 2. Chunk (L1 tokenization, overlapping windows) ----------
def chunk_text(
    text: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
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


# ---------- 3. Embed (L1 embeddings, pretrained, batched per L5) ----------
def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    # Batching sends many texts through the model at once (L5). Normalizing makes
    # cosine similarity a plain dot product.
    return get_embedder().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


# ---------- 4. Vector store (numpy) ----------
def build_index(
    text: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS, overlap: int = DEFAULT_OVERLAP
) -> Index:
    chunks = chunk_text(text, chunk_tokens, overlap)
    if not chunks:
        raise ValueError("No text could be extracted from the document.")
    return Index(chunks=chunks, embeddings=embed_texts(chunks))


# ---------- 5. Retrieve (cosine similarity top-k) + context compaction (L6) ----------
def compact_chunks(
    index: Index,
    candidates,
    k: int,
    threshold: float = DEFAULT_DEDUPE_THRESHOLD,
    max_tokens: int = MAX_CONTEXT_TOKENS,
):
    """Context compaction: walk candidates best-first and keep at most k chunks.

    A chunk is dropped when it is a near-duplicate (cosine >= threshold) of a chunk
    already kept. Stops early if the token budget would be exceeded.
    Returns (kept, n_dropped_as_duplicates).
    """
    kept, dropped, used = [], 0, 0
    for chunk_id, score, text in candidates:
        if len(kept) >= k:
            break
        vec = index.embeddings[chunk_id]
        if any(float(vec @ index.embeddings[kid]) >= threshold for kid, _, _ in kept):
            dropped += 1
            continue
        n_tokens = count_tokens(text)
        if kept and used + n_tokens > max_tokens:
            break
        kept.append((chunk_id, score, text))
        used += n_tokens
    return kept, dropped


def retrieve(
    index: Index,
    question: str,
    k: int = DEFAULT_TOP_K,
    compact: bool = True,
    threshold: float = DEFAULT_DEDUPE_THRESHOLD,
    q_vec=None,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
):
    """Return ([(chunk_id, score, text), ...], n_duplicates_dropped).

    Pass q_vec to reuse a precomputed (batched) question embedding.
    """
    q = q_vec if q_vec is not None else embed_texts([question])[0]
    scores = index.embeddings @ q
    fetch = min(len(scores), k * FETCH_FACTOR if compact else k)
    order = np.argsort(scores)[::-1][:fetch]
    candidates = [(int(i), float(scores[i]), index.chunks[i]) for i in order]
    if not compact:
        return candidates[:k], 0
    return compact_chunks(index, candidates, k, threshold, max_context_tokens)


# ---------- 6. Generate (L2 LLM call: low temperature, fixed seed) ----------
# The system prompt is identical for every question, so providers can reuse the
# cached prefix (L5: KV-cache awareness). Question-specific text comes after it.
SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the provided "
    "context passages. Cite the passages you used like [1], [2]. If the "
    "context does not contain the answer, say you could not find it in the document."
)


def answer_question(client: OpenAI, question: str, retrieved, max_retries: int = 3) -> str:
    context = "\n\n".join(
        f"[{n}] {text}" for n, (_, _, text) in enumerate(retrieved, start=1)
    )
    kwargs = {"seed": int(LLM_SEED)} if LLM_SEED.strip() else {}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))  # back off 2s, 4s on rate limits