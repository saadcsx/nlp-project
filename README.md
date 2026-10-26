# Student Document Q&A (Tier 1 – Basic)

Upload a PDF or `.txt` file, ask questions in plain English, and get a grounded answer with the source text shown next to it.

Built using only CME 295 Lectures 1–7 concepts:

| Step         | Implementation                                             | Lecture      |
| ------------ | ---------------------------------------------------------- | ------------ |
| Chunking     | `tiktoken` tokens, 400-token windows, 80-token overlap     | L1           |
| Embeddings   | Pretrained `text-embedding-3-small` (no training)          | L1           |
| Vector store | `numpy` array, cosine similarity top-k                     | L6 retrieval |
| Generation   | `gpt-4o-mini`, temperature 0, answers only from context    | L2           |
| Citation     | Retrieved chunks shown under the answer, cited as [1], [2] | L6           |

## Install

```bash
pip install -r requirements.txt
```

## Environment variables

| Variable         | Required | Default                  |
| ---------------- | -------- | ------------------------ |
| `OPENAI_API_KEY` | yes      | –                        |
| `EMBED_MODEL`    | no       | `text-embedding-3-small` |
| `LLM_MODEL`      | no       | `gpt-4o-mini`            |

Easiest: copy `.env.example` to `.env` and put your key in it. The app loads `.env` automatically (it is git-ignored, so the key is never committed).

```
OPENAI_API_KEY=your-key-here
```

Alternatively set it in the shell (`export OPENAI_API_KEY=...` in Git Bash/macOS/Linux, `$env:OPENAI_API_KEY="..."` in PowerShell).

## Run

```bash
python -m streamlit run app.py
```

Open the local URL, upload a document, and ask a question. The document is embedded once and reused for every question.

## Files

- `rag.py` – load, chunk, embed, retrieve, generate
- `app.py` – Streamlit interface
- `requirements.txt` – dependencies
- `.env.example` – template for your API key file
