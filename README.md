# AI Document Q&A RAG System

Chat with a PDF from the command line. Extract text, chunk it, embed the
chunks locally, store them in ChromaDB, and answer questions over the
retrieved context with Gemini.

| file | what it is |
|---|---|
| `ingest.py` | extract a PDF, chunk it, embed the chunks, and store them in a local ChromaDB collection |
| `query.py` | embed a question, retrieve the closest chunks, and ask Gemini to answer from them |

---

## Setup

```powershell
cd "AI to Docs Project"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a Gemini API key from Google AI Studio.

## Usage

```powershell
python ingest.py   # point it at a PDF, builds ./vectordb
python query.py     # ask a question about the ingested PDF
```

`ingest.py` splits the document into overlapping character chunks, embeds
them with `all-MiniLM-L6-v2` (local, via `sentence-transformers`), and stores
them in a persistent ChromaDB collection. `query.py` embeds the question the
same way, retrieves the top 5 nearest chunks, and passes them as context to
Gemini for a grounded answer.
