# 📄 Offline PDF Assistant

Chat with your PDF documents — **fully local**, no cloud API, no data leaves your machine.
Built with [Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com) and [Ollama](https://ollama.com).

## How it works

1. **Upload** a PDF in the web UI.
2. The text is extracted (`PyPDFLoader`) and split into overlapping chunks (1000 chars, 200 overlap).
3. Each chunk is embedded with `nomic-embed-text` and stored in a **Chroma** vector store (one collection per file, keyed by its hash).
4. When you ask a question, the most relevant chunks (top 6) are retrieved and passed to **`llama3.2`** as context (RAG).
5. Overview questions ("what is this about?", "summarize") use the beginning of the document instead of similarity search.
6. Answers are streamed back and cite page numbers where useful.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

## Setup

```bash
# 1. Pull the models
ollama pull nomic-embed-text
ollama pull llama3.2

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Configuration

Change the models or retrieval depth at the top of `app.py`:

```python
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TOP_K = 6
```

## Limitations

- Scanned PDFs (images without a text layer) need OCR first — the app will tell you if no text is found.
- Answer quality depends on the local model; larger models give better answers but need more RAM.

## Tech stack

Python · Streamlit · LangChain · Ollama · ChromaDB

## Author

**Aydin Sasanian** — November 2026
GitHub: [@Aydin-sa](https://github.com/Aydin-sa)
