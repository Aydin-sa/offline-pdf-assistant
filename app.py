"""
Offline PDF Assistant — chat with your PDFs fully locally (Streamlit + LangChain + Ollama).

Author: Aydin Sasanian
Date:   November 2026
"""

import os
import hashlib
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- Settings ----------------
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TOP_K = 6

USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🤖"

st.set_page_config(
    page_title="Offline PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 820px;}

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(139,124,255,0.18), rgba(0,201,167,0.10));
        border: 1px solid rgba(139,124,255,0.30);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #B4A9FF, #5EEAD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {margin: 0.4rem 0 0.9rem 0; color: #B8BCCB; font-size: 1rem;}
    .pill {
        display: inline-block;
        padding: 0.18rem 0.7rem;
        margin: 0 0.35rem 0.3rem 0;
        border-radius: 999px;
        font-size: 0.78rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #D5D8E5;
    }

    .empty-card {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: #1A1D2B;
        border: 1px dashed rgba(255,255,255,0.15);
        color: #B8BCCB;
    }
    .empty-card b {color: #E6E8F0;}

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
    }
    [data-testid="stFileUploader"] section {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>📄 Offline PDF Assistant</h1>
        <p>Upload a PDF and ask questions about it — everything runs locally with Ollama.</p>
        <span class="pill">🔒 100% local</span>
        <span class="pill">🧠 {LLM_MODEL}</span>
        <span class="pill">🧬 {EMBED_MODEL}</span>
        <span class="pill">📚 RAG · top {TOP_K}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- PDF processing (cached per file) ----------------
def build_vectorstore(file_bytes: bytes, file_hash: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    # Drop empty pages (e.g. scanned images without a text layer)
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        return None, 0, []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    # Unique collection per file -> no mixing with old PDFs or duplicate chunks
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=f"pdf_{file_hash[:16]}",
    )
    return vectorstore, len(docs), splits


def page_label(doc):
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else None


def format_docs(docs):
    parts = []
    for d in docs:
        p = page_label(d)
        label = f"[Page {p}]" if p else ""
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def render_sources(sources):
    if not sources:
        return
    pages = sorted({s["page"] for s in sources if s["page"]})
    title = "📚 Sources" + (f" · pages {', '.join(map(str, pages))}" if pages else "")
    with st.expander(title):
        for s in sources:
            st.markdown(f"**Page {s['page'] or '?'}**")
            st.caption(s["snippet"])


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    st.markdown(f"**LLM:** `{LLM_MODEL}`  \n**Embeddings:** `{EMBED_MODEL}`  \n**Chunks per answer:** `{TOP_K}`")

    if "n_pages" in st.session_state and st.session_state.get("file_hash"):
        st.divider()
        st.markdown("### 📊 Document")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.n_pages)
        c2.metric("Chunks", len(st.session_state.splits))
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. The PDF text is extracted and split into chunks.\n"
            "2. Chunks are embedded and stored in Chroma.\n"
            "3. Your question retrieves the most relevant chunks.\n"
            "4. The local LLM answers using only those chunks."
        )
    st.caption("Make sure Ollama is running: `ollama serve`")
    st.divider()
    st.caption("Built by **Aydin Sasanian** · November 2026")


# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is None:
    st.session_state.pop("file_hash", None)
    st.markdown(
        """
        <div class="empty-card">
            <b>👋 Get started</b><br><br>
            1. Drop a PDF above (text-based, not scanned)<br>
            2. Wait a moment while it gets indexed<br>
            3. Ask anything — e.g. <i>"What is this about?"</i> or <i>"Summarize chapter 2"</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

# Only process when a NEW file is uploaded (Streamlit reruns the whole script on every input)
if st.session_state.get("file_hash") != file_hash:
    with st.spinner("Reading and indexing the PDF... please wait"):
        try:
            vs, n_pages, splits = build_vectorstore(file_bytes, file_hash)
        except Exception as e:
            st.error(f"Could not process the PDF / reach Ollama: {e}")
            st.info(f"Is Ollama running? Try: `ollama pull {EMBED_MODEL}` and `ollama pull {LLM_MODEL}`")
            st.stop()

    if vs is None:
        st.error("No text found in this PDF. It is probably scanned (images only) and needs OCR first.")
        st.stop()

    st.session_state.file_hash = file_hash
    st.session_state.vectorstore = vs
    st.session_state.splits = splits
    st.session_state.n_pages = n_pages
    st.session_state.messages = []
    st.rerun()  # refresh so the sidebar shows the document stats

st.success(
    f"✅ **{uploaded_file.name}** is ready — {st.session_state.n_pages} pages, "
    f"{len(st.session_state.splits)} chunks. Ask away!"
)

# ---------------- LLM + prompt ----------------
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions about a PDF document. "
     "Use ONLY the context excerpts below. If the question is general "
     "(e.g. 'what is this about?', 'summarize'), summarize the main topics you see in the context. "
     "If the context truly does not contain the answer, say: "
     "\"I couldn't find that in the uploaded PDF.\" "
     "Answer in English and mention page numbers when useful.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

GENERAL_HINTS = ("what is this", "about", "summar", "overview", "main topic", "topic of")


def get_context(question: str):
    q = question.lower()
    if any(h in q for h in GENERAL_HINTS):
        # Overview questions: give the model the beginning of the document
        # (similarity search works badly for "what is this about?")
        docs = st.session_state.splits[:TOP_K]
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(question)
    sources = [
        {"page": page_label(d), "snippet": d.page_content[:300].strip() + "…"}
        for d in docs
    ]
    return format_docs(docs), sources


# ---------------- Chat UI ----------------
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if question := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    sources = []
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            context, sources = get_context(question)
            answer = st.write_stream(chain.stream({"context": context, "question": question}))
            render_sources(sources)
        except Exception as e:
            answer = f"Error talking to Ollama: {e}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
