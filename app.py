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

st.set_page_config(page_title="Offline PDF Chat", page_icon="📄")
st.title("📄 Offline PDF Assistant")
st.write("Upload a PDF and ask questions about it — runs fully locally with Ollama.")


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


def format_docs(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        label = f"[Page {page + 1}]" if isinstance(page, int) else ""
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is None:
    st.session_state.pop("file_hash", None)
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

st.success(f"PDF processed: {st.session_state.n_pages} pages, {len(st.session_state.splits)} chunks. Ask away!")

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


def get_context(question: str) -> str:
    q = question.lower()
    if any(h in q for h in GENERAL_HINTS):
        # Overview questions: give the model the beginning of the document
        # (similarity search works badly for "what is this about?")
        docs = st.session_state.splits[:TOP_K]
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(question)
    return format_docs(docs)


# ---------------- Chat UI ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            context = get_context(question)
            answer = st.write_stream(chain.stream({"context": context, "question": question}))
        except Exception as e:
            answer = f"Error talking to Ollama: {e}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})