"""
Minimal RAG pipeline using LangGraph + LangChain + Chroma with local embeddings.
Run:
    python app.py
Then type your question when prompted.
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from rich.logging import RichHandler

# LangChain / LangGraph / Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rag")

# ---------- Config ----------
DATA_DIR = os.getenv("DATA_DIR", "Data")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# ---------- Helpers ----------
def load_documents(data_dir: str) -> List[Document]:
    docs: List[Document] = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, fname))
            docs.extend(loader.load())
        elif fname.lower().endswith((".txt", ".md")):
            docs.extend(TextLoader(os.path.join(data_dir, fname), encoding="utf-8").load())
    return docs

def build_vectorstore(docs: List[Document]):
    if not docs:
        raise ValueError("No documents found in data/. Add PDFs or text files.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)
    embedder = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = Chroma.from_documents(splits, embedder, persist_directory=DB_DIR)
    vs.persist()
    return vs

# ---------- Graph State ----------
class AgentState(BaseModel):
    """A simple state for RAG: stores messages and working memory."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)

# ---------- Nodes ----------
def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from Chroma based on last user question."""
    question = ""
    for m in reversed(state.messages):
        if m.get("role") == "user":
            question = m.get("content", "")
            break
    if not question:
        return state

    client = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = Chroma(persist_directory=DB_DIR, embedding_function=client)
    retrieved = vs.similarity_search(question, k=5)
    ctx = [f"[{i+1}] {d.page_content}" for i, d in enumerate(retrieved)]
    log.info(f"Retrieved {len(ctx)} chunks")
    state.context = ctx
    return state

def answer_node(state: AgentState) -> AgentState:
    """Compose a simple answer from retrieved context (no external LLM by default)."""
    question = ""
    for m in reversed(state.messages):
        if m.get("role") == "user":
            question = m.get("content", "")
            break

    if not state.context:
        answer = "I couldn't find relevant context. Try rephrasing or adding more documents."
    else:
        summary = "\\n\\n".join(state.context)
        answer = f"Q: {question}\\n\\nTop context:\\n{summary}\\n\\nDraft answer (summarize and cite the numbered chunks above)."

    state.messages.append({"role": "assistant", "content": answer})
    return state

# ---------- Graph Wiring ----------
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)
app = graph.compile()

def main():
    # Build (or load) vectorstore
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        log.info("Building vectorstore...")
        docs = load_documents(DATA_DIR)
        _ = build_vectorstore(docs)
        log.info("Vectorstore built.")
    else:
        log.info("Using existing vectorstore at %s", DB_DIR)

    # Simple REPL
    log.info("RAG agent ready. Type your question (or 'exit'):")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        state = AgentState(messages=[{"role": "user", "content": q}])
        out = app.invoke(state)
        print("\nAnswer:\n" + out["messages"][-1]["content"] + "\n")

if __name__ == "__main__":
    main()
