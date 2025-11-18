# RAG Agent (LangGraph + LangChain + Chroma)

This repo is a ready-to-run scaffold tailored to your interview task PDF. It includes:
- A Jupyter notebook to ingest your docs, build a Chroma vector DB with **SentenceTransformers** embeddings, and run a **LangGraph**-powered RAG agent.
- A Python script (`app.py`) that builds the same graph for quick CLI/manual testing.
- `TASK.md` with the auto-extracted requirements for review.

## Quickstart

1. Create a virtual env and install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Put your source documents into `data/` (your uploaded PDF is already copied as `data/source.pdf`).

3. Open the notebook:
   ```bash
   jupyter lab  # or jupyter notebook
   ```

4. Run the notebook cells to:
   - Load & split documents
   - Build embeddings & Chroma store
   - Construct the LangGraph
   - Query the agent

## Notes
- The default embedding model is **all-MiniLM-L6-v2** from `sentence-transformers`—no external API keys required.
- Swap in OpenAI or other providers if needed by editing the notebook or `app.py`.
