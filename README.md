# 🚀 RAGfy Docs

> **AI-Powered Document Intelligence Platform**  
> Upload PDFs. Ask questions. Get precise, source-grounded answers.

---

## What is RAGfy?

RAGfy is a Retrieval-Augmented Generation (RAG) chatbot that lets you have intelligent conversations with your PDF documents. It combines semantic search, keyword search, and TF-IDF reranking to find the most relevant content — then uses Mistral AI to generate accurate, context-grounded answers.

---

## Features

- 📄 **Multi-PDF Upload** — Process multiple PDFs in one go
- 🔍 **Hybrid Retrieval** — Combines semantic (vector) search + BM25 keyword search
- 📊 **TF-IDF Reranking** — scikit-learn powered reranker with bigram support
- 🔁 **Deduplication** — Removes duplicate chunks before ranking
- 💬 **Chat History** — Keeps track of your Q&A session
- 📌 **Source Attribution** — Shows which document and page each answer came from

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral AI (`mistral-large-latest`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (LangChain Community) |
| Reranking | scikit-learn TF-IDF cosine similarity |
| PDF Loading | LangChain `PyPDFLoader` |
| Frontend | Streamlit |
| Orchestration | LangChain |

---

## Project Structure

```
RAG3/
├── main.py              # Main Streamlit app
├── .env                 # API keys (never commit this)
├── requirement.txt     # Python dependencies
├── chroma_db/           # Auto-created vector store (gitignored)
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
(https://github.com/shikhargthub/RAGfy-Docs)
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux

```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your key from [console.mistral.ai](https://console.mistral.ai).

### 5. Run the app

```bash
streamlit run main.py
```

---

## How It Works

```
User Query
    │
    ▼
Query Rewrite (Mistral LLM)
    │
    ▼
Hybrid Retrieval
  ├── Semantic Search (ChromaDB + MiniLM embeddings)
  └── Keyword Search (BM25)
    │
    ▼
Deduplication
    │
    ▼
TF-IDF Reranking (scikit-learn, top 3 chunks)
    │
    ▼
Context Assembly (with source + page metadata)
    │
    ▼
LLM Answer Generation (Mistral)
    │
    ▼
Response + Retrieved Chunks displayed in UI
```

---

## Requirement

```
streamlit
langchain
langchain-community
langchain-huggingface
chromadb
sentence-transformers
pypdf
mistralai
scikit-learn
```
---

## Usage

1. Launch the app with `streamlit run main.py`
2. Upload one or more PDF files using the file uploader
3. Click **Process PDFs** and wait for the success message
4. Type your question in the text input and click **Ask**
5. View the answer, source chunks, and chat history

---

## Environment Variables

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Your Mistral AI API key from console.mistral.ai |

---

## Notes

- The `chroma_db/` folder is auto-created on first run and reset on each new upload
- Embeddings run locally via HuggingFace — no external API call needed for retrieval
- The app does not persist chat history across sessions (in-memory only)

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Author

Built by **Shikhar Gupta**  
📧 shikharkumargupta143@gmail.com
🔗 (https://github.com/shikhargthub)
