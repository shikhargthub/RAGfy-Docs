import streamlit as st
import tempfile
import os
import shutil
from typing import List

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# scikit-learn reranker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="🤖")
st.title("🚀 RAGfy Docs")
st.write("AI-Powered Document Intelligence Platform")

# ----------------------------
# LLM
# ----------------------------
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# ----------------------------
# Embeddings
# ----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------------
# Text Splitter
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# ----------------------------
# Prompt
# ----------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful assistant.
Use ONLY the provided context.
If answer not found, say:
"I could not find the answer in the documents provided."
"""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# ----------------------------
# Session state
# ----------------------------
for key in ["vectorstore", "bm25", "chat_history", "chunks"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

# ----------------------------
# PDF Upload + Processing
# ----------------------------
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing documents..."):
            all_docs = []

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    path = tmp.name

                loader = PyPDFLoader(path)
                pages = loader.load()
                os.unlink(path)  # FIX: clean up temp file after loading

                for i, page in enumerate(pages):
                    page.metadata["source"] = file.name
                    page.metadata["page"] = i + 1

                all_docs.extend(pages)

            chunks = splitter.split_documents(all_docs)

            # FIX: wipe old Chroma DB before reprocessing to avoid stale embedding conflicts
            chroma_path = "chroma_db"
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=chroma_path
            )

            # FIX: set explicit k so BM25 doesn't return all chunks
            bm25 = BM25Retriever.from_documents(chunks, k=6)

            st.session_state.vectorstore = vectorstore
            st.session_state.bm25 = bm25
            st.session_state.chunks = chunks  # stored for reranker corpus

        st.success(f"✅ Processed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# ----------------------------
# Query Rewrite
# ----------------------------
def rewrite_query(query: str) -> str:
    response = llm.invoke([
        ("system",
         "You are a search query optimizer. Rewrite the user's question into a "
         "clear, specific query that will retrieve the most relevant documents. "
         "Return ONLY the rewritten query, no explanation."),
        ("human", query)
    ])
    return response.content.strip()

# ----------------------------
# Deduplication
# ----------------------------
def deduplicate(docs: List) -> List:
    """Remove duplicate chunks by page_content hash."""
    seen = set()
    unique = []
    for doc in docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique

# ----------------------------
# TF-IDF Reranker (scikit-learn)
# ----------------------------
def rerank_docs_tfidf(query: str, docs: List, top_k: int = 3) -> List:
    """
    Rerank documents using TF-IDF cosine similarity.
    Much better than simple term overlap — accounts for term frequency
    and inverse document frequency across the candidate set.
    """
    if not docs:
        return []

    contents = [doc.page_content for doc in docs]

    # Fit TF-IDF on candidate docs + query together
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams for richer matching
        max_features=10000
    )

    corpus = contents + [query]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    doc_vectors = tfidf_matrix[:-1]   # all rows except last
    query_vector = tfidf_matrix[-1]   # last row is the query

    scores = cosine_similarity(query_vector, doc_vectors).flatten()

    # Pair and sort by score descending
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in ranked[:top_k]]

# ----------------------------
# Query Input
# ----------------------------
query = st.text_input("Ask a question about your documents")

if st.button("Ask") and query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process PDFs first.")
    else:
        with st.spinner("Thinking..."):

            # Step 1: Rewrite query
            rewritten_query = rewrite_query(query)
            st.caption(f"🔍 Rewritten query: *{rewritten_query}*")

            # Step 2: Hybrid retrieval
            semantic_docs = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 6}
            ).invoke(rewritten_query)

            keyword_docs = st.session_state.bm25.invoke(rewritten_query)

            # Step 3: Deduplicate merged results
            merged = deduplicate(semantic_docs + keyword_docs)

            # Step 4: TF-IDF rerank
            top_docs = rerank_docs_tfidf(rewritten_query, merged, top_k=3)

            # Step 5: Build context
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source','?')}, "
                f"Page: {doc.metadata.get('page','?')}]\n{doc.page_content}"
                for doc in top_docs
            ])

            # Step 6: LLM answer
            messages = prompt_template.format_messages(
                context=context,
                question=query
            )
            response = llm.invoke(messages)
            answer = response.content

        # Save + display
        st.session_state.chat_history.append((query, answer))

        st.subheader("🧠 Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Chunks"):
            for i, doc in enumerate(top_docs, 1):
                st.markdown(
                    f"**Chunk {i}** — `{doc.metadata.get('source','?')}` "
                    f"p.{doc.metadata.get('page','?')}"
                )
                st.write(doc.page_content[:400] + "...")

# ----------------------------
# Chat History
# ----------------------------
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")