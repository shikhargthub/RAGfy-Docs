import streamlit as st
import tempfile
from typing import List
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="🤖")
st.title("🚀 RAGfy Docs")
st.write("AI-Powered Document Intelligence Platform")

# ----------------------------
# LLM (same as your working one)
# ----------------------------
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# ----------------------------
# Embeddings (same as before)
# ----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ----------------------------
# Text Splitter
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

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
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# PDF Upload + Processing
# ----------------------------
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Process PDFs"):
        all_docs = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            pages = loader.load()

            for i, page in enumerate(pages):
                page.metadata["source"] = file.name
                page.metadata["page"] = i + 1

            all_docs.extend(pages)

        chunks = splitter.split_documents(all_docs)

        # Vector DB (same as original)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        # BM25
        bm25 = BM25Retriever.from_documents(chunks)

        st.session_state.vectorstore = vectorstore
        st.session_state.bm25 = bm25

        st.success("Documents processed!")

# ----------------------------
# Query Rewrite
# ----------------------------
def rewrite_query(query):
    response = llm.invoke([
        ("system", "Rewrite query for better retrieval"),
        ("human", query)
    ])
    return response.content.strip()

# ----------------------------
# Reranking
# ----------------------------
def rerank_docs(query: str, docs: List):
    query_terms = set(query.lower().split())
    scored = []

    for doc in docs:
        terms = set(doc.page_content.lower().split())
        score = len(query_terms.intersection(terms))
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:3]]

# ----------------------------
# Query Input
# ----------------------------
query = st.text_input("Ask a question")

if st.button("Ask") and query:

    if st.session_state.vectorstore is None:
        st.warning("Upload PDFs first")
    else:
        # Step 1: Rewrite
        rewritten_query = rewrite_query(query)

        # Step 2: Retrieve (Hybrid manually)
        semantic_docs = st.session_state.vectorstore.as_retriever().invoke(rewritten_query)
        keyword_docs = st.session_state.bm25.invoke(rewritten_query)

        # Merge results
        docs = semantic_docs + keyword_docs

        # Step 3: Rerank
        top_docs = rerank_docs(query, docs)

        # Step 4: Context
        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Step 5: Answer
        messages = prompt_template.format_messages(
            context=context,
            question=query
        )

        response = llm.invoke(messages)
        answer = response.content

        # Save chat
        st.session_state.chat_history.append((query, answer))

        # Display
        st.subheader("🧠 Answer")
        st.write(answer)





# ----------------------------
# Chat History
# ----------------------------
st.subheader("Chat History")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    st.markdown("---")