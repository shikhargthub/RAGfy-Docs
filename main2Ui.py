import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from mistralai import Mistral


# -----------------------------
# HUGGINGFACE EMBEDDINGS
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# MISTRAL CLIENT
# -----------------------------
MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY"

client = Mistral(api_key=MISTRAL_API_KEY)
MODEL = "mistral-large-latest"


# -----------------------------
# PDF PROCESSING
# -----------------------------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore


# -----------------------------
# MISTRAL QA FUNCTION
# -----------------------------
def get_mistral_answer(context, question):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer ONLY using the provided context."
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}

If the answer is not in the context, say you don't know.
"""
        }
    ]

    response = client.chat.complete(
        model=MODEL,
        messages=messages
    )

    return response.choices[0].message.content


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📄 RAG App (HF Embeddings + Mistral AI)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
        st.success("PDF processed successfully!")


# -----------------------------
# QUERY SECTION
# -----------------------------
query = st.text_input("Ask a question from the PDF")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first")
    else:
        retriever = st.session_state.vectorstore.as_retriever()

        # 🔍 Retrieve relevant chunks
        docs = retriever.invoke(query)

        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])

        # 🤖 Get answer from Mistral
        answer = get_mistral_answer(context, query)

        st.subheader("🧠 Answer")
        st.write(answer)

        # Optional debug view
        with st.expander("🔍 Retrieved Chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)