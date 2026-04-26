import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ----------------------------
# 1. Streamlit UI config
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (LangChain + Mistral)")
st.write("Ask questions based on your documents stored in ChromaDB")

# ----------------------------
# 2. LLM
# ----------------------------
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# ----------------------------
# 3. Embeddings
# ----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ----------------------------
# 4. Vector DB
# ----------------------------
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 4,
        "lambda_mult": 0.5
    }
)

# ----------------------------
# 5. Prompt
# ----------------------------
prompt_template = ChatPromptTemplate.from_messages([
   ("system",
 """You are a helpful assistant for answering questions.
Use only the provided context.
If the answer is not present in the context, say:
"I could not find the answer in the documents provided."
"""
),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# ----------------------------
# 6. Session memory (chat history)
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# 7. Input box
# ----------------------------
query = st.text_input("Enter your question:")

if st.button("Ask") and query:

    # retrieve docs
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # prompt
    messages = prompt_template.format_messages(
        context=context,
        question=query
    )

    # response
    response = llm.invoke(messages)

    answer = response.content

    # store chat
    st.session_state.chat_history.append((query, answer))

# ----------------------------
# 8. Display chat history
# ----------------------------
st.subheader("Chat History")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    st.markdown("---")