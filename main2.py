# dynamic pdf uploding and question answering system using RAG
# 1. Load PDF
# 2. Split into chunks
# 3. Create embeddings
# 4. Store in Chroma DB
# 5. Create retriever
# 6. Create prompt template
# 7. Chat loop for Q&A

from dotenv import load_dotenv
import os
import tempfile

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------
# 1. LLM
# ----------------------------
llm = ChatMistralAI(model="mistral-large-latest")

# ----------------------------
# 2. Embeddings
# ----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ----------------------------
# 3. PDF Processing Function
# ----------------------------
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    return vectorstore

# ----------------------------
# 4. Get PDF from user
# ----------------------------
pdf_path = input("Enter PDF file path: ")

if not os.path.exists(pdf_path):
    print("Invalid file path. Exiting...")
    exit()

print("Processing PDF...")

vectorstore = process_pdf(pdf_path)

print("PDF processed successfully!")

# ----------------------------
# 5. Retriever
# ----------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 4,
        "lambda_mult": 0.5
    }
)

# ----------------------------
# 6. Prompt
# ----------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant for answering questions.
Use only the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the documents provided."
"""),
    ("human", "{context}\n\nQuestion: {question}")
])

# ----------------------------
# 7. Chat Loop
# ----------------------------
print("\nRAG system activated. Ask your questions now!")
print("Press 0 to exit\n")

while True:
    query = input("You: ")

    if query == "0":
        print("Exiting system. Goodbye!")
        break

    # Retrieve docs
    relevant_docs = retriever.invoke(query)

    # Build context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Create prompt
    messages = prompt_template.format_messages(
        context=context,
        question=query
    )

    # Get response
    response = llm.invoke(messages)

    print("\nAnswer:", response.content)
    print("-" * 50)