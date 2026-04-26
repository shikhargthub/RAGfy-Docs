from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 1. LLM (keep separate)
llm = ChatMistralAI(model="mistral-large-latest")


# 2. Embeddings (correct)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 4,
        "lamda_mult":0.5
                   }
)


#prompt template

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant for answering questions.
Use only the provided context to answer the question.
If the answer is not present in the context, say: I could not find the answer in the documents provided."""
        ),
        (
            "human",
            """{context}

Question: {question}"""
        )
    ]
)
    
print("RAG system activated.Ask your questions now!")
print("press 0 to exit the system")

while True:
    query=input("Enter your question: ")
    if query=="0":
        print("Exiting the system. Goodbye!")
        break
    
    #retrieving relevant documents
    relevant_docs = retriever.invoke(query)
    
    #formatting the prompt
    formatted_prompt=prompt_template.format_prompt(
        context="\n".join([doc.page_content for doc in relevant_docs]),
        question=query
    )
    
    #getting the answer from the LLM
    answer=llm.invoke(formatted_prompt.to_messages())
    
    print("Bot:",answer.content)




