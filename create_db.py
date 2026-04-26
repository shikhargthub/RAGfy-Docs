#load pdf
# split into chunks
# create the embeddings
# store in chroma db


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

# 
loader=PyPDFLoader("document loader/DIP_Filtering_Hinglish.pdf")
docs=loader.load()

#split into chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks=splitter.split_documents(docs)

# creating embeddings

embeding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# storing
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeding_model,
    persist_directory="chroma_db"
)



