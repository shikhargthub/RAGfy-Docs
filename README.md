# RAGfy-Docs
📚 RAG (Retrieval-Augmented Generation) Project

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, HuggingFace embeddings, and vector search to enhance LLM responses with external knowledge.

🚀 Features
Document ingestion and preprocessing
Text embeddings using HuggingFace Sentence Transformers
Vector storage for efficient similarity search
Retrieval-based context augmentation
Integration with LangChain pipeline
Modular and extensible architecture
🧠 Tech Stack
Python 3.10+
LangChain
HuggingFace Transformers
Sentence Transformers
PyTorch
FAISS / Vector Store (optional depending on implementation)
📁 Project Structure
RAG/
│── main.py / main2Ui.py     # Entry point
│── data/                    # Source documents
│── embeddings/             # Embedding logic
│── vectorstore/            # Stored vectors (FAISS or similar)
│── utils/                  # Helper functions
│── requirements.txt
│── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/rag-project.git
cd rag-project
2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
3. Install dependencies
pip install -r requirements.txt
📦 Key Dependencies

If you don’t have requirements.txt, install manually:

pip install langchain langchain-huggingface
pip install sentence-transformers
pip install transformers
pip install torch
pip install faiss-cpu
🧪 How It Works
Load documents from /data
Split text into chunks
Convert chunks into embeddings using HuggingFace models
Store embeddings in a vector database (FAISS)
On query:
Retrieve top-k similar chunks
Pass them as context to LLM
Generate final response
▶️ Run the Project
python main.py

or UI version:

python main2Ui.py
🧩 Example Use Case

Ask a question like:

“What are the key concepts of machine learning?”

The system:

Retrieves relevant document sections
Injects them into LLM context
Produces an informed answer grounded in your data
⚠️ Common Issues
1. ModuleNotFoundError: langchain_huggingface

Fix:

pip install langchain-huggingface
2. Slow startup / torch import delay

First run may take time due to:

PyTorch initialization
Transformers loading compiled modules
📌 Future Improvements
Add OpenAI / local LLM support
Web UI (Streamlit / React)
Persistent vector DB (Pinecone / ChromaDB)
Streaming responses
Document upload interface
👨‍💻 Author

Shikhar Gupta

📜 License

This project is open-source and available under the MIT License.
