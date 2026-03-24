📚 Exam-Helper: RAG-Powered Study Assistant
Exam-Helper is an intelligent Retrieval-Augmented Generation (RAG) system designed to help students navigate dense academic materials. By indexing textbooks, notes, and past papers into a vector database, it provides precise, context-aware answers to exam-related queries.

🚀 Overview
Traditional LLMs can hallucinate or lack specific details from your unique syllabus. Exam-Helper solves this by:

Ingesting your study PDFs or text files.

Vectorizing the content using high-performance embeddings.

Storing data in ChromaDB for lightning-fast semantic search.

Retrieving only the relevant "shards" of information to provide grounded, factual answers.

🛠️ Tech Stack
LLM Framework: LangChain / LlamaIndex

Vector Database: ChromaDB

Embeddings: OpenAI text-embedding-3-small / HuggingFace Transformers

Language: Python 3.9+

Interface: Streamlit / Flask (optional)

📂 System Architecture
The project follows a standard RAG pipeline:

Data Ingestion: Documents are loaded and split into manageable chunks using a RecursiveCharacterTextSplitter.

Vectorization: Each chunk is converted into a high-dimensional vector.

Storage: Vectors are stored in a ChromaDB collection with associated metadata.

Retrieval & Generation: When a user asks a question, the system performs a similarity search in ChromaDB and passes the top-k results to the LLM as context.

🔧 Installation & Setup
Clone the repository

Bash
git clone https://github.com/your-username/exam-helper.git
cd exam-helper
Install dependencies

Bash
pip install -r requirements.txt
Set up environment variables
Create a .env file and add your API keys:

Code snippet
OPENAI_API_KEY=your_api_key_here
Run the application

Bash
python main.py
📖 Usage
Upload: Place your exam materials in the /docs folder.

Index: Run the indexing script to populate the ChromaDB vector store.

Query: Ask questions like "Explain the process of Mitosis according to Chapter 4" or "Summarize the key themes in the 2023 Physics paper."

📈 Future Roadmap
[ ] Support for OCR (to read handwritten study notes).

[ ] Integration with specialized medical/legal LLMs.

[ ] Multi-user session history for personalized revision.
