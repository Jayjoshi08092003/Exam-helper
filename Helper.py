import os
import sys
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# AGENTIC COMPONENTS
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool # Import the tool decorator/wrapper
from langchain_community.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper


# --- Configuration ---
# 🔑 ACTION REQUIRED: Update these with your keys!
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GOOGLE_CSE_ID"] = "" # <-- ADD THIS LINE

PDF_PATH = "ACNC_SSCP-ISC-official.pdf"
CHROMA_PATH = "chroma_db"

# Gemini Models
LLM_MODEL = "gemini-2.5-flash" 
EMBEDDING_MODEL = "text-embedding-004" 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Global variables for the Flask app
app = Flask(__name__)
RAG_AGENT_EXECUTOR = None
VECTOR_DB = None


# --- Helper Function: Check for Existing DB ---
def check_vector_store():
    """Check if the vector store already exists."""
    return os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0


# --- The Main Ingestion Pipeline ---
def ingest_documents():
    """
    Loads PDF, splits text, creates embeddings using Gemini, and stores them in ChromaDB.
    """
    global VECTOR_DB
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ Error initializing Gemini Embeddings: {e}")
        print("Please verify the GOOGLE_API_KEY is set correctly.")
        sys.exit(1)

    if check_vector_store():
        print(f"✅ Vector database already exists at '{CHROMA_PATH}'. Loading...")
        VECTOR_DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return

    # Ingestion steps (same as before)
    print(f"📄 Starting ingestion of {PDF_PATH}...")
    if not os.path.isfile(PDF_PATH):
        print(f"❌ Error: PDF file '{PDF_PATH}' not found.")
        sys.exit(1)

    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        sys.exit(1)

    print(f"✂️ Splitting document into chunks of size {CHUNK_SIZE}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(splits)}")

    print(f"🧠 Generating embeddings and storing in ChromaDB...")
    VECTOR_DB = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print(f"🎉 Ingestion complete! Data stored locally at '{CHROMA_PATH}'.")


# --- Function: Create Retrieval Chain as a Tool ---
@tool("pdf_retriever", return_direct=False)
def pdf_retriever_func(question: str) -> str:
    """
    Retrieves and answers a question ONLY from the ACNC_SSCP-ISC-official.pdf document.
    Use this for questions highly specific to the document's content.
    """
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert helper for the provided document. Use ONLY the given context to answer the question. "
                "Context: {context}",
            ),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = VECTOR_DB.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": question})
    
    # Format the response to include the source pages for the agent's benefit
    sources = set()
    for doc in response.get("context", []):
        page = doc.metadata.get('page', 'N/A')
        sources.add(f"Page {page}")
    
    source_info = f" (Sources: {', '.join(sources)})" if sources else ""
    return response['answer'] + source_info


# --- Function: Create the Agentic Executor ---
def create_agentic_executor():
    """
    Creates the main Agent Executor that chooses between the PDF tool and the Google Search tool.
    """
    print("🧠 Setting up Agentic Executor (The 'Brain')...")

    # 1. Define all available Tools
    
    # Your PDF Tool (already defined via decorator)
    pdf_tool = pdf_retriever_func
    
    # Google Search Tool (requires API key and CSE ID)
    if not os.environ.get("GOOGLE_CSE_ID") or os.environ.get("GOOGLE_CSE_ID") == "YOUR_GOOGLE_SEARCH_ENGINE_ID_HERE":
        print("⚠️ Warning: GOOGLE_CSE_ID not set correctly. Google Search tool will be unavailable.")
        google_tool = None
    else:
        search = GoogleSearchAPIWrapper()
        google_tool = GoogleSearchRun(api_wrapper=search, name="google_search")
        google_tool.description = (
            "A tool for searching the web for general knowledge, current events, or information "
            "NOT found in the ACNC_SSCP-ISC-official.pdf document. Use this when the query is "
            "external, general, or asks about dates/events."
        )

    tools = [pdf_tool]
    if google_tool:
        tools.append(google_tool)

    # 2. Initialize the Agent's LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)

    # 3. Define the Agent Prompt (crucial for decision-making)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert AI assistant. You must analyze the user's question and decide whether to use the 'pdf_retriever' tool for document-specific answers, or the 'google_search' tool for general or external information. You must use a tool to answer the question."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 4. Create the Tool-Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Create the Executor
    # Set verbose=True to see the agent's decision-making process in the console!
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor


# --- Flask Routes ---

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """API endpoint to handle questions."""
    
    global RAG_AGENT_EXECUTOR
    if RAG_AGENT_EXECUTOR is None:
        return jsonify({'error': 'RAG system not initialized. Check server logs.'}), 500
        
    data = request.get_json()
    user_input = data.get('question', '').strip()

    if not user_input:
        return jsonify({'answer': 'Please enter a question.'})

    try:
        # Invoke the Agent Executor
        response = RAG_AGENT_EXECUTOR.invoke({"input": user_input})
        
        answer = response['output']
        
        # Simple source extraction based on the tool's output format
        sources_list = []
        if "(Sources:" in answer:
            # Extract and clean up the source information from the answer string
            start = answer.find("(Sources:") + len("(Sources:")
            end = answer.rfind(")")
            source_string = answer[start:end].strip()
            sources_list = [s.strip().replace('Page ', '') for s in source_string.split(',')]
            answer = answer[:answer.find("(Sources:")].strip() # Remove sources from the answer text
        
        # Note: Google Search results will be included in the answer text by the agent.
        
        return jsonify({'answer': answer, 'sources': sources_list})

    except Exception as e:
        print(f"Error during Agent execution: {e}")
        return jsonify({'error': 'An error occurred while processing your question.'}), 500


# --- Application Initialization ---
if __name__ == "__main__":
    # Ensure the main API Key is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ CRITICAL ERROR: GOOGLE_API_KEY (for Gemini) is not set.")
        sys.exit(1)

    # 1. Ingest/Load the Vector Database
    ingest_documents()
    
    # 2. Set up the Agentic Executor globally
    RAG_AGENT_EXECUTOR = create_agentic_executor()

    # 3. Start the Flask server
    print("\n=======================================================")
    print(f"| Flask AGENTIC RAG is running. Model: {LLM_MODEL} |")
    print("| Access it at: http://127.0.0.1:5000/ |")
    print("=======================================================")
    
    app.run(debug=True, use_reloader=False)