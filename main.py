import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
import csv
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ===== Configuration =====
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.1:8b"
DATA_DIR = "MOCK_DATA.csv"
DB_PATH = "./chroma_db_mxbai/"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

# ===== Setup Phase =====
def setup_models():
    """Configure embedding and LLM models"""
    print("Configuring models...")
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    print("✓ Models configured successfully")

def load_documents():
      """Load documents from CSV"""
      print(f"Loading documents from {DATA_DIR}...")
      documents = []
      with open(DATA_DIR, newline="", encoding="utf-8") as f:
          reader = csv.DictReader(f)
          for row in reader:
              text = (
                  f"Client: {row['first_name']} {row['last_name']}. "
                  f"Company: {row['company']}. "
                  f"Notes: {row['contact_notes']}"
              )
              documents.append(Document(text=text))
      print(f"✓ Loaded {len(documents)} documents")
      return documents

def setup_vector_db(documents):
    """Create and populate vector database"""
    print("Setting up vector database...")
    
    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection("documents")
    
    # Create storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Build index
    print("Building vector index (this may take a moment)...")
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)]
    )
    
    print("✓ Vector database setup complete")
    return vector_index

def create_query_engine(vector_index):
    """Create configured query engine"""
    query_engine = vector_index.as_query_engine(
        response_mode="refine",
        similarity_top_k=20
    )
    
    # Customize prompts (optional)
    qa_template = PromptTemplate(
        "You are an expert assistant answering questions based on provided context.\n"
        "Context:\n{context_str}\n"
        "Question: {query_str}\n"
        "Provide accurate, helpful answers based only on the provided context.\n"
        "Answer: "
    )
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": qa_template
    })
    
    return query_engine

# ===== Query Function =====
def ask(query_engine, question: str):
    """Query the RAG pipeline"""
    response = query_engine.query(question)
    return response

# ===== Main =====
if __name__ == "__main__":
    # Initialize pipeline
    setup_models()
    documents = load_documents()
    vector_index = setup_vector_db(documents)
    query_engine = create_query_engine(vector_index)
    
    print("\n" + "="*50)
    print("RAG Pipeline Ready!")
    print("="*50 + "\n")
    
    # Example queries
    questions = [
        "What client talked to me about liquidity needs?"
    ]
    
    for q in questions:
        print(f"Q: {q}")
        response = ask(query_engine, q)
        print(f"A: {response}\n")