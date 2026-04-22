import csv
import chromadb
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate, Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

PROMPT_TEMPLATES = {
    "direct": (
        "You are an expert assistant answering questions based on provided context.\n"
        "Context:\n{context_str}\n"
        "Question: {query_str}\n"
        "Provide accurate, helpful answers based only on the provided context.\n"
        "Answer: "
    ),
    "evidence": (
        "You are an expert assistant answering questions about client notes.\n"
        "Context:\n{context_str}\n"
        "Question: {query_str}\n"
        "List each matching client by name and cite the relevant excerpt from their notes.\n"
        "Answer: "
    ),
    "fallback": (
        "You are an expert assistant answering questions about client notes.\n"
        "Context:\n{context_str}\n"
        "Question: {query_str}\n"
        "If the context does not contain relevant information, say exactly: "
        "'I cannot find evidence in the client notes to answer this question.' "
        "Otherwise, provide an accurate answer based only on the context.\n"
        "Answer: "
    ),
}


@dataclass
class PipelineConfig:
    embedding_model: str = "mxbai-embed-large"
    llm_model: str = "llama3.1:8b"
    data_path: str = "MOCK_DATA.csv"
    chunk_size: int = 256
    chunk_overlap: int = 32
    similarity_top_k: int = 20
    splitter: str = "sentence"      # "sentence" or "simple"
    prompt_style: str = "direct"    # "direct", "evidence", "fallback"

    @property
    def db_path(self):
        safe = self.embedding_model.replace("-", "_").replace(":", "_")
        return f"./chroma_db_{safe}_{self.splitter}_{self.chunk_size}/"

    @property
    def collection_name(self):
        return f"docs_{self.chunk_size}_{self.chunk_overlap}"

    def label(self):
        return (
            f"{self.embedding_model}|{self.llm_model}|"
            f"{self.splitter}|chunk{self.chunk_size}|"
            f"k{self.similarity_top_k}|{self.prompt_style}"
        )


def _setup_models(config: PipelineConfig):
    Settings.embed_model = OllamaEmbedding(model_name=config.embedding_model)
    Settings.llm = Ollama(model=config.llm_model, request_timeout=120.0)


def _load_documents(config: PipelineConfig):
    documents = []
    with open(config.data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"Client: {row['first_name']} {row['last_name']}. "
                f"Company: {row['company']}. "
                f"Notes: {row['contact_notes']}"
            )
            documents.append(Document(
                text=text,
                metadata={
                    "client_id": row["id"],
                    "client_name": f"{row['first_name']} {row['last_name']}",
                    "company": row["company"],
                },
            ))
    return documents


def _get_splitter(config: PipelineConfig):
    if config.splitter == "sentence":
        return SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    return TokenTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def _build_index(config: PipelineConfig, documents, force_rebuild: bool):
    db = chromadb.PersistentClient(path=config.db_path)
    if force_rebuild:
        try:
            db.delete_collection(config.collection_name)
        except Exception:
            pass
    collection = db.get_or_create_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    existing = collection.count()
    if existing > 0 and not force_rebuild:
        print(f"  Reusing existing index ({existing} chunks) at {config.db_path}")
        return VectorStoreIndex.from_vector_store(vector_store)

    print(f"  Building index from {len(documents)} documents...")
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[_get_splitter(config)],
        show_progress=True,
    )


def build_pipeline(config: PipelineConfig, force_rebuild: bool = False):
    """
    Returns (query_engine, retriever) for the given config.
    query_engine: full RAG pipeline (retrieval + LLM generation)
    retriever: retrieval only (no LLM), for scoring Recall@k / Precision@k
    """
    print(f"Building pipeline: {config.label()}")
    _setup_models(config)
    documents = _load_documents(config)
    index = _build_index(config, documents, force_rebuild)

    retriever = index.as_retriever(similarity_top_k=config.similarity_top_k)

    query_engine = index.as_query_engine(
        response_mode="refine",
        similarity_top_k=config.similarity_top_k,
    )
    qa_template = PromptTemplate(PROMPT_TEMPLATES[config.prompt_style])
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})

    return query_engine, retriever
