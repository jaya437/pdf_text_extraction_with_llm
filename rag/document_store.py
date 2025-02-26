import logging
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from core.models import PageState, ExtractedResult
from utils.config_loader import ConfigManager

logger = logging.getLogger("pdf_processor.rag.document_store")


def create_documents_from_states(processed_states: List[PageState]) -> List[Document]:
    """
    Converts processed PageState objects into LangChain Document objects.

    Args:
        processed_states: List of processed page states

    Returns:
        List of LangChain Document objects
    """
    documents = []
    for state in processed_states:
        if state.extracted_text.strip():
            doc = Document(
                page_content=state.extracted_text,
                metadata={"pdf_name": state.pdf_name, "page": state.page_num}
            )
            documents.append(doc)
        else:
            logger.warning(
                f"No extracted text for {state.pdf_name} - Page {state.page_num}, skipping document creation.")
    return documents


def create_documents_from_results(results: List[ExtractedResult]) -> List[Document]:
    """
    Converts extraction results into LangChain Document objects.

    Args:
        results: List of extraction results

    Returns:
        List of LangChain Document objects
    """
    all_states = []
    for result in results:
        all_states.extend(result.pages)
    return create_documents_from_states(all_states)


def setup_qdrant_vectorstore(documents: List[Document]) -> QdrantVectorStore:
    """
    Sets up an in-memory Qdrant collection and indexes the provided documents.

    Args:
        documents: List of documents to index

    Returns:
        Configured QdrantVectorStore
    """
    logger.info("Setting up Qdrant vector store...")

    # Get vector store configuration
    vector_config = ConfigManager.get_config("vector_store")
    embedding_model = vector_config.get("embedding_model", "text-embedding-ada-002")
    collection_name = vector_config.get("collection_name", "pdf_documents")
    vector_size = vector_config.get("vector_size", 1536)

    # Create embeddings model
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Set up Qdrant client
    qdrant_client = QdrantClient(":memory:")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Create vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # Add documents to vector store
    vector_store.add_documents(documents)
    logger.info(f"Indexed {len(documents)} documents in Qdrant.")

    return vector_store