import logging
from typing import Dict, Any, List, Tuple, Optional

# Updated imports for latest LangChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_qdrant import QdrantVectorStore

from core.models import LLMConfig
from core.llm_factory import LLMFactory
from utils.config_loader import ConfigManager

logger = logging.getLogger("pdf_processor.rag.qa_chain")


def setup_qa_chain(vector_store: QdrantVectorStore, llm_config: LLMConfig):
    """
    Sets up the question-answering chain using the latest LangChain patterns.

    Args:
        vector_store: Configured vector store
        llm_config: LLM configuration

    Returns:
        Configured QA chain
    """
    # Get vector store configuration
    vector_config = ConfigManager.get_config("vector_store")
    search_type = vector_config.get("search_type", "mmr")
    search_k = vector_config.get("search_k", 4)

    # Create retriever with the specified search type
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": search_k}
    )

    # Create LLM
    llm = LLMFactory.create_llm(llm_config)

    # Create the prompt template
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Question: {question}

Context:
{context}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Define the document formatting function
    def format_docs(docs):
        return "\n\n".join(f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs))

    # Build the RAG chain using the latest LCEL style
    rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def run_chatbot(qa_chain, vector_store: QdrantVectorStore, llm_config: LLMConfig) -> None:
    """
    Runs an interactive chatbot session using the QA chain.

    Args:
        qa_chain: The configured QA chain
        vector_store: The vector store (for accessing document metadata)
        llm_config: LLM configuration
    """
    chat_history = []
    print("Chatbot is running. Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your query about the PDF content: ").strip()
        if user_query.lower() == "exit":
            print("Exiting chat session.")
            break

        # Execute the chain
        try:
            answer = qa_chain.invoke(user_query)

            # Get the source documents (optional)
            # We need to run the retriever separately to get the source documents
            retriever = vector_store.as_retriever()
            source_docs = retriever.get_relevant_documents(user_query)

            print("\n=== Retrieved Answer ===")
            print(answer)

            if source_docs:
                print("\n=== Source Documents ===")
                for doc in source_docs:
                    print(f"Page {doc.metadata.get('page')} from {doc.metadata.get('pdf_name')}")

            # Update chat history
            chat_history.append((user_query, answer))
            print("\n--- Chat History Updated ---\n")

        except Exception as e:
            logger.error(f"Error executing QA chain: {e}", exc_info=True)
            print(f"An error occurred: {str(e)}")