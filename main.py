#!/usr/bin/env python3
import os
import logging
from dotenv import load_dotenv

# Use absolute imports instead of relative imports
from utils.config_loader import ConfigManager
from utils.logging_setup import setup_logging
from core.llm_factory import LLMFactory
from processors.pdf_processor import process_all_pdfs
from rag.document_store import create_documents_from_results, setup_qdrant_vectorstore
from rag.qa_chain import setup_qa_chain, run_chatbot


def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()

    # Initialize configuration with default paths
    config_path = "config/config.json"
    prompts_path = "config/prompts.json"
    ConfigManager.initialize(config_path, prompts_path)

    # Setup logging
    logger = setup_logging()
    logger.info("Starting PDF processing application")

    # Get configuration
    config = ConfigManager.get_config()

    # Get PDF processing settings from config
    pdf_dir = config["pdf_processing"]["pdf_directory"]
    output_dir = config["pdf_processing"]["output_directory"]
    qa_mode = config.get("qa_mode", False)  # Default to False if not in config

    # Get LLM configuration from config
    llm_provider = config.get("llm", {}).get("provider", "anthropic")
    llm_provider = "openai"

    # Use the specified LLM provider
    if llm_provider == "openai" and "openai" in config.get("llm_alternatives", {}):
        llm_config = LLMFactory.get_alternative_config("openai")
        logger.info(f"Using OpenAI configuration: {llm_config.model_name}")
    else:
        llm_config = LLMFactory.get_default_config()
        logger.info(f"Using LLM configuration: {llm_config.provider} - {llm_config.model_name}")

    # Process PDFs
    logger.info(f"Processing PDFs from directory: {pdf_dir}")
    results = process_all_pdfs(pdf_dir, output_dir, llm_config)
    logger.info(f"PDF processing complete. Processed {len(results)} PDFs.")

    # Run QA chain if enabled in config
    if qa_mode:
        logger.info("Setting up QA system...")
        documents = create_documents_from_results(results)
        # vector_store = setup_qdrant_vectorstore(documents)
        # qa_chain = setup_qa_chain(vector_store, llm_config)
        # logger.info("QA system ready.")
        #
        # try:
        #     run_chatbot(qa_chain, vector_store, llm_config)
        # except KeyboardInterrupt:
        #     logger.info("Chatbot session terminated by user.")

    logger.info("Application execution complete.")


if __name__ == "__main__":
    main()