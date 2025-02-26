import logging
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages.base import BaseMessage

from core.models import PageState, LLMConfig
from core.llm_factory import LLMFactory
from utils.config_loader import ConfigManager

logger = logging.getLogger("pdf_processor.text_extractor")


def normalize_text(text: str) -> str:
    """
    Normalize only newline and tab escape sequences in the text.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    return text.replace("\\n", "\n").replace("\\t", "\t")


def extract_text_pypdf(state: PageState) -> PageState:
    """
    Node: Extract text from the PDF using PyPDF for the specified page,
    and normalize any escape characters.

    Args:
        state: Current page state

    Returns:
        Updated page state
    """
    logger.info(f"[Page {state.page_num}] Extracting text using PyPDF.")
    try:
        reader = PdfReader(state.pdf_path)
        # PyPDF page numbers are zero-indexed
        page = reader.pages[state.page_num - 1]
        text_pypdf = page.extract_text()
        if text_pypdf:
            text_pypdf = normalize_text(text_pypdf)
        state.extracted_text_pypdf = text_pypdf.strip() if text_pypdf else ""
        logger.info(f"[Page {state.page_num}] PyPDF text extraction completed.")
    except Exception as e:
        logger.error(f"[Page {state.page_num}] Error in PyPDF text extraction: {e}", exc_info=True)
        raise
    return state


def extract_text_generic(state: PageState) -> PageState:
    """
    Node: Extract text from the provided PDF page image using LLM.
    This function instructs the LLM to extract text with special attention to checkbox data.

    Args:
        state: Current page state

    Returns:
        Updated page state
    """
    logger.info(f"[Page {state.page_num}] Starting text extraction using generic prompt with LLM.")

    # Get prompts from configuration
    prompts = ConfigManager.get_prompt("text_extraction")

    # Build reference block using the PyPDF text (if available)
    reference_block = ""
    if state.extracted_text_pypdf:
        template = prompts["reference_block_template"]
        reference_block = template.format(pypdf_text=state.extracted_text_pypdf)

    # System prompt
    system_prompt = prompts["base_system_prompt"].format(reference_block=reference_block)

    try:
        # Create LLM instance
        llm = LLMFactory.create_llm(state.llm_config)

        # Create messages - Updated for latest LangChain
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{state.image_url}",
                        "detail": "high"
                    }
                }
            ])
        ]

        # Invoke the model - handling changes in response format
        response = llm.invoke(messages)

        # Handle the response based on its type
        if isinstance(response, AIMessage):
            extracted_text = response.content
        elif isinstance(response, str):
            extracted_text = response
        else:
            # Handle other response types if needed
            extracted_text = str(response)

        state.extracted_text = extracted_text.strip()
        logger.info(f"[Page {state.page_num}] Text extraction completed successfully using LLM.")

    except Exception as e:
        logger.error(f"[Page {state.page_num}] Error in text extraction with LLM: {e}", exc_info=True)
        raise

    return state


def re_extract_text_with_keywords(state: PageState, missing_keywords: list[str]) -> PageState:
    """
    Helper: Re-extract text with a modified prompt that includes missing keywords using LLM.

    Args:
        state: Current page state
        missing_keywords: List of keywords to include in the extraction

    Returns:
        Updated page state
    """
    logger.info(f"[Page {state.page_num}] Re-extracting text with missing keywords included in prompt.")

    # Get prompts from configuration
    prompts = ConfigManager.get_prompt("text_extraction")

    # Build reference block
    template = prompts["reference_block_template"]
    reference_block = template.format(pypdf_text=state.extracted_text_pypdf)

    # Additional instructions for missing keywords
    additional_instructions = f" Ensure that the following keywords are included in the extraction: {', '.join(missing_keywords)}."

    # System prompt
    system_prompt = prompts["re_extraction_system_prompt"].format(
        additional_instructions=additional_instructions,
        reference_block=reference_block
    )

    try:
        # Create LLM instance
        llm = LLMFactory.create_llm(state.llm_config)

        # Create messages - Updated for latest LangChain
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{state.image_url}",
                        "detail": "high"
                    }
                }
            ])
        ]

        # Invoke the model - handling changes in response format
        response = llm.invoke(messages)

        # Handle the response based on its type
        if isinstance(response, AIMessage):
            extracted_text = response.content
        elif isinstance(response, str):
            extracted_text = response
        else:
            # Handle other response types if needed
            extracted_text = str(response)

        state.extracted_text = extracted_text.strip()
        logger.info(f"[Page {state.page_num}] Re-extraction completed successfully using LLM.")

    except Exception as e:
        logger.error(f"[Page {state.page_num}] Error in re-extraction with LLM: {e}", exc_info=True)
        raise

    return state