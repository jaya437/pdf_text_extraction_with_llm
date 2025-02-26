import io
import base64
import logging
import os
from typing import List
import fitz  # PyMuPDF
from PIL import Image

from core.models import PageState, LLMConfig, ExtractedResult
from core.pipeline import execute_pipeline
from utils.file_utils import ensure_directory_exists, get_output_filepath, save_text_to_file
from processors.text_extractor import extract_text_pypdf

logger = logging.getLogger("pdf_processor.pdf_processor")


def pdf_to_image_urls(input_pdf_path: str) -> list[str]:
    """
    Opens the input PDF and converts each page into a base64-encoded PNG image.

    Args:
        input_pdf_path: Path to the PDF file

    Returns:
        List of base64-encoded PNG images
    """
    logger.info(f"Converting PDF pages to image URLs: {input_pdf_path}")
    pdf_document = fitz.open(input_pdf_path)
    image_urls = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_urls.append(base64_img)
        logger.debug(f"Converted page {page_num + 1} to image URL.")

    logger.info(f"Converted {len(image_urls)} pages into image URLs.")
    return image_urls


def process_page(pdf_name: str, pdf_path: str, page_num: int, image_bytes_url: str,
                 llm_config: LLMConfig) -> PageState:
    """
    Process one page through the LangGraph pipeline.

    Args:
        pdf_name: Name of the PDF
        pdf_path: Path to the PDF file
        page_num: Page number
        image_bytes_url: Base64-encoded PNG image of the page
        llm_config: LLM configuration

    Returns:
        Processed PageState
    """
    logger.info(f"[{pdf_name} - Page {page_num}] Starting processing.")
    initial_state = PageState(
        pdf_name=pdf_name,
        pdf_path=pdf_path,
        page_num=page_num,
        image_url=image_bytes_url,
        llm_config=llm_config
    )
    # Pre-populate the state with PyPDF extracted text as a reference.
    initial_state = extract_text_pypdf(initial_state)

    try:
        # Updated to use the new pipeline execution function
        final_state = execute_pipeline(initial_state)
        logger.info(f"[{pdf_name} - Page {page_num}] Processing completed.")
    except Exception as e:
        logger.error(f"[{pdf_name} - Page {page_num}] Processing failed: {e}", exc_info=True)
        raise
    return final_state


def process_pdf(input_pdf_path: str, output_dir: str, llm_config: LLMConfig) -> ExtractedResult:
    """
    Processes a single PDF file: converts pages to images, extracts text,
    and saves the extracted text to a file.

    Args:
        input_pdf_path: Path to the PDF file
        output_dir: Directory to store extracted text
        llm_config: LLM configuration

    Returns:
        Extraction result
    """
    pdf_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
    logger.info(f"Starting processing for PDF: {input_pdf_path}")

    image_urls = pdf_to_image_urls(input_pdf_path)
    logger.info(f"Converted PDF to {len(image_urls)} image URLs.")

    processed_states = []

    for idx, image_url in enumerate(image_urls, start=1):
        try:
            logger.info(f"Processing {pdf_name} - Page {idx}")
            state = process_page(pdf_name, input_pdf_path, idx, image_url, llm_config)
            processed_states.append(state)
        except Exception as e:
            logger.error(f"Error processing page {idx} of {pdf_name}: {e}", exc_info=True)

    # Create result
    result = ExtractedResult(
        pdf_name=pdf_name,
        pdf_path=input_pdf_path,
        pages=processed_states
    )

    # Save to file
    output_filepath = get_output_filepath(output_dir, pdf_name)
    save_text_to_file(output_filepath, result.get_all_text())

    return result


def process_all_pdfs(pdf_directory: str, output_dir: str, llm_config: LLMConfig) -> List[ExtractedResult]:
    """
    Processes all PDF files in the given directory.

    Args:
        pdf_directory: Directory containing PDF files
        output_dir: Directory to store extracted text
        llm_config: LLM configuration

    Returns:
        List of extraction results
    """
    from utils.file_utils import find_pdf_files

    all_results = []
    pdf_paths = find_pdf_files(pdf_directory)

    for pdf_path in pdf_paths:
        logger.info(f"Processing file: {pdf_path}")
        result = process_pdf(pdf_path, output_dir, llm_config)
        all_results.append(result)

    return all_results
