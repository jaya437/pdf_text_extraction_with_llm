import os
import glob
import logging
from typing import List, Optional

logger = logging.getLogger("pdf_processor.file_utils")


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure the specified directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory to check/create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.debug(f"Created directory: {directory_path}")


def find_pdf_files(directory_path: str) -> List[str]:
    """
    Find all PDF files in the specified directory.

    Args:
        directory_path: Path to search for PDF files

    Returns:
        List of full paths to PDF files
    """
    pdf_paths = glob.glob(os.path.join(directory_path, "*.pdf"))

    if not pdf_paths:
        logger.warning(f"No PDF files found in directory: {directory_path}")

    return pdf_paths


def get_output_filepath(output_dir: str, pdf_name: str, suffix: str = "_all_pages.txt") -> str:
    """
    Generate an output file path for extracted text.

    Args:
        output_dir: Directory to store output files
        pdf_name: Name of the PDF (without extension)
        suffix: Suffix to append to the PDF name

    Returns:
        Full path to the output file
    """
    ensure_directory_exists(output_dir)
    return os.path.join(output_dir, f"{pdf_name}{suffix}")


def save_text_to_file(output_filepath: str, text: str) -> None:
    """
    Save text content to a file.

    Args:
        output_filepath: Path to the output file
        text: Text content to save
    """
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Text saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving text to {output_filepath}: {e}", exc_info=True)
        raise