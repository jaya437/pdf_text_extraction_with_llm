# PDF Text Extractor

A robust solution for extracting high-quality text from PDF documents using a hybrid approach of traditional extraction methods and LLM-enhanced processing.

## Overview

This project extracts text from PDF documents using a dual-approach methodology:
1. Standard text extraction using PyPDF
2. Enhanced extraction using LLM-based image processing
3. Comparative analysis to identify and recover missing content

The system employs an iterative process to improve extraction quality, comparing results between methods and re-attempting extraction when confidence is low, ultimately saving the highest-quality output to a text file.

## Key Features

- **Hybrid Extraction**: Combines traditional PDF text extraction with LLM-based image processing
- **Iterative Refinement**: Automatically retries extraction with focused prompts when quality is below threshold
- **Provider Agnostic**: Works with any LLM provider through abstracted configurations
- **Quality Validation**: Compares extraction methods to identify and recover missing information
- **Configurable**: Easily adjust extraction settings, prompts, and confidence thresholds

## Architecture

The extraction pipeline follows this process for each page:
1. Extract initial text using PyPDF
2. Convert page to image for visual analysis
3. Construct prompt containing both initial text and image
4. Request enhanced text extraction from LLM
5. Compare PyPDF extraction with LLM extraction
6. If confidence < 90%, retry extraction with prompts focused on missing elements (max 3 attempts)
7. Select highest quality extraction as final output
8. Save consolidated text to output file

## Project Structure

```
pdf_processor/
│
├── config/
│   ├── config.json             # Main configuration file
│   └── prompts.json            # Prompt templates
│
├── core/
│   ├── __init__.py
│   ├── models.py               # Data models using Pydantic
│   ├── llm_factory.py          # LLM provider abstractions
│   └── pipeline.py             # LangGraph pipeline implementation
│
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF processing functions
│   ├── text_extractor.py       # Text extraction nodes
│   └── text_comparison.py      # Text comparison functionality
│
├── utils/
│   ├── __init__.py
│   ├── config_loader.py        # Loading and validating configs
│   ├── logging_setup.py        # Logging configuration
│   └── file_utils.py           # File handling utilities
│
├── rag/
│   ├── __init__.py
│   ├── document_store.py       # Document and vector store handling
│   └── qa_chain.py             # QA chain implementation
│
├── __init__.py
├── main.py                     # Main entry point
└── requirements.txt            # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-text-extractor.git
   cd pdf-text-extractor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your LLM provider in `config/config.json`

## Usage

### Basic Usage

```python
from pdf_processor.main import process_pdf

# Process a PDF file
process_pdf(
    input_path="path/to/your/document.pdf", 
    output_path="path/to/output.txt"
)
```

### Advanced Configuration

Modify `config/config.json` to customize:
- LLM provider settings
- Confidence thresholds
- Maximum retry attempts
- Page processing options

Example configuration:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4-vision-preview",
    "api_key_env": "OPENAI_API_KEY"
  },
  "extraction": {
    "confidence_threshold": 0.9,
    "max_retries": 3,
    "page_batch_size": 5
  },
  "output": {
    "format": "txt",
    "include_metadata": false
  }
}
```

## Customizing Prompts

Edit `config/prompts.json` to modify the prompts used for:
- Initial text extraction
- Comparison analysis
- Retry extraction with missing keywords

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.