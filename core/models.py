import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Dict, Any, List, Union, ClassVar


class LLMConfig(BaseModel):
    """Configuration for the LLM provider and model"""
    model_config = ConfigDict(extra="ignore")  # Updated for Pydantic v2

    provider: Literal["openai", "anthropic"]
    model_name: str
    temperature: float
    max_tokens: int
    api_key_env: str
    api_key: Optional[str] = None

    def get_api_key(self) -> str:
        """Get API key from either the config or environment variables"""
        import os
        from dotenv import load_dotenv

        # If an API key is directly provided in the config, use it
        if self.api_key:
            return self.api_key

        # Load environment variables from .env file
        # This should be done at the start of your application,
        # but we'll do it here as a fallback
        load_dotenv()

        # Get the key from environment variables
        key = os.environ.get(self.api_key_env)

        # Debug output to see what's happening
        print(f"Looking for API key in environment variable: {self.api_key_env}")

        if not key:
            raise ValueError(f"API key for {self.provider} not found in environment variable {self.api_key_env}")

        # Print a masked version of the key for debugging
        masked_key = key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "********"
        print(f"Found API key: {masked_key}")

        return key


class PageState(BaseModel):
    """State for processing a PDF page"""
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Updated for Pydantic v2

    pdf_name: str
    pdf_path: str = ""  # Full PDF path
    page_num: int
    image_url: str  # Base64-encoded PNG image of the page
    extracted_text: str = ""  # Text extracted via LLM
    extracted_text_pypdf: str = ""  # Text extracted via PyPDF
    llm_config: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.1,
        max_tokens=4000,
        api_key_env="ANTHROPIC_API_KEY"
    ))

    def dict_state(self) -> Dict[str, Any]:
        """
        Convert to a dictionary state for LangGraph 0.2.x

        Returns:
            Dictionary representation of the state
        """
        return {"state": self}


class ExtractedResult(BaseModel):
    """Result of the extraction process for a single PDF"""
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Updated for Pydantic v2

    pdf_name: str
    pdf_path: str
    pages: List[PageState]

    def get_all_text(self) -> str:
        """Get all extracted text from all pages"""
        result = ""
        for page in sorted(self.pages, key=lambda p: p.page_num):
            result += f"Page {page.page_num}:\n"
            result += page.extracted_text + "\n"
            result += "=" * 50 + "\n\n"
        return result

    def get_page_count(self) -> int:
        """Get the number of pages processed"""
        return len(self.pages)

    def get_successful_pages(self) -> List[PageState]:
        """Get list of successfully processed pages (with extracted text)"""
        return [page for page in self.pages if page.extracted_text.strip()]

    def get_success_rate(self) -> float:
        """Get the success rate as a percentage"""
        if not self.pages:
            return 0.0

        successful = len(self.get_successful_pages())
        return (successful / len(self.pages)) * 100