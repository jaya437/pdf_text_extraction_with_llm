import re
import json
import copy
import logging
from typing import Dict, Any, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.base import BaseMessage

from core.models import PageState
from core.llm_factory import LLMFactory
from utils.config_loader import ConfigManager
from processors.text_extractor import re_extract_text_with_keywords

logger = logging.getLogger("pdf_processor.text_comparison")


def compare_extraction(state: PageState) -> PageState:
    """
    Node: Compare the text extracted via LLM and via PyPDF using LLM reasoning.

    Args:
        state: Current page state

    Returns:
        Updated page state with the best extraction result
    """
    # Get configuration values
    text_extraction_config = ConfigManager.get_config("text_extraction")
    max_attempts = text_extraction_config.get("max_attempts", 3)
    threshold = text_extraction_config.get("confidence_threshold", 80.0)
    comparison_temperature = text_extraction_config.get("comparison_temperature", 0.7)

    # Get prompts from configuration
    comparison_prompts = ConfigManager.get_prompt("comparison")

    attempt = 0
    best_confidence = -1.0
    best_state = copy.deepcopy(state)

    while attempt < max_attempts:
        attempt += 1

        # System prompt for comparison
        system_prompt = comparison_prompts["system_prompt"]

        # User message template
        user_message_template = comparison_prompts["user_message_template"]
        user_content = user_message_template.format(
            pypdf_text=state.extracted_text_pypdf,
            llm_text=state.extracted_text
        )

        try:
            # Create a new LLM config with higher temperature for comparison
            comparison_config = copy.deepcopy(state.llm_config)
            comparison_config.temperature = comparison_temperature

            # Create LLM instance
            llm = LLMFactory.create_llm(comparison_config)

            # Create messages - Updated for latest LangChain
            messages: list[BaseMessage] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]

            # Invoke the model
            response = llm.invoke(messages)

            # Handle the response based on its type
            if isinstance(response, AIMessage):
                response_text = response.content.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                # Handle other response types if needed
                response_text = str(response).strip()

            if not response_text:
                raise ValueError("Empty response text received from LLM.")

            # Remove markdown code fences if present
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)

            result = json.loads(response_text)
            confidence = float(result.get("confidence", 0))
            missing_keywords = result.get("missing_keywords", [])
        except Exception as e:
            logger.error(f"[Page {state.page_num}] Error during LLM comparison: {e}", exc_info=True)
            # Fallback: simple regex-based comparison
            pattern = r'\b(?:\w+\s+){1,}\w+\b'
            pypdf_phrases = set(re.findall(pattern, state.extracted_text_pypdf))
            llm_phrases = set(re.findall(pattern, state.extracted_text))
            if not pypdf_phrases:
                confidence = 100.0
            else:
                matched = [phrase for phrase in pypdf_phrases if phrase in state.extracted_text]
                confidence = (len(matched) / len(pypdf_phrases)) * 100
            missing_keywords = list(pypdf_phrases - llm_phrases)

        logger.info(
            f"[Page {state.page_num}] Attempt {attempt}: LLM comparison confidence is {confidence:.2f}% and missing_keywords: {missing_keywords}")

        # Update best state if this attempt yields a higher confidence
        if confidence > best_confidence:
            best_confidence = confidence
            best_state = copy.deepcopy(state)

        if confidence >= threshold:
            logger.info(f"[Page {state.page_num}] Confidence meets threshold. Finalizing extraction.")
            break
        else:
            # Update state by re-extracting with missing keywords included in the prompt
            state = re_extract_text_with_keywords(state, missing_keywords)

    logger.info(f"[Page {state.page_num}] Final best confidence: {best_confidence:.2f}%")
    return best_state