import logging
import copy
import os
from typing import Dict, Any, Optional

# Updated imports for latest LangChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from core.models import LLMConfig
from utils.config_loader import ConfigManager

logger = logging.getLogger("pdf_processor.llm_factory")


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create_llm(config: LLMConfig) -> BaseChatModel:
        """
        Create an LLM instance based on the configuration

        Args:
            config: LLM configuration

        Returns:
            A configured LLM instance

        Raises:
            ValueError: If the provider is not supported
        """
        logger.debug(f"Creating LLM instance for provider: {config.provider}, model: {config.model_name}")

        try:
            if config.provider == "openai":
                return ChatOpenAI(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    openai_api_key=config.get_api_key()
                )
            elif config.provider == "anthropic":
                return ChatAnthropic(
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    anthropic_api_key=config.get_api_key()
                )
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
        except Exception as e:
            logger.error(f"Error creating LLM instance: {e}", exc_info=True)
            raise

    @staticmethod
    def get_default_config() -> LLMConfig:
        """
        Get the default LLM configuration from the config file

        Returns:
            LLM configuration
        """
        llm_config = ConfigManager.get_config("llm")
        return LLMConfig(**llm_config)

    @staticmethod
    def get_alternative_config(name: str) -> LLMConfig:
        """
        Get an alternative LLM configuration from the config file

        Args:
            name: Name of the alternative configuration

        Returns:
            LLM configuration
        """
        alternatives = ConfigManager.get_config("llm_alternatives")

        if name not in alternatives:
            raise ValueError(f"Alternative LLM configuration not found: {name}")

        return LLMConfig(**alternatives[name])

    @staticmethod
    def get_config_with_temperature(config: LLMConfig, temperature: float) -> LLMConfig:
        """
        Create a copy of the config with a different temperature

        Args:
            config: Original LLM configuration
            temperature: New temperature value

        Returns:
            Modified LLM configuration
        """
        new_config = copy.deepcopy(config)
        new_config.temperature = temperature
        return new_config

    @staticmethod
    def create_multi_provider_llm(config: Optional[LLMConfig] = None) -> BaseChatModel:
        """
        Create an LLM instance, with fallback to alternative providers if the primary one fails.

        Args:
            config: Optional LLM configuration. If None, the default config will be used.

        Returns:
            A configured LLM instance
        """
        if config is None:
            config = LLMFactory.get_default_config()

        try:
            return LLMFactory.create_llm(config)
        except Exception as primary_error:
            logger.warning(f"Failed to create primary LLM ({config.provider}): {primary_error}")

            # Try to use alternative configurations
            alternatives = ConfigManager.get_config("llm_alternatives")
            for alt_name, alt_config_dict in alternatives.items():
                try:
                    logger.info(f"Trying alternative LLM: {alt_name}")
                    alt_config = LLMConfig(**alt_config_dict)
                    return LLMFactory.create_llm(alt_config)
                except Exception as alt_error:
                    logger.warning(f"Failed to create alternative LLM ({alt_name}): {alt_error}")

            # If we get here, all attempts failed
            logger.error("All attempts to create LLM instances failed")
            raise RuntimeError("Could not create any LLM instance") from primary_error