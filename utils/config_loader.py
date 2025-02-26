import json
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages loading and accessing configuration settings."""

    _config: Dict[str, Any] = {}
    _prompts: Dict[str, Any] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, config_path: str = "config/config.json", prompts_path: str = "config/prompts.json") -> None:
        """
        Initialize the config manager by loading configuration and prompt files.

        Args:
            config_path: Path to the config JSON file
            prompts_path: Path to the prompts JSON file
        """
        if cls._initialized:
            return

        # Load the main configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as config_file:
            cls._config = json.load(config_file)

        # Load prompts
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(prompts_path, 'r') as prompts_file:
            cls._prompts = json.load(prompts_file)

        cls._initialized = True

    @classmethod
    def get_config(cls, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration settings.

        Args:
            section: Optional section name to retrieve specific settings

        Returns:
            Dictionary containing the requested configuration
        """
        if not cls._initialized:
            cls.initialize()

        if section is None:
            return cls._config

        if section not in cls._config:
            raise KeyError(f"Configuration section not found: {section}")

        return cls._config[section]

    @classmethod
    def get_prompt(cls, section: str, prompt_name: Optional[str] = None) -> str:
        """
        Get a prompt template.

        Args:
            section: Section name in the prompts file
            prompt_name: Optional prompt name within the section

        Returns:
            The requested prompt template
        """
        if not cls._initialized:
            cls.initialize()

        if section not in cls._prompts:
            raise KeyError(f"Prompt section not found: {section}")

        section_data = cls._prompts[section]

        if prompt_name is None:
            # If no specific prompt is requested, return the whole section
            return section_data

        if prompt_name not in section_data:
            raise KeyError(f"Prompt not found: {prompt_name} in section {section}")

        return section_data[prompt_name]