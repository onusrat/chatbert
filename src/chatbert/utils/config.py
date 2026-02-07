"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        OmegaConf DictConfig object with configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return OmegaConf.create(config_dict)


def merge_configs(base_config: DictConfig, override_config: Optional[Dict[str, Any]] = None) -> DictConfig:
    """Merge base config with override values.

    Args:
        base_config: Base configuration.
        override_config: Dictionary of override values.

    Returns:
        Merged configuration.
    """
    if override_config is None:
        return base_config

    override = OmegaConf.create(override_config)
    return OmegaConf.merge(base_config, override)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        save_path: Path to save YAML file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


class ChatBERTConfig:
    """Configuration class for ChatBERT models."""

    def __init__(self, config: Union[DictConfig, Dict[str, Any], str, Path]):
        """Initialize configuration.

        Args:
            config: Either a DictConfig, dict, or path to config file.
        """
        if isinstance(config, (str, Path)):
            self._config = load_config(config)
        elif isinstance(config, dict):
            self._config = OmegaConf.create(config)
        else:
            self._config = config

    @property
    def model(self) -> DictConfig:
        """Model configuration."""
        return self._config.model

    @property
    def training(self) -> DictConfig:
        """Training configuration."""
        return self._config.training

    @property
    def inference(self) -> DictConfig:
        """Inference configuration."""
        return self._config.inference

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        save_config(self._config, path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, Path]) -> "ChatBERTConfig":
        """Load configuration from pretrained model directory.

        Args:
            model_name_or_path: Model name or path to model directory.

        Returns:
            ChatBERTConfig instance.
        """
        path = Path(model_name_or_path)
        config_path = path / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        return cls(config_path)
