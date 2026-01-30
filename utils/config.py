"""
Configuration management for TwinBrain.
Provides utilities for loading and validating YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration container with validation and access methods."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._validate()
    
    def _validate(self):
        """Validate required configuration fields."""
        required_sections = ['training', 'model', 'loss']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'training.warmup_epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'training.warmup_epochs')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self):
        return f"Config({self._config})"


def load_config(config_path: Path) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def merge_configs(base_config: Config, override_config: Optional[Config] = None) -> Config:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (optional)
        
    Returns:
        Merged Config object
    """
    if override_config is None:
        return base_config
    
    merged_dict = base_config.to_dict()
    
    def deep_merge(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(merged_dict, override_config.to_dict())
    
    return Config(merged_dict)
