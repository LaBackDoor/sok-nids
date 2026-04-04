"""Shared YAML configuration loader for all experiments.

Usage:
    from config_loader import load_yaml_config
    raw = load_yaml_config()                     # default path
    raw = load_yaml_config("path/to/config.yaml")
"""

import hashlib
import json
from pathlib import Path

import yaml


_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_yaml_config(path: str | Path | None = None) -> dict:
    """Load and return the raw YAML config as a dict.

    Args:
        path: Path to YAML file. Defaults to experiments/config.yaml.

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def config_section_hash(section: dict) -> str:
    """Compute a deterministic SHA-256 hash of a config section.

    Used by the checkpoint system to detect config changes.
    """
    canonical = json.dumps(section, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
