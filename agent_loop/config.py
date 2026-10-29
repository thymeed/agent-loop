"""Settings from config.yaml, falling back to built-in defaults."""
from pathlib import Path

import yaml

DEFAULTS = {
    "models": {
        "lead": "ollama:gemma3:4b",
        "worker": "ollama:gemma3:4b",
        "reviewer": "ollama:gemma3:4b",
    },
    "max_rejections": 3,
    "test_timeout": 15,
    "sandbox": "docker",
    "sandbox_image": "python:3.12-slim",
    "sandbox_memory": "256m",
    "sandbox_cpus": 1,
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = {**DEFAULTS, "models": dict(DEFAULTS["models"])}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg["models"].update(loaded.pop("models", {}))
        cfg.update(loaded)
    return cfg


def make_llm(spec: str):
    """Build a chat model from 'provider:model'."""
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model)
    raise ValueError(f"unknown provider {provider!r} in {spec!r}")


CONFIG = load_config()
