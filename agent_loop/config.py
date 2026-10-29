"""Settings from config.yaml, falling back to built-in defaults."""
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "models": {
        "lead": "ollama:gemma3:4b",
        "worker": "ollama:qwen2.5-coder:7b",
        "reviewer": "ollama:qwen2.5-coder:7b",
    },
    "max_rejections": 3,
    "test_timeout": 15,
    "sandbox": "docker",
    "sandbox_image": "python:3.12-slim",
    "sandbox_memory": "256m",
    "sandbox_cpus": 1,
    "num_ctx": 8192,
    "max_project_bytes": 500_000,
    "max_context_chars": 12_000,
    "project_memory": "512m",
    "project_test_timeout": 120,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def config_path(env=None) -> Path:
    """config.yaml, or the file named by AGENT_LOOP_CONFIG (so evals can compare setups).

    A named file that does not exist is an error: silently falling back to the default
    models would make an eval measure the wrong thing."""
    override = (os.environ if env is None else env).get("AGENT_LOOP_CONFIG")
    if not override:
        return DEFAULT_CONFIG_PATH
    path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AGENT_LOOP_CONFIG points at a missing file: {override}")
    return path


CONFIG_PATH = config_path()


def load_config(path: Path = CONFIG_PATH, env=None) -> dict:
    cfg = {**DEFAULTS, "models": dict(DEFAULTS["models"])}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg["models"].update(loaded.pop("models", {}))
        cfg.update(loaded)
    if cfg["sandbox"] not in ("docker", "local"):
        raise ValueError(f"sandbox must be 'docker' or 'local', got {cfg['sandbox']!r}")
    if cfg["sandbox"] == "local" and (os.environ if env is None else env).get("AGENT_LOOP_ALLOW_LOCAL") != "1":
        # One config line must not be enough to run model-written code on this machine.
        raise ValueError(
            "sandbox: local runs model-written code with NO isolation (your files, your network, "
            "your account). Use sandbox: docker. If you really mean it, also set the environment "
            "variable AGENT_LOOP_ALLOW_LOCAL=1.")
    return cfg


def make_llm(spec: str):
    """Build a chat model from 'provider:model'."""
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama's default context is small and silently truncates long prompts.
        return ChatOllama(model=model, num_ctx=CONFIG["num_ctx"])
    raise ValueError(f"unknown provider {provider!r} in {spec!r}")


CONFIG = load_config()
