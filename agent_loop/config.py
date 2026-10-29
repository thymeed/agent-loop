"""Settings: built-in defaults, then config.yaml, then an optional experiment patch.

AGENT_LOOP_CONFIG names a small YAML *patch* that is layered over config.yaml, so an experiment
holds only what differs and cannot drift from the real config.
"""
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "models": {
        "lead": "ollama:gemma3:4b",
        "worker": "ollama:qwen2.5-coder:7b",
        "reviewer": "ollama:qwen2.5-coder:7b",
    },
    "model_options": {},
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

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_CONFIG_PATH = CONFIG_PATH

# The only per-role model options, each with its check. Anything else is an error: a typo in an
# experiment must not silently change what is being measured, and an option a model rejects
# (reasoning on a model that cannot think) must be opt-in per role.
MODEL_OPTION_RULES = {
    "reasoning": (lambda v: v is None or isinstance(v, bool), "true, false or null (the model's own default)"),
    "num_predict": (lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 32768,
                    "a whole number from 1 to 32768"),
    "temperature": (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 2,
                    "a number from 0 to 2"),
}


def override_path(env=None) -> Path | None:
    """The experiment patch named by AGENT_LOOP_CONFIG, or None.

    A named file that does not exist is an error: silently ignoring it would make an
    experiment measure the control."""
    named = (os.environ if env is None else env).get("AGENT_LOOP_CONFIG")
    if not named:
        return None
    path = Path(named).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AGENT_LOOP_CONFIG points at a missing file: {named}")
    return path


def _layer(cfg: dict, source: Path) -> None:
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name}: expected a mapping at the top level")
    loaded = dict(loaded)
    cfg["models"].update(loaded.pop("models", None) or {})
    for role, options in (loaded.pop("model_options", None) or {}).items():
        if not isinstance(options, dict):
            raise ValueError(f"model_options for {role!r} must be a mapping")
        cfg["model_options"].setdefault(role, {}).update(options)  # key by key, per role
    cfg.update(loaded)


def _check_model_options(cfg: dict) -> None:
    for role, options in cfg["model_options"].items():
        spec = cfg["models"].get(role)
        if spec is None:
            raise ValueError(f"model_options for {role!r}, which has no model")
        if spec.partition(":")[0] != "ollama":
            raise ValueError(f"model_options for {role!r}: options are only supported for ollama models")
        for key, value in options.items():
            if key not in MODEL_OPTION_RULES:
                raise ValueError(f"unknown model option {key!r} for {role!r} "
                                 f"(allowed: {', '.join(sorted(MODEL_OPTION_RULES))})")
            check, wanted = MODEL_OPTION_RULES[key]
            if not check(value):
                raise ValueError(f"model option {key!r} for {role!r} must be {wanted}, got {value!r}")


def load_config(path: Path = CONFIG_PATH, env=None) -> dict:
    cfg = {**DEFAULTS, "models": dict(DEFAULTS["models"]),
           "model_options": {role: dict(o) for role, o in DEFAULTS["model_options"].items()}}
    if path.exists():
        _layer(cfg, path)
    override = override_path(env)
    if override is not None:
        _layer(cfg, override)
    cfg["override"] = override.name if override else None
    _check_model_options(cfg)
    if cfg["sandbox"] not in ("docker", "local"):
        raise ValueError(f"sandbox must be 'docker' or 'local', got {cfg['sandbox']!r}")
    if cfg["sandbox"] == "local" and (os.environ if env is None else env).get("AGENT_LOOP_ALLOW_LOCAL") != "1":
        # One config line must not be enough to run model-written code on this machine.
        raise ValueError(
            "sandbox: local runs model-written code with NO isolation (your files, your network, "
            "your account). Use sandbox: docker. If you really mean it, also set the environment "
            "variable AGENT_LOOP_ALLOW_LOCAL=1.")
    return cfg


def make_llm(spec: str, options: dict | None = None):
    """Build a chat model from 'provider:model' and that role's (already validated) options."""
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama's default context is small and silently truncates long prompts.
        return ChatOllama(model=model, num_ctx=CONFIG["num_ctx"], **(options or {}))
    raise ValueError(f"unknown provider {provider!r} in {spec!r}")


CONFIG = load_config()
