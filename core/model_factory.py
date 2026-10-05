"""core.model_factory — Dynamic multi-provider LLM instantiator.

Resolves model identifiers from ``configs/models.yaml`` or parses inline
``provider:model_name`` strings.  Supports Google GenAI, OpenAI, Anthropic,
Groq, and (optionally) Ollama.

Lazy imports ensure that only the requested provider package needs to be
installed — missing optional providers won't crash the harness at import time.

Example usage::

    # From YAML registry
    model = create_model("gemini-2.0-flash")

    # From inline provider-qualified string
    model = create_model("openai:gpt-4o-mini", temperature=0.0)
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_CONFIG = _PROJECT_ROOT / "configs" / "models.yaml"

# ── Provider dispatch table ──────────────────────────────────────────────────
# Maps provider key → (module_path, class_name)

PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "groq": ("langchain_groq", "ChatGroq"),
    "ollama": ("langchain_ollama", "ChatOllama"),
}


# ── YAML registry loader ────────────────────────────────────────────────────


def load_model_registry(
    config_path: Path | str = _DEFAULT_MODELS_CONFIG,
) -> dict[str, dict[str, Any]]:
    """Load the model registry from a YAML configuration file.

    Args:
        config_path: Path to the ``models.yaml`` file.

    Returns:
        Dictionary mapping model IDs to their configuration dicts, each
        containing at least ``provider`` and ``model_name`` keys.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(
            f"Invalid models.yaml structure — expected top-level 'models' key, "
            f"got: {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
        )

    return raw["models"]


# ── Provider class resolver ──────────────────────────────────────────────────


def _resolve_provider_class(provider: str) -> type[BaseChatModel]:
    """Dynamically import and return the chat model class for a provider.

    Args:
        provider: Provider key (e.g., ``"google"``, ``"openai"``).

    Returns:
        The chat model class (e.g., ``ChatGoogleGenerativeAI``).

    Raises:
        ValueError: If the provider is not recognized.
        ImportError: If the provider package is not installed.
    """
    if provider not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {sorted(PROVIDER_MAP.keys())}"
        )

    module_path, class_name = PROVIDER_MAP[provider]

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Provider package '{module_path}' is not installed. "
            f"Install it with: pip install {module_path.replace('_', '-')}"
        ) from exc

    return getattr(module, class_name)


# ── Main factory function ────────────────────────────────────────────────────


def _build_model(
    model_id: str,
    temperature: float,
    config_path: str,
    kwargs_key: tuple[tuple[str, Any], ...],
) -> BaseChatModel:
    """Resolve and instantiate a chat model (uncached core).

    Args:
        model_id: Registry key or ``provider:model_name`` string.
        temperature: Sampling temperature for the model.
        config_path: Path to ``models.yaml`` for registry lookups.
        kwargs_key: Sorted tuple of ``(key, value)`` items with explicit
                    constructor overrides (these beat YAML settings).

    Returns:
        A freshly instantiated ``BaseChatModel``.

    Raises:
        ValueError: If the model_id cannot be resolved.
        ImportError: If the required provider package is not installed.
    """
    model_kwargs: dict[str, Any] = {}
    if ":" in model_id:
        # ── Inline provider:model_name format ────────────────────────────
        provider, model_name = model_id.split(":", maxsplit=1)
        provider = provider.strip().lower()
        model_name = model_name.strip()
        logger.info(
            "Resolving model from inline format: provider=%s, model=%s",
            provider,
            model_name,
        )
    else:
        # ── YAML registry lookup ─────────────────────────────────────────
        registry = load_model_registry(config_path)

        if model_id not in registry:
            available = sorted(registry.keys())
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {available}"
            )

        entry = registry[model_id]
        provider = entry["provider"]
        model_name = entry["model_name"]
        logger.info(
            "Resolving model from registry: id=%s, provider=%s, model=%s",
            model_id,
            provider,
            model_name,
        )

        # Extract optional configuration parameters from YAML (e.g. timeout, max_retries)
        for k, v in entry.items():
            if k not in ("provider", "model_name", "tier", "description"):
                model_kwargs[k] = v

    # Explicit kwargs override YAML settings
    model_kwargs.update(dict(kwargs_key))

    # ── Instantiate the model ────────────────────────────────────────────
    model_class = _resolve_provider_class(provider)

    return model_class(
        model=model_name,
        temperature=temperature,
        **model_kwargs,
    )


@lru_cache(maxsize=64)
def _create_model_cached(
    model_id: str,
    temperature: float,
    config_path: str,
    kwargs_key: tuple[tuple[str, Any], ...],
) -> BaseChatModel:
    """Cached wrapper around :func:`_build_model`.

    Chat model instances are stateless per configuration and thread-safe
    for ``.invoke()``, so identical ``(model_id, temperature, config_path,
    kwargs)`` tuples safely share one instance. This is critical for
    benchmark validity: the agent graph resolves its model on **every
    turn** (Zero Hardcoding), and without caching each measured turn
    would pay a fresh YAML read plus provider client-pool construction
    inside the latency window (audit §3.4).
    """
    return _build_model(model_id, temperature, config_path, kwargs_key)


def clear_model_cache() -> None:
    """Clear the model instance cache.

    Useful in tests and long-lived processes that must guarantee a freshly
    constructed provider client (e.g., after rotating API keys).
    """
    _create_model_cached.cache_clear()


def create_model(
    model_id: str,
    temperature: float = 0.0,
    config_path: Path | str = _DEFAULT_MODELS_CONFIG,
    **kwargs: Any,
) -> BaseChatModel:
    """Create (or retrieve a cached) chat model by ID or provider-qualified string.

    Resolution order:

    1. If ``model_id`` contains ``:``, parse as ``provider:model_name``.
    2. Otherwise, look up ``model_id`` in the YAML registry.

    Instances are cached per ``(model_id, temperature, config_path,
    kwargs)`` configuration: repeated calls with identical parameters —
    e.g., every turn of a benchmark run — return the same instance
    without re-reading YAML or rebuilding the provider client pool, so
    per-turn latency reflects inference, not construction. Call
    :func:`clear_model_cache` to reset.

    Args:
        model_id: Either a registry key (e.g., ``"gemini-2.0-flash"``) or a
                  provider-qualified string (e.g., ``"openai:gpt-4o-mini"``).
        temperature: Sampling temperature for the model.
        config_path: Path to ``models.yaml`` for registry lookups.
        **kwargs: Additional keyword arguments passed to the model
                  constructor. Must be hashable to benefit from caching.

    Returns:
        An instantiated ``BaseChatModel`` ready for ``.invoke()`` calls.

    Raises:
        ValueError: If the model_id cannot be resolved.
        ImportError: If the required provider package is not installed.
    """
    config_path_str = str(Path(config_path))
    kwargs_key = tuple(sorted(kwargs.items()))
    try:
        return _create_model_cached(
            model_id, temperature, config_path_str, kwargs_key
        )
    except TypeError:
        # Unhashable kwargs cannot form a cache key — construct directly.
        # (If construction itself raised TypeError, this re-raises it.)
        return _build_model(model_id, temperature, config_path_str, kwargs_key)
