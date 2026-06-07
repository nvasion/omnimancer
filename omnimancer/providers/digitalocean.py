"""
DigitalOcean inference provider implementation for Omnimancer.

DigitalOcean's GenAI serverless inference API is OpenAI-compatible, so this
provider subclasses the OpenAI provider and only overrides the default
endpoint, default model, and model catalog. The endpoint can still be
overridden per-config via ``base_url`` (e.g. a regional gateway or an agent
endpoint).
"""

from typing import Any, List

from ..core.models import ModelInfo
from .openai import OpenAIProvider


class DigitalOceanProvider(OpenAIProvider):
    """
    DigitalOcean inference provider (OpenAI-compatible).
    """

    BASE_URL = "https://inference.do-ai.run/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs: Any) -> None:
        """
        Initialize the DigitalOcean inference provider.

        Args:
            api_key: DigitalOcean inference API key (model access key)
            model: Model to use (e.g. 'llama3.3-70b-instruct'). Defaults to
                a Llama 3.3 70B instruct model when not provided.
            **kwargs: Additional configuration. Supports ``base_url`` to point
                at a custom DigitalOcean inference endpoint.
        """
        super().__init__(api_key, model or "llama3.3-70b-instruct", **kwargs)

    def get_model_info(self) -> ModelInfo:
        info = super().get_model_info()
        info.provider = "digitalocean"
        return info

    def _get_static_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """
        Static fallback list of common DigitalOcean inference models.

        Live models are fetched from ``{base_url}/models`` when available; this
        list is only used as a fallback.
        """
        return [
            # cost_per_token values are approximate blended per-token prices.
            ModelInfo(
                name="llama3.3-70b-instruct",
                provider="digitalocean",
                description="Meta Llama 3.3 70B Instruct",
                max_tokens=128000,
                cost_per_token=0.00000065,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                latest_version=True,
            ),
            ModelInfo(
                name="llama3-8b-instruct",
                provider="digitalocean",
                description="Meta Llama 3 8B Instruct - fast and efficient",
                max_tokens=8192,
                cost_per_token=0.0000002,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="openai-gpt-4o",
                provider="digitalocean",
                description="OpenAI GPT-4o via DigitalOcean inference",
                max_tokens=128000,
                cost_per_token=0.0000025,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
        ]
