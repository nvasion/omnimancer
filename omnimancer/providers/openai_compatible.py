"""Generic OpenAI-compatible provider for self-hosted endpoints.

The intended targets are vLLM, llama.cpp's server, LM Studio, and similar
backends that speak the OpenAI chat-completions dialect but differ from the
real OpenAI service in three ways this subclass encodes:

* **Keyless is the norm** — no fail-fast on a missing API key, and no
  ``Authorization`` header is sent when the key is empty.
* **The model catalog belongs to the endpoint** — ``/v1/models`` is the
  source of truth (vLLM reports ``max_model_len`` per model, which is the
  served context size); the static GPT catalog is suppressed entirely.
* **Cold starts are real** — self-hosted gateways may load a model on the
  first request, taking minutes, so the timeout error carries an actionable
  hint instead of a bare "timed out".

Use it in config via a named alias entry::

    "gateway": {
        "provider_type": "openai-compatible",
        "base_url": "http://vllm-gateway.internal:8888/v1",
        "model": "qwen3-coder-30b",
        "auth_type": "none",
        "timeout": 360
    }
"""

import logging
from typing import List

import httpx

from ..core.models import ModelInfo
from ..utils.errors import NetworkError

# Import under a name that does not end in "Provider": the lazy loader in
# ProviderInitializer.get_provider_class picks the first BaseProvider
# subclass in this module whose name ends in "provider", so the base class
# must not be discoverable here under its own name.
from .openai import OpenAIProvider as _OpenAIBase

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(_OpenAIBase):
    """OpenAI-compatible provider for self-hosted/keyless endpoints."""

    PROVIDER_LABEL = "OpenAI-compatible endpoint"

    def _require_api_key(self) -> None:
        """Keyless endpoints are the norm here — never fail on a missing key."""
        return

    def _timeout_network_error(self) -> NetworkError:
        return NetworkError(
            f"Request to {self.PROVIDER_LABEL} ({self.base_url}) timed out "
            f"after {self.request_timeout:.0f}s (one retry included). "
            "Self-hosted endpoints can take several minutes to load a model "
            "on the first request — raise this provider's timeout "
            "(providers.<name>.timeout in config.json) or retry once the "
            "model is warm."
        )

    # Same list-invariance ignore as OpenAIProvider.fetch_live_models: the
    # base declares List[ModelInfo | EnhancedModelInfo], the openai family
    # narrows to List[ModelInfo].
    def _get_static_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """No static catalog: the endpoint's /models list is the only truth."""
        return []

    def get_model_info(self) -> ModelInfo:
        """Describe the configured model, not OpenAI's GPT catalog.

        The base implementation falls back to GPT defaults for unknown
        names ("Max tokens: 4,096 | $0.00002/token"), which is noise for
        free self-hosted models.
        """
        return ModelInfo(
            name=self.model,
            provider="openai-compatible",
            description=f"{self.model} ({self.base_url})",
            max_tokens=self.max_tokens,
            cost_per_token=0.0,
            available=True,
            supports_tools=True,
            supports_multimodal=False,
        )

    async def validate_credentials(self) -> bool:
        """Cheap liveness check via GET /models (keyless-safe).

        The base class POSTs a chat completion, which on a cold self-hosted
        gateway can trigger a multi-minute model load — far too heavy for
        the /providers health display.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._build_headers(),
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug("Liveness check failed for %s: %s", self.base_url, e)
            return False

    async def fetch_live_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """Model list straight from the endpoint, no name filtering.

        vLLM includes ``max_model_len`` per model — the actual served
        context size, which must drive client-side context accounting.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._build_headers(),
                    timeout=15.0,
                )
                response.raise_for_status()

                models = []
                for entry in response.json().get("data", []):
                    model_id = entry.get("id", "")
                    if not model_id:
                        continue
                    context_len = entry.get("max_model_len") or 4096
                    models.append(
                        ModelInfo(
                            name=model_id,
                            provider="openai-compatible",
                            description=f"{model_id} (self-hosted)",
                            max_tokens=context_len,
                            cost_per_token=0.0,
                            available=True,
                            supports_tools=True,
                            supports_multimodal=False,
                        )
                    )
                return models
        except Exception as e:
            logger.warning("Could not fetch models from %s: %s", self.base_url, e)
            return []
