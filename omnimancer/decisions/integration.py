"""Optional, session-only headless routing integration.

Classifier usage and estimated cost remain in ``routing`` metadata. The existing
headless worker usage and ``total_cost_usd`` retain their worker-only meaning;
consumers can add ``routing.estimated_cost_usd`` to estimate the combined cost.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import httpx

from ..providers.openai import OpenAIProvider
from ..providers.openai_compatible import OpenAICompatibleProvider
from .routing import RoutingDecision, RoutingPolicy, classify_route

MAX_POLICY_BYTES = 64 * 1024


def _fallback(status: str) -> dict[str, Any]:
    metadata = RoutingDecision(status=status).as_dict()
    metadata["applied_target"] = None
    return metadata


def _load_policy(path: str | Path) -> tuple[RoutingPolicy | None, str | None]:
    """Read a bounded file; never expose its path or validation exception."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None, "policy_unreadable"
            raw = handle.read(MAX_POLICY_BYTES + 1)
    except (OSError, ValueError):
        return None, "policy_unreadable"
    if len(raw) > MAX_POLICY_BYTES:
        return None, "policy_too_large"
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None, "policy_invalid"
        return RoutingPolicy(**data), None
    except (ValueError, TypeError, RecursionError):
        return None, "policy_invalid"


def _available_targets(engine: Any, policy: RoutingPolicy) -> dict[str, Any]:
    """Use initialized catalogs only; discovery must not send extra requests."""
    try:
        custom_models = engine.config_manager.get_custom_models()
    except Exception:
        custom_models = []
    targets = {}
    for label, target in policy.targets.items():
        provider = engine.providers.get(target.provider)
        # Some subclasses select a deployment independently of .model. Restrict
        # routing to implementations whose outgoing model field is covered.
        if type(provider) not in (OpenAIProvider, OpenAICompatibleProvider):
            continue
        # A fresh compatible endpoint has no static catalog. Its already
        # configured model is a valid session target without discovery.
        if target.model == provider.model:
            targets[label] = target
            continue
        try:
            names = {
                model.name
                for model in provider.get_available_models()
                if model.available
            }
            names.update(
                model.name
                for model in custom_models
                if model.provider == target.provider and model.available
            )
        except Exception:
            continue
        if target.model in names:
            targets[label] = target
    return targets


async def route_headless(
    engine: Any,
    prompt: str,
    policy_path: str | Path | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    resume: str | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Choose once before the worker runs, preserving baseline on any failure.

    Enabling this feature sends the task text and configured criteria to
    TypeSafe. No configuration, tools, permissions, or conversation history
    are included. ``target`` is the classifier recommendation;
    ``applied_target`` is populated only after a successful runtime switch.
    """
    if policy_path is None:
        return None
    if resume is not None:
        return _fallback("resume")
    if any(value is not None for value in (provider, model, base_url)):
        return _fallback("explicit_selection")

    policy, error = _load_policy(policy_path)
    if policy is None:
        return _fallback(error or "policy_invalid")
    targets = _available_targets(engine, policy)
    if len(targets) < 2:
        return _fallback("insufficient_targets")
    policy = policy.model_copy(update={"targets": targets})
    outcome = await engine._fire_hook(
        "pre_send_message",
        {"message": prompt, "provider": "typesafe", "model": policy.model},
        match_target=prompt,
    )
    if not outcome.allowed:
        return _fallback("hook_blocked")
    decision = await classify_route(prompt, policy, api_key=api_key, client=client)
    metadata = decision.as_dict()
    metadata["applied_target"] = None
    if decision.target is None:
        return metadata
    if policy.mode == "shadow":
        metadata["status"] = "shadow"
        return metadata

    target = targets.get(decision.target)
    if target is None or target.provider not in engine.providers:
        metadata["status"] = "target_unavailable"
        return metadata

    previous_provider = engine.current_provider
    destination_provider = engine.providers[target.provider]
    provider_models = [(destination_provider, destination_provider.model)]
    if previous_provider is not None and previous_provider is not destination_provider:
        provider_models.append((previous_provider, previous_provider.model))
    chat_manager = engine.chat_manager
    previous_context = chat_manager.current_context
    context_snapshot = copy.deepcopy(previous_context)

    def restore() -> None:
        for instance, original_model in provider_models:
            instance.model = original_model
        engine.current_provider = previous_provider
        engine.chat_manager = chat_manager
        chat_manager.current_context = previous_context
        if previous_context is not None:
            previous_context.__dict__.clear()
            previous_context.__dict__.update(context_snapshot.__dict__)

    try:
        # Selecting an already configured model needs only a provider switch;
        # the engine otherwise insists on a populated/custom model catalog.
        switch_model = (
            None if target.model == destination_provider.model else target.model
        )
        switched = await engine.switch_model(target.provider, switch_model)
        if (
            not switched
            or engine.current_provider is not destination_provider
            or destination_provider.model != target.model
        ):
            restore()
            metadata["status"] = "switch_failed"
            return metadata
    except asyncio.CancelledError:
        restore()
        raise
    except Exception:
        restore()
        metadata["status"] = "switch_failed"
        return metadata

    metadata["status"] = "routed"
    metadata["applied_target"] = decision.target
    return metadata
