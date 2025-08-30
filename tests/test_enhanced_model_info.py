#!/usr/bin/env python3
"""Test script for EnhancedModelInfo class."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


from omnimancer.core.models import EnhancedModelInfo, ModelInfo


def test_enhanced_model_info():
    """Test EnhancedModelInfo functionality."""
    print("Testing EnhancedModelInfo class...")

    # Test basic creation
    model = EnhancedModelInfo(
        name="claude-3-5-sonnet",
        provider="anthropic",
        description="Advanced reasoning model",
        max_tokens=8192,
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
        swe_score=85.5,
        context_window=200000,
        supports_tools=True,
        supports_multimodal=True,
        latest_version=True,
    )

    # Test cost display
    cost_display = model.get_cost_display()
    print(f"Cost display: {cost_display}")
    assert cost_display == "$3.00 in, $15.00 out"

    # Test SWE display
    swe_display = model.get_swe_display()
    print(f"SWE display: {swe_display}")
    assert "85.5%" in swe_display

    # Test SWE rating calculation
    rating = model.get_swe_rating()
    print(f"SWE rating: {rating}")
    assert rating == "★★★"  # Should be 3 stars for 85.5%

    # Test cost tier
    cost_tier = model.get_cost_tier()
    print(f"Cost tier: {cost_tier}")
    assert cost_tier == "💰💰"  # Average cost is 9.0, should be 2 coins

    # Test validation
    assert model.validate_pricing() == True
    assert model.validate_swe_score() == True

    # Test free model
    free_model = EnhancedModelInfo(
        name="ollama-llama3",
        provider="ollama",
        description="Local model",
        max_tokens=4096,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        is_free=True,
    )

    assert free_model.get_cost_display() == "Free"
    assert free_model.get_cost_tier() == "Free"
    assert free_model.validate_pricing() == True

    # Test conversion to/from ModelInfo
    legacy_model = ModelInfo(
        name="gpt-4",
        provider="openai",
        description="GPT-4 model",
        max_tokens=8192,
        cost_per_token=0.00003,  # $30 per million tokens
        available=True,
        supports_tools=True,
    )

    enhanced_from_legacy = EnhancedModelInfo.from_model_info(legacy_model)
    print(f"Enhanced from legacy: {enhanced_from_legacy.name}")
    assert enhanced_from_legacy.name == "gpt-4"
    assert enhanced_from_legacy.cost_per_million_input == 30.0
    assert enhanced_from_legacy.cost_per_million_output == 30.0

    # Test conversion back to legacy
    back_to_legacy = enhanced_from_legacy.to_model_info()
    assert back_to_legacy.name == "gpt-4"
    assert abs(back_to_legacy.cost_per_token - 0.00003) < 0.000001

    # Test SWE rating edge cases
    low_score_model = EnhancedModelInfo(
        name="test-low",
        provider="test",
        description="Low score model",
        max_tokens=1000,
        cost_per_million_input=1.0,
        cost_per_million_output=1.0,
        swe_score=25.0,
    )
    assert low_score_model.get_swe_rating() == "★☆☆"

    mid_score_model = EnhancedModelInfo(
        name="test-mid",
        provider="test",
        description="Mid score model",
        max_tokens=1000,
        cost_per_million_input=1.0,
        cost_per_million_output=1.0,
        swe_score=50.0,
    )
    assert mid_score_model.get_swe_rating() == "★★☆"

    no_score_model = EnhancedModelInfo(
        name="test-none",
        provider="test",
        description="No score model",
        max_tokens=1000,
        cost_per_million_input=1.0,
        cost_per_million_output=1.0,
    )
    assert no_score_model.get_swe_rating() == ""
    assert no_score_model.get_swe_display() == "N/A"

    print("✅ All EnhancedModelInfo tests passed!")


if __name__ == "__main__":
    test_enhanced_model_info()
