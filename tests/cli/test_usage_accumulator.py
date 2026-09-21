"""Tests for TokenAccumulator cache-token tracking and checkpoint restore."""

from omnimancer.cli.usage import TokenAccumulator
from omnimancer.core.models import ChatResponse


class TestCacheTokenAccumulation:
    def test_accumulates_cache_tokens(self):
        acc = TokenAccumulator()
        acc.add(
            ChatResponse(
                content="a",
                model_used="m",
                tokens_used=10,
                input_tokens=6,
                output_tokens=4,
                cache_read_input_tokens=1000,
                cache_creation_input_tokens=200,
            )
        )
        acc.add(
            ChatResponse(
                content="b",
                model_used="m",
                tokens_used=10,
                input_tokens=6,
                output_tokens=4,
                cache_read_input_tokens=1500,
            )
        )
        total = acc.total
        assert total["cache_read_input_tokens"] == 2500
        assert total["cache_creation_input_tokens"] == 200
        assert total["input_tokens"] == 12

    def test_totals_include_cache_keys_even_when_unused(self):
        total = TokenAccumulator().total
        assert total["cache_read_input_tokens"] == 0
        assert total["cache_creation_input_tokens"] == 0


class TestRestore:
    def test_restore_from_checkpoint_totals(self):
        acc = TokenAccumulator()
        acc.restore(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_cost_usd": 0.02,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 400,
            }
        )
        acc.add(
            ChatResponse(
                content="a",
                model_used="m",
                tokens_used=10,
                input_tokens=6,
                output_tokens=4,
            )
        )
        total = acc.total
        assert total["input_tokens"] == 106
        assert total["output_tokens"] == 54
        assert abs(total["total_cost_usd"] - 0.02) < 1e-9
        assert total["cache_read_input_tokens"] == 3000

    def test_restore_tolerates_missing_keys(self):
        acc = TokenAccumulator()
        acc.restore({"input_tokens": 5})
        assert acc.total["input_tokens"] == 5
        assert acc.total["output_tokens"] == 0
