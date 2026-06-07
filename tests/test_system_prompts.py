"""Tests for system prompt builder."""

from omnimancer.cli.system_prompts import build_agent_prompt


class TestBuildAgentPrompt:

    def test_tool_capable_provider_gets_tool_section(self):
        prompt = build_agent_prompt(supports_tools=True)
        assert "TOOL CALLING" in prompt
        assert "Read, Write, Edit, Bash, Glob, Grep, WebFetch" in prompt

    def test_tool_capable_provider_no_markers(self):
        prompt = build_agent_prompt(supports_tools=True)
        assert "OPERATION MARKERS" not in prompt
        assert "[FILE_WRITE:" not in prompt
        assert "PATTERN SUMMARY" not in prompt

    def test_non_tool_provider_gets_markers(self):
        prompt = build_agent_prompt(supports_tools=False)
        assert "OPERATION MARKERS" in prompt
        assert "[FILE_WRITE:" in prompt

    def test_non_tool_provider_gets_examples(self):
        prompt = build_agent_prompt(supports_tools=False)
        assert "PATTERN SUMMARY" in prompt

    def test_non_tool_provider_no_tool_section(self):
        prompt = build_agent_prompt(supports_tools=False)
        assert "TOOL CALLING:" not in prompt

    def test_both_have_core_sections(self):
        for supports_tools in [True, False]:
            prompt = build_agent_prompt(supports_tools=supports_tools)
            assert "SECURITY FEATURES" in prompt
            assert "FILE OPERATIONS" in prompt
            assert "COMMAND EXECUTION" in prompt
            assert "AGENT EXECUTION PATTERN" in prompt
            assert "Working Directory" in prompt

    def test_default_is_no_tools(self):
        prompt = build_agent_prompt()
        assert "OPERATION MARKERS" in prompt
