"""Prompt assembly must not explode on the JSON braces in the tool protocol."""

from openforge.prompts import build_system_prompt
from openforge.tools import tool_doc


def test_text_mode_prompt_contains_protocol_and_catalogue():
    prompt = build_system_prompt("/tmp/proj", tool_mode="text", tool_list=tool_doc())
    assert "## Tool protocol" in prompt
    assert '{"name": "read_file"' in prompt
    assert "run_bash" in prompt
    assert "/tmp/proj" in prompt


def test_native_mode_prompt_has_no_text_protocol():
    prompt = build_system_prompt("/tmp/proj", tool_mode="native", tool_list="")
    assert "## Tool protocol" not in prompt
    assert "tool-calling interface" in prompt


def test_project_notes_are_appended():
    prompt = build_system_prompt("/tmp", tool_mode="native", extra="use ruff")
    assert "## Project notes" in prompt and "use ruff" in prompt


def test_tool_doc_marks_optional_arguments():
    doc = tool_doc()
    assert "offset?" in doc  # optional
    assert "path:" in doc  # required, no '?'
