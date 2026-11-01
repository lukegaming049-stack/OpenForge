"""Tool registry.

One definition per tool, consumed three ways:
  * JSON Schema for providers that support native tool calling,
  * a markdown description injected into the system prompt,
  * the dispatcher the agent loop calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import files, shell
from .context import ApprovalRequest, ToolContext, ToolError

__all__ = [
    "TOOLS",
    "Tool",
    "ToolContext",
    "ToolError",
    "ApprovalRequest",
    "schemas",
    "execute",
    "tool_doc",
    "tool_names",
]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]
    writes: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


TOOLS: dict[str, Tool] = {
    "read_file": Tool(
        name="read_file",
        description=(
            "Read a text file from the workspace. Returns the file with line numbers. "
            "Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "offset": {"type": "integer", "description": "1-based first line to return.", "default": 1},
                "limit": {"type": "integer", "description": "How many lines to return.", "default": 400},
            },
            "required": ["path"],
        },
        run=files.read_file,
    ),
    "write_file": Tool(
        name="write_file",
        description=(
            "Create a file, or replace one entirely. Prefer edit_file for existing files — "
            "it produces a smaller diff. The user sees the diff and must approve it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "content": {"type": "string", "description": "Full new file contents."},
            },
            "required": ["path", "content"],
        },
        run=files.write_file,
        writes=True,
    ),
    "edit_file": Tool(
        name="edit_file",
        description=(
            "Replace one exact string in a file. old_string must match the file byte for byte, "
            "including indentation, and must be unique unless replace_all is true."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string", "description": "Exact text to find."},
                "new_string": {"type": "string", "description": "Replacement text (may be empty to delete)."},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
        },
        run=files.edit_file,
        writes=True,
    ),
    "list_dir": Tool(
        name="list_dir",
        description="List a directory tree (depth <= 4). Use it to orient yourself in an unfamiliar repo.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "depth": {"type": "integer", "default": 1},
            },
            "required": [],
        },
        run=files.list_dir,
    ),
    "glob": Tool(
        name="glob",
        description="Find files by glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
            },
            "required": ["pattern"],
        },
        run=files.glob_files,
    ),
    "grep": Tool(
        name="grep",
        description=(
            "Search file contents with a Python regular expression. Returns path:line: text. "
            "Use it to find where a symbol is defined or used."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex."},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "description": "Only search filenames matching this glob."},
                "max_results": {"type": "integer", "default": 60},
                "ignore_case": {"type": "boolean", "default": False},
            },
            "required": ["pattern"],
        },
        run=files.grep,
    ),
    "run_bash": Tool(
        name="run_bash",
        description=(
            "Run a shell command in the workspace and return its output. "
            "The user must approve every command. Not for interactive programs or long daemons."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 120, "description": "Seconds, max 300."},
                "cwd": {"type": "string", "description": "Working directory, relative to root."},
            },
            "required": ["command"],
        },
        run=shell.run_bash,
        writes=True,
    ),
}


def tool_names() -> list[str]:
    return list(TOOLS)


def schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """JSON Schemas for native tool calling."""
    return [TOOLS[n].schema() for n in (names or TOOLS) if n in TOOLS]


def tool_doc(names: list[str] | None = None) -> str:
    """Markdown tool catalogue for the system prompt (text protocol mode)."""
    lines: list[str] = []
    for name in names or TOOLS:
        tool = TOOLS.get(name)
        if not tool:
            continue
        params = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        rendered = ", ".join(
            f"{p}{'' if p in required else '?'}: {spec.get('type', 'any')}"
            for p, spec in params.items()
        )
        lines.append(f"- **{name}**({rendered}) — {tool.description}")
    return "\n".join(lines)


def execute(
    ctx: ToolContext, name: str, arguments: dict[str, Any] | None, *, max_output: int = 24_000
) -> tuple[str, bool]:
    """Run a tool. Returns ``(output, is_error)`` — never raises."""
    tool = TOOLS.get(name)
    if tool is None:
        return (
            f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}",
            True,
        )
    args = dict(arguments or {})
    if "__raw__" in args:
        return (
            "Error: your tool arguments were not valid JSON. "
            f"You sent: {str(args['__raw__'])[:300]}. Send a JSON object of arguments.",
            True,
        )
    missing = [
        key
        for key in tool.parameters.get("required", [])
        if key not in args or args[key] in (None, "")
    ]
    if missing:
        return (f"Error: {name} is missing required argument(s): {', '.join(missing)}", True)

    try:
        output = tool.run(ctx, **args)
    except ToolError as exc:
        return (f"Error: {exc}", True)
    except TypeError as exc:
        return (f"Error: bad arguments for {name}: {exc}", True)
    except Exception as exc:  # pragma: no cover - defensive
        return (f"Error: {name} failed: {type(exc).__name__}: {exc}", True)

    text = output if isinstance(output, str) else str(output)
    if len(text) > max_output:
        text = text[:max_output] + f"\n... (output truncated at {max_output} chars)"
    return (text, False)
