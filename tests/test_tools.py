"""Tool behaviour: numbering, guards, approvals, edits."""

import pathlib


from openforge.tools import ToolContext, execute
from openforge.tools.context import ApprovalRequest


def ctx(root, approve=lambda r: True, **kw) -> ToolContext:
    return ToolContext(root=pathlib.Path(root), approve=approve, **kw)


def test_read_file_numbers_lines(workspace):
    out, err = execute(ctx(workspace), "read_file", {"path": "src/app.py"})
    assert not err
    assert "     1| def hello():" in out
    assert "     2|     return 'hi'" in out


def test_read_file_offset_and_limit(workspace):
    big = workspace / "big.txt"
    big.write_text("\n".join(f"line{i}" for i in range(1, 101)), encoding="utf-8")
    out, _ = execute(ctx(workspace), "read_file", {"path": "big.txt", "offset": 50, "limit": 2})
    assert "    50| line50" in out
    assert "    51| line51" in out
    assert "line52" not in out


def test_write_file_creates_parents(workspace):
    out, err = execute(
        ctx(workspace), "write_file", {"path": "deep/nested/new.py", "content": "x = 1\n"}
    )
    assert not err
    assert (workspace / "deep/nested/new.py").read_text() == "x = 1\n"
    assert "created deep/nested/new.py" in out


def test_write_file_denied_writes_nothing(workspace):
    out, err = execute(
        ctx(workspace, approve=lambda r: False),
        "write_file",
        {"path": "nope.txt", "content": "hi"},
    )
    assert not err  # a denial is a normal tool result, not an error
    assert not (workspace / "nope.txt").exists()
    assert "denied" in out.lower()


def test_write_file_noop_when_identical(workspace):
    out, _ = execute(
        ctx(workspace), "write_file", {"path": "src/app.py", "content": "def hello():\n    return 'hi'\n"}
    )
    assert "nothing written" in out


def test_edit_file_replaces_once(workspace):
    out, err = execute(
        ctx(workspace),
        "edit_file",
        {"path": "src/app.py", "old_string": "return 'hi'", "new_string": "return 'bye'"},
    )
    assert not err
    assert "bye" in (workspace / "src/app.py").read_text()
    assert "+1 -1" in out


def test_edit_file_missing_string_is_an_error_with_hint(workspace):
    out, err = execute(
        ctx(workspace),
        "edit_file",
        {"path": "src/app.py", "old_string": "return 'nope'", "new_string": "x"},
    )
    assert err
    assert "old_string not found" in out
    assert "Closest match at line" in out


def test_edit_file_ambiguous_requires_replace_all(workspace):
    target = workspace / "dup.txt"
    target.write_text("a\na\n", encoding="utf-8")
    out, err = execute(
        ctx(workspace), "edit_file", {"path": "dup.txt", "old_string": "a", "new_string": "b"}
    )
    assert err and "appears 2 times" in out
    out, err = execute(
        ctx(workspace),
        "edit_file",
        {"path": "dup.txt", "old_string": "a", "new_string": "b", "replace_all": True},
    )
    assert not err
    assert target.read_text() == "b\nb\n"


def test_path_traversal_is_refused(workspace):
    out, err = execute(ctx(workspace), "read_file", {"path": "../../etc/passwd"})
    assert err
    assert "outside the workspace" in out


def test_path_traversal_allowed_when_opted_in(workspace, tmp_path):
    secret = tmp_path.parent / "outside.txt"
    secret.write_text("hello", encoding="utf-8")
    out, err = execute(
        ctx(workspace, allow_outside_root=True), "read_file", {"path": str(secret)}
    )
    assert not err
    assert "hello" in out


def test_unknown_tool(workspace):
    out, err = execute(ctx(workspace), "nope", {})
    assert err and "unknown tool" in out


def test_missing_required_argument(workspace):
    out, err = execute(ctx(workspace), "read_file", {})
    assert err and "missing required argument" in out


def test_raw_arguments_surface_a_useful_error(workspace):
    out, err = execute(ctx(workspace), "read_file", {"__raw__": "{not json"})
    assert err and "not valid JSON" in out


def test_grep_and_glob(workspace):
    out, err = execute(ctx(workspace), "grep", {"pattern": "def hello"})
    assert not err
    assert "src/app.py:1" in out
    out, _ = execute(ctx(workspace), "glob", {"pattern": "**/*.py"})
    assert "src/app.py" in out


def test_grep_rejects_bad_regex(workspace):
    out, err = execute(ctx(workspace), "grep", {"pattern": "("})
    assert err and "invalid regex" in out


def test_list_dir_marks_dirs(workspace):
    out, _ = execute(ctx(workspace), "list_dir", {"path": "."})
    assert "src/" in out


def test_run_bash_requires_approval(workspace):
    seen: list[ApprovalRequest] = []

    def approve(request):
        seen.append(request)
        return True

    out, err = execute(ctx(workspace, approve=approve), "run_bash", {"command": "echo hi"})
    assert not err
    assert "exit code: 0" in out and "hi" in out
    assert seen and seen[0].kind == "shell"


def test_run_bash_denied_does_not_execute(workspace):
    out, err = execute(
        ctx(workspace, approve=lambda r: False), "run_bash", {"command": "touch created.txt"}
    )
    assert not err
    assert not (workspace / "created.txt").exists()
    assert "denied" in out.lower()


def test_dry_run_blocks_writes(workspace):
    out, err = execute(
        ctx(workspace, dry_run=True), "write_file", {"path": "dry.txt", "content": "x"}
    )
    assert not err
    assert "DRY RUN" in out
    assert not (workspace / "dry.txt").exists()


def test_output_is_truncated(workspace):
    big = workspace / "big_out.txt"
    big.write_text("\n".join("y" * 80 for _ in range(200)), encoding="utf-8")
    out, err = execute(ctx(workspace), "read_file", {"path": "big_out.txt"}, max_output=200)
    assert not err
    assert "output truncated at 200 chars" in out


def test_absolute_path_inside_root_is_allowed(workspace):
    out, err = execute(ctx(workspace), "read_file", {"path": str(workspace / "src/app.py")})
    assert not err
    assert "def hello" in out
