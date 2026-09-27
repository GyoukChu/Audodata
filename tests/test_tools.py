from __future__ import annotations

import subprocess
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from autodata.harness.tools import (
    Workspace, make_bash_tool, make_read_tool, make_task_tool, make_write_tool,
)


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root)
    workspace.write_text("paper.txt", "first\nsecond\nthird\n")
    return workspace


def test_workspace_relative_absolute_and_virtual_paths(ws):
    expected = ws.root / "paper.txt"
    for path in ("paper.txt", "./paper.txt", str(expected), "/workspace/project/paper.txt"):
        assert ws.resolve(path) == expected
        assert ws.read_text(path) == "first\nsecond\nthird\n"
    assert ws.resolve("/workspace/project") == ws.root
    assert ws.resolve("subdir/../paper.txt") == expected


@pytest.mark.parametrize("path", [
    "../outside.txt", "../../outside", "/etc/passwd", "/workspace/project/../outside",
    "/workspace/project-other/paper.txt", "/workspace/project/../../etc/passwd",
])
def test_workspace_rejects_escapes(ws, path):
    with pytest.raises(PermissionError):
        ws.resolve(path)
    with pytest.raises(PermissionError):
        ws.write_text(path, "no")
    with pytest.raises(PermissionError):
        ws.read_text(path)


def test_workspace_rejects_symlink_escapes_and_absolute_sibling(ws):
    outside = ws.root.parent / "outside.txt"
    outside.write_text("private")
    (ws.root / "link").symlink_to(outside)
    (ws.root / "external-dir").symlink_to(ws.root.parent, target_is_directory=True)
    for path in ("link", "external-dir/outside.txt", "external-dir/new.txt", str(outside)):
        with pytest.raises(PermissionError):
            ws.resolve(path)
    assert outside.read_text() == "private"


def test_workspace_writes_utf8_and_truncates_large_papers(ws):
    paper = "界" * 150_000
    target = ws.write_text("nested/paper.txt", paper)
    assert target == ws.root / "nested/paper.txt"
    assert ws.read_text("nested/paper.txt") == paper
    assert ws.read_text("nested/paper.txt", max_chars=1000) == "界" * 1000 + "\n[truncated 149000 chars]"
    assert ws.read_text("paper.txt", max_chars=0) == "\n[truncated 19 chars]"


async def test_read_write_tool_shape_errors_and_utf8_bytes(ws):
    write = make_write_tool(ws)
    read = make_read_tool(ws)
    result = await write.handler({"filePath": "/workspace/project/output/result.json", "content": "é"})
    assert result == f"Wrote 2 bytes to {ws.root / 'output/result.json'}"
    assert await read.handler({"filePath": "output/result.json"}) == "é"
    assert (await read.handler({"filePath": "missing"})).startswith("Error:")
    assert (await write.handler({"filePath": "../escape", "content": "bad"})).startswith("Error:")
    assert (await read.handler({"filePath": "../escape"})).startswith("Error:")
    assert read.to_openai() == {"type": "function", "function": {
        "name": "read", "description": read.description, "parameters": read.parameters,
    }}


@pytest.mark.parametrize("requested", [
    "./output/../eval_input.json", "/workspace/project/eval_input.json", "eval_input.json",
])
async def test_write_allow_receives_normalized_relative_path(ws, requested):
    seen = []

    def allow(path):
        seen.append(path)
        return path == "eval_input.json"

    write = make_write_tool(ws, allow=allow)
    assert (await write.handler({"filePath": requested, "content": "{}"})).startswith("Wrote 2 bytes")
    assert seen == ["eval_input.json"]
    assert ws.read_text("eval_input.json") == "{}"
    assert "Only data files may be written" in write.description


async def test_write_allow_denies_existing_new_and_symlink_targets(ws):
    ws.write_text(".opencode/tools/api_config.json", "protected")
    (ws.root / "eval_input.json").symlink_to(ws.root / ".opencode/tools/api_config.json")
    write = make_write_tool(ws, allow=lambda path: path == "eval_input.json")
    for requested, normalized in [
        ("./new/script.py", "new/script.py"),
        (str(ws.root / ".opencode/tools/api_config.json"), ".opencode/tools/api_config.json"),
        ("eval_input.json", ".opencode/tools/api_config.json"),
    ]:
        assert await write.handler({"filePath": requested, "content": "overwrite"}) == (
            f"Error: writing {normalized} is not permitted in this workspace"
        )
    assert not (ws.root / "new").exists()
    assert await make_read_tool(ws).handler({"filePath": ".opencode/tools/api_config.json"}) == "protected"


@pytest.mark.parametrize("command,expected", [
    ("cat ./paper.txt", "first\nsecond\nthird\n"),
    ("cat /workspace/project/paper.txt", "first\nsecond\nthird\n"),
    ("head -n 1 paper.txt", "first\n"),
    ("head -n1 paper.txt", "first\n"),
    ("tail --lines=1 paper.txt", "third\n"),
    ("tail -1 paper.txt", "third\n"),
    ("head -c 5 paper.txt", "first"),
])
async def test_bash_allowed_file_commands(ws, command, expected):
    assert await make_bash_tool(ws).handler({"command": command}) == expected


async def test_bash_ls_wc_pwd_defaults_and_quoted_paths(ws):
    bash = make_bash_tool(ws)
    assert "paper.txt" in await bash.handler({"command": "ls"})
    assert "paper.txt" in await bash.handler({"command": "ls -la /workspace/project"})
    assert (await bash.handler({"command": "wc -l paper.txt"})).split()[0] == "3"
    assert (await bash.handler({"command": "pwd"})).strip() == str(ws.root)
    ws.write_text("file with spaces.txt", "spaces")
    ws.write_text("-filename", "dash")
    assert await bash.handler({"command": "cat 'file with spaces.txt'"}) == "spaces"
    assert await bash.handler({"command": "cat -- -filename"}) == "dash"


@pytest.mark.parametrize("command", [
    "cat paper.txt | wc", "cat paper.txt > out", "cat paper.txt >> out", "cat < paper.txt",
    "cat paper.txt; ls", "cat paper.txt && ls", "cat paper.txt || ls", "cat paper.txt &",
    "cat `pwd`", "cat $(pwd)", "cat paper.txt\nls", "cat paper.txt\rls",
    "rm paper.txt", "python3 -c 'print(1)'", "env cat paper.txt", "/bin/cat paper.txt",
    "cd /workspace/project && cat paper.txt", "cat ../outside", "ls /etc",
    "wc --files0-from=/etc/passwd", "ls -RL .", "ls --dereference .", "tail -f paper.txt",
    "cat", "cat -", "head -n nope paper.txt", "cat 'unterminated", "",
    "uv run python3 some_other_script.py", "python3 .opencode/tools/evaluate_rubric.py --input /etc/passwd",
    "cd /workspace/project && uv run python3 .opencode/tools/evaluate_rubric.py --weak-only && ls",
    "cd /tmp && python3 .opencode/tools/evaluate_rubric.py --weak-only",
    "python3 .opencode/tools/evaluate_rubric.py --weak-only --strong-only",
    "python3 .opencode/tools/evaluate_rubric.py --strong-only --weak-only",
])
async def test_bash_rejections_never_execute(ws, command, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a rejected command reached subprocess.run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    runner = AsyncMock(return_value="report")
    result = await make_bash_tool(ws, evaluate_rubric_runner=runner).handler({"command": command})
    assert result.startswith("Error: command not permitted in this sandbox. Allowed:")
    runner.assert_not_awaited()


async def test_bash_symlink_path_is_denied(ws):
    outside = ws.root.parent / "outside"
    outside.write_text("private")
    (ws.root / "link").symlink_to(outside)
    assert "path escapes workspace" in await make_bash_tool(ws).handler({"command": "cat link"})


EVALUATOR_ARGS = [
    "--input", "./eval_input.json", "--weak-only", "--output-dir", "./eval_attempts",
    "--config", ".opencode/tools/api_config.json", "--timeout", "600",
]


@pytest.mark.parametrize("prefix", [
    "cd /workspace/project && uv run python3 ", "uv run python3 ", "python3 ", "python ", "",
    "cd /workspace/project && python ",
])
async def test_evaluator_command_routes_exact_argv_and_verbatim_output(ws, prefix, monkeypatch):
    command = prefix + (
        ".opencode/tools/evaluate_rubric.py --input ./eval_input.json --weak-only "
        "--output-dir ./eval_attempts --config .opencode/tools/api_config.json --timeout 600"
    )
    output = "WEAK_PASSED\n" + "report" * 1000 + "\n"
    runner = AsyncMock(return_value=output)

    def forbidden(*args, **kwargs):
        pytest.fail("the evaluator must be called directly")

    monkeypatch.setattr(subprocess, "run", forbidden)
    bash = make_bash_tool(ws, evaluate_rubric_runner=runner, max_output_chars=10)
    assert await bash.handler({"command": command}) == output
    runner.assert_awaited_once_with(EVALUATOR_ARGS)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_verbatim_multiline_evaluator_commands(ws, newline):
    prompt = (Path(__file__).resolve().parents[1] / "prompts/cs/main_agent.md").read_text()
    commands = re.findall(r"  cd /workspace/project && uv run python3 .*?--timeout 600", prompt, re.DOTALL)
    assert len(commands) == 2
    for mode, command in zip(("--weak-only", "--strong-only"), commands):
        runner = AsyncMock(return_value="REPORT_PATH: report.json\n")
        bash = make_bash_tool(ws, evaluate_rubric_runner=runner)
        assert await bash.handler({"command": command.replace("\n", newline)}) == runner.return_value
        runner.assert_awaited_once_with([mode if arg == "--weak-only" else arg for arg in EVALUATOR_ARGS])


@pytest.mark.parametrize("suffix", ["\nls", "\r\nls", " \\\n    ; ls", " \\\n    && ls", " \\\n    | cat paper.txt"])
async def test_continuation_normalization_does_not_allow_extra_commands(ws, suffix):
    runner = AsyncMock()
    output = await make_bash_tool(ws, evaluate_rubric_runner=runner).handler({
        "command": "python3 .opencode/tools/evaluate_rubric.py --weak-only" + suffix,
    })
    assert output.startswith("Error: command not permitted")
    runner.assert_not_awaited()


@pytest.mark.parametrize("option", ["--input", "--output-dir", "--config"])
@pytest.mark.parametrize("equals", [False, True])
async def test_evaluator_path_validation_keeps_safe_tokens_and_denies_escapes(ws, option, equals):
    (ws.root / "external").symlink_to(ws.root.parent, target_is_directory=True)
    runner = AsyncMock(return_value="report")
    bash = make_bash_tool(ws, evaluate_rubric_runner=runner)
    prefix = "python3 .opencode/tools/evaluate_rubric.py "
    for path in ("./a/../data.json", "/workspace/project/data.json", str(ws.root / "data.json")):
        tokens = [f"{option}={path}"] if equals else [option, path]
        assert await bash.handler({"command": prefix + " ".join(tokens)}) == "report"
        runner.assert_awaited_with(tokens)
    runner.reset_mock()
    for path in ("../escape.json", "external/escape.json", "/etc/passwd", "/workspace/project/../escape"):
        tokens = [f"{option}={path}"] if equals else [option, path]
        output = await bash.handler({"command": prefix + " ".join(tokens)})
        assert "path escapes workspace" in output
    runner.assert_not_awaited()


async def test_bash_cat_only_and_evaluator_unavailable(ws):
    bash = make_bash_tool(ws, file_commands=("cat",))
    assert "first" in await bash.handler({"command": "cat paper.txt"})
    assert (await bash.handler({"command": "ls"})).startswith("Error:")
    assert (await bash.handler({"command": "python3 .opencode/tools/evaluate_rubric.py"})).startswith("Error:")
    # Supplying another name does not turn the allowlist into an arbitrary executor.
    assert (await make_bash_tool(ws, file_commands=("rm",)).handler({"command": "rm paper.txt"})).startswith("Error:")


async def test_bash_truncation(ws):
    ws.write_text("long.txt", "x" * 150_000)
    output = await make_bash_tool(ws, max_output_chars=1000).handler({"command": "cat long.txt"})
    assert output == "x" * 1000 + "\n[truncated 149000 chars]"


async def test_bash_subprocess_argument_list_cwd_timeout_and_errors(ws, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "stdout", "stderr")

    monkeypatch.setattr(subprocess, "run", run)
    bash = make_bash_tool(ws)
    assert await bash.handler({"command": "cat paper.txt", "timeout": 12}) == "stdout\nstderr"
    assert calls[0][0] == ["cat", "--", str(ws.root / "paper.txt")]
    assert calls[0][1]["cwd"] == ws.root and calls[0][1]["timeout"] == 12
    assert not calls[0][1].get("shell", False)
    await bash.handler({"command": "ls"})
    assert calls[1][1]["timeout"] == 60 and calls[1][0][-1] == str(ws.root)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert "TimeoutExpired" in await bash.handler({"command": "cat paper.txt"})
    assert (await bash.handler({})).startswith("Error:")
    assert (await bash.handler({"command": None})).startswith("Error:")


async def test_bash_runner_exception_becomes_error(ws):
    runner = AsyncMock(side_effect=RuntimeError("evaluation failed"))
    output = await make_bash_tool(ws, evaluate_rubric_runner=runner).handler({
        "command": "python3 .opencode/tools/evaluate_rubric.py --weak-only",
    })
    assert output == "Error: RuntimeError: evaluation failed"


async def test_task_dispatch_and_unknown_type():
    runner = AsyncMock(return_value="subagent final answer")
    task = make_task_tool(runner, ["challenger", "quality_verifier"])
    assert await task.handler({
        "description": "generate", "prompt": "Read the paper.", "subagent_type": "challenger",
    }) == "subagent final answer"
    runner.assert_awaited_once_with("challenger", "generate", "Read the paper.")
    error = await task.handler({"description": "", "prompt": "", "subagent_type": "unknown"})
    assert error.startswith("Error:") and "challenger" in error and "quality_verifier" in error
    assert runner.await_count == 1
