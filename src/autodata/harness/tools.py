"""Workspace tools and a deliberately small, shell-free command sandbox."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from pathlib import Path
import re
import shlex
import subprocess
from typing import Awaitable, Callable


def _truncate(text: str, max_chars: int | None) -> str:
    """Bound retained content; the omitted-character marker is additional."""
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars < 0:
        raise ValueError("max_chars must be nonnegative")
    return text[:max_chars] + f"\n[truncated {len(text) - max_chars} chars]"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Awaitable[str]]

    def to_openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


class Workspace:
    def __init__(self, root: Path, *, virtual_root: str = "/workspace/project"):
        self.root = Path(root).resolve()
        self.virtual_root = virtual_root.rstrip("/") or "/"

    def resolve(self, path: str) -> Path:
        if path == self.virtual_root:
            candidate = self.root
        elif path.startswith(self.virtual_root.rstrip("/") + "/"):
            candidate = self.root / path[len(self.virtual_root.rstrip("/")) + 1:]
        else:
            candidate = self.root / path
        candidate = candidate.resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"path escapes workspace: {path}")
        return candidate

    def read_text(self, path: str, *, max_chars: int | None = None) -> str:
        return _truncate(self.resolve(path).read_text(encoding="utf-8"), max_chars)

    def write_text(self, path: str, content: str) -> Path:
        resolved = self.resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return resolved


def make_read_tool(ws: Workspace) -> Tool:
    async def read(arguments: dict) -> str:
        try:
            return ws.read_text(arguments["filePath"])
        except Exception as exc:
            return f"Error: {exc}"

    return Tool("read", "Read a UTF-8 file inside the workspace.", {
        "type": "object", "properties": {"filePath": {"type": "string"}},
        "required": ["filePath"], "additionalProperties": False,
    }, read)


def make_write_tool(ws: Workspace, *, allow: Callable[[str], bool] | None = None) -> Tool:
    async def write(arguments: dict) -> str:
        try:
            relative_path = ws.resolve(arguments["filePath"]).relative_to(ws.root).as_posix()
            if allow is not None and not allow(relative_path):
                return f"Error: writing {relative_path} is not permitted in this workspace"
            content = arguments["content"]
            path = ws.write_text(relative_path, content)
            return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
        except Exception as exc:
            return f"Error: {exc}"

    return Tool("write", "Write a UTF-8 data file inside the workspace, creating parent directories. "
                "Only data files may be written.", {
        "type": "object", "properties": {
            "filePath": {"type": "string"}, "content": {"type": "string"},
        }, "required": ["filePath", "content"], "additionalProperties": False,
    }, write)


# Allowlisted flags cannot open additional files, follow directory symlinks, or
# turn these read-only utilities into a long-running process (e.g. tail -f).
_SHORT_FLAGS = {
    "cat": "AbEenstTuv", "ls": "aAbBcCdDfFghiklmnopqQrRsStTuUvwxX1",
    "head": "qv", "tail": "qv", "wc": "cmlLw", "pwd": "LP",
}
_LONG_FLAGS = {
    "cat": {"--number", "--number-nonblank", "--squeeze-blank", "--show-all",
            "--show-ends", "--show-tabs", "--show-nonprinting"},
    "ls": {"--all", "--almost-all", "--directory", "--human-readable", "--recursive",
           "--reverse", "--size", "--classify", "--color=never", "--color=auto"},
    "head": {"--quiet", "--silent", "--verbose"},
    "tail": {"--quiet", "--silent", "--verbose"},
    "wc": {"--bytes", "--chars", "--lines", "--max-line-length", "--words"},
    "pwd": {"--logical", "--physical"},
}


def _file_argv(ws: Workspace, command: str, arguments: list[str]) -> list[str]:
    flags: list[str] = []
    paths: list[str] = []
    literal = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        index += 1
        if argument == "--" and not literal:
            literal = True
            continue
        if argument == "-":
            raise ValueError("stdin operands are not permitted")
        if not literal and argument.startswith("-"):
            if command in ("head", "tail"):
                count = re.fullmatch(r"(-[nc]|--lines|--bytes)(?:=?([+-]?\d+))?", argument)
                if count:
                    if count[2] is None:
                        if index == len(arguments) or not re.fullmatch(r"[+-]?\d+", arguments[index]):
                            raise ValueError("a numeric line/byte count is required")
                        flags.extend([argument, arguments[index]])
                        index += 1
                    else:
                        flags.append(argument)
                    continue
                if re.fullmatch(r"-\d+", argument):
                    flags.append(argument)
                    continue
            if argument in _LONG_FLAGS[command] or (
                not argument.startswith("--") and len(argument) > 1
                and all(flag in _SHORT_FLAGS[command] for flag in argument[1:])
            ):
                flags.append(argument)
                continue
            raise ValueError(f"option not permitted: {argument}")
        path = ws.resolve(argument)
        # Avoid devices/FIFOs that could block or provide unbounded data.
        if path.exists() and not (path.is_file() or path.is_dir()):
            raise ValueError("only regular files and directories are permitted")
        paths.append(str(path))
    if command == "pwd":
        if paths:
            raise ValueError("pwd does not accept path arguments")
        return [command, *flags]
    if not paths:
        if command != "ls":
            raise ValueError("a workspace file path is required")
        paths = [str(ws.root)]
    return [command, *flags, "--", *paths]


def _validate_evaluator_args(ws: Workspace, arguments: list[str]) -> None:
    mode: str | None = None
    index = 0
    while index < len(arguments):
        option, equals, value = arguments[index].partition("=")
        index += 1
        if option in ("--weak-only", "--strong-only") and not equals:
            if mode is not None and option != mode:
                raise ValueError("--weak-only and --strong-only are mutually exclusive")
            mode = option
            continue
        if option not in ("--input", "--output-dir", "--config", "--timeout"):
            raise ValueError(f"evaluator option not permitted: {option}")
        if not equals:
            if index == len(arguments):
                raise ValueError(f"missing value for {option}")
            value = arguments[index]
            index += 1
        if not value:
            raise ValueError(f"missing value for {option}")
        if option == "--timeout":
            timeout = float(value)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite")
        else:
            # Validate containment without rewriting the tokens passed to the
            # runner, which owns relative and virtual workspace path resolution.
            ws.resolve(value)


def make_bash_tool(
    ws: Workspace, *,
    evaluate_rubric_runner: Callable[[list[str]], Awaitable[str]] | None = None,
    file_commands: tuple[str, ...] = ("cat", "ls", "head", "tail", "wc", "pwd"),
    max_output_chars: int = 400_000,
) -> Tool:
    allowed = tuple(command for command in file_commands if command in _SHORT_FLAGS)
    allowed_text = ", ".join(allowed)
    if evaluate_rubric_runner is not None:
        allowed_text += "; python3 .opencode/tools/evaluate_rubric.py <args>"
    denial = f"Error: command not permitted in this sandbox. Allowed: {allowed_text}"

    async def bash(arguments: dict) -> str:
        try:
            command = arguments["command"]
            if not isinstance(command, str):
                return denial
            command = re.sub(r"\\\r?\n[ \t]*", " ", command)
            if any(
                marker in command for marker in ("|", ";", ">", "<", "`", "$(", "\n", "\r", "\x00")
            ):
                return denial
            argv = shlex.split(command)
            if not argv:
                return denial
            cd_prefix = argv[:3] == ["cd", ws.virtual_root, "&&"]
            if cd_prefix:
                # This exact prefix is the only exception to the ampersand ban.
                if command.count("&") != 2:
                    return denial
                argv = argv[3:]
            elif "&" in command:
                return denial
            evaluator = argv
            if evaluator[:2] == ["uv", "run"]:
                evaluator = evaluator[2:]
            if evaluator and evaluator[0] in ("python", "python3"):
                evaluator = evaluator[1:]
            if evaluator and ws.resolve(evaluator[0]) == ws.resolve(".opencode/tools/evaluate_rubric.py"):
                if evaluate_rubric_runner is None:
                    return denial
                _validate_evaluator_args(ws, evaluator[1:])
                return await evaluate_rubric_runner(evaluator[1:])
            if cd_prefix or not argv or argv[0] not in allowed:
                return denial
            resolved = _file_argv(ws, argv[0], argv[1:])
            timeout = float(arguments.get("timeout") or 60)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite")
            completed = await asyncio.to_thread(
                subprocess.run, resolved, cwd=ws.root, timeout=timeout,
                stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
            )
            output = completed.stdout
            if completed.stderr:
                output += ("\n" if output and not output.endswith("\n") else "") + completed.stderr
            if completed.returncode:
                output = f"Error: command exited with status {completed.returncode}.\n" + output
            return _truncate(output, max_output_chars)
        except (ValueError, PermissionError) as exc:
            return f"{denial}. {exc}"
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    return Tool("bash", (
        "Run a sandboxed workspace command. Allowed: " + allowed_text
        + ". No shell operators; only the evaluator may use the cd /workspace/project && prefix."
    ), {
        "type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "description": "Timeout in seconds (default 60)."},
        }, "required": ["command"], "additionalProperties": False,
    }, bash)


def make_task_tool(
    subagent_runner: Callable[[str, str, str], Awaitable[str]], allowed_types: list[str],
) -> Tool:
    async def task(arguments: dict) -> str:
        subagent_type = arguments.get("subagent_type")
        if subagent_type not in allowed_types:
            return f"Error: unknown subagent_type {subagent_type!r}. Allowed: {', '.join(allowed_types)}"
        return await subagent_runner(subagent_type, arguments["description"], arguments["prompt"])

    return Tool("task", "Run a fresh subagent and return its final answer.", {
        "type": "object", "properties": {
            "description": {"type": "string"}, "prompt": {"type": "string"},
            "subagent_type": {"type": "string", "enum": list(allowed_types)},
        }, "required": ["description", "prompt", "subagent_type"], "additionalProperties": False,
    }, task)
