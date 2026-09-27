You are implementing part of a research-code reproduction. Work ONLY inside this repository (git root = this directory).

READ FIRST (in this order):
1. docs/IMPLEMENTATION_SPEC.md  — the fixed interfaces. You own sections 2.2 and 2.3.
2. src/autodata/config.py       — ModelEndpoint etc. (do not modify).
3. tests/fake_openai_server.py  — shared test helper (you may extend it, keep existing API).
4. docs/knowledge-base/paper-analysis.md (section 3 only)
   and prompts/cs/main_agent.md — to understand how the harness will be used (a GLM-5.3 main agent served by vLLM with
   OpenAI-compatible tool calling; subagents; a sandboxed bash tool that only runs evaluate_rubric.py or cat/ls/head/tail/wc).

DELIVERABLES (create these files; do not touch any other source file):
- src/autodata/llm/client.py       (LLMClient, ChatResult, ToolCall exactly per spec 2.2; add an optional ctor kwarg
                                    `transport: httpx.AsyncBaseTransport | None = None` used to build the httpx client
                                    passed to openai.AsyncOpenAI(http_client=...) so tests can inject httpx.MockTransport;
                                    same for the sync path via httpx.MockTransport with a sync client if you implement
                                    chat_sync with a sync OpenAI client; chat_sync must be safe to call from worker threads)
- src/autodata/harness/tools.py    (Tool, Workspace, make_read_tool, make_write_tool, make_bash_tool, make_task_tool per 2.3)
- src/autodata/harness/agent.py    (Agent, AgentEvent, AgentResult per 2.3)
- tests/test_llm_client.py, tests/test_tools.py, tests/test_agent.py — pytest, no GPU, no network. Prefer
  httpx.MockTransport via tests/fake_openai_server.make_transport (the real-socket FakeServer may be blocked in your sandbox).
  Cover: sampling params/extra_body merging (top_k, chat_template_kwargs end up in extra_body), tool-call parsing incl. invalid
  JSON arguments, reasoning_content vs reasoning field, retry on 429/500 then success, empty-response retry,
  assistant_message() round-trip shape; Workspace path escapes (../, absolute outside root, /workspace/project prefix);
  bash sandbox: allowed cat/ls/head/tail/wc, rejected pipes/redirects/;/&&/rm, the exact verbatim evaluate_rubric command
  "cd /workspace/project && uv run python3 .opencode/tools/evaluate_rubric.py --input ./eval_input.json --weak-only --output-dir ./eval_attempts --config .opencode/tools/api_config.json --timeout 600"
  must route to evaluate_rubric_runner with argv ['--input','./eval_input.json','--weak-only','--output-dir','./eval_attempts','--config','.opencode/tools/api_config.json','--timeout','600'];
  Agent loop: multi-step tool calls then final text, max_steps stop, finish_reason=length stop, handler exception becomes a
  tool result string, transcript JSONL written incrementally, event_hook receives tool_call/tool_result events.

CONSTRAINTS:
- Python 3.12, type hints, async. Use the `openai` package (>=1.60) AsyncOpenAI/OpenAI, `httpx`, stdlib. Do not add dependencies.
- Do NOT run `uv pip install` / `pip install`; the venv at .venv is complete. Run tests with:  .venv/bin/python -m pytest tests/test_llm_client.py tests/test_tools.py tests/test_agent.py -q
- Keep the interfaces in docs/IMPLEMENTATION_SPEC.md exactly (other people are coding against them in parallel). If you must add
  a parameter, make it keyword-only with a default and mention it in your final report.
- Retries: exponential backoff 1s→60s with jitter; retry on httpx connection/timeouts, openai APIConnectionError/APITimeoutError,
  status 408/409/429/5xx, and when the response has neither content nor tool_calls (treat as empty; retry). Log to stderr with the
  endpoint name.
- LLMClient.assistant_message must produce a message that the vLLM chat template for GLM/Qwen accepts: keys role, content (string;
  use "" when None), reasoning_content only if reasoning is non-empty, tool_calls in OpenAI format (function.arguments as a JSON string).
- Tool results and file reads may be large (a paper is ~150k chars); respect the max_chars / max_output_chars limits with a clear
  "[truncated N chars]" marker.
- The bash tool must never raise; it returns strings. It must not execute arbitrary shell. Parse with shlex; reject any of
  | ; & > < ` $( and newlines. File commands run via subprocess.run([...], cwd=ws.root, timeout=timeout or 60) with args that
  resolve inside the workspace (for ls/pwd with no args use the root).

FINAL REPORT (print at the end): files created, how to run the tests, test results (pasted), any deviation from the spec.
