# Implementation spec — Agentic Self-Instruct (CS pipeline) reproduction

Paper: "Autodata: An agentic data scientist to create high quality synthetic data" (arXiv 2606.25996v3).
Full method notes, verbatim prompts and every ruling: docs/knowledge-base/
(read `_summary.md`, `paper-analysis.md` §3, `prompts-verbatim.md`, `ambiguities.md`).

Project root: <repo>  (uv venv at `.venv`, run `source env.sh` first;
vLLM 0.30.0 + torch 2.13 cu130 installed; HF weights already downloaded into the HF cache under env.sh's HF_HOME).
Python 3.12, `src/` layout, package `autodata`. Install: `uv pip install -e ".[dev]"`. Tests: `pytest`.

## 1. What the system does (one paper)
```
workspace/<paper_id>/   paper.txt, .opencode/tools/{evaluate_rubric.py (shim), api_config.json}, eval_input.json,
                        eval_attempts/run_NNN_<mode>/..., output/result.json, trajectory/*.jsonl, harness_summary.json
main agent (GLM-5.3, tool-calling LLM; prompt = prompts/cs/main_agent.md rendered with acceptance thresholds;
            task prompt = prompts/cs/task_prompt.md with the paper text)
   tools: task(description, prompt, subagent_type in {challenger, quality_verifier}) -> subagent final text
          bash(command)  -> sandboxed: `cat/ls/head/tail/wc` of workspace files, or running evaluate_rubric.py
          write(filePath, content) / read(filePath) -> inside the workspace only
loop (driven by the LLM, budget: run.max_rounds challenger calls, run.main_agent_max_steps LLM turns):
   challenger -> QV -> write eval_input.json -> bash evaluate_rubric.py --weak-only -> (weak passed?) -> --strong-only
   -> check strong + gap -> ACCEPTED (write output/result.json) | feedback -> new challenger round
subagents: challenger / quality_verifier = fresh Agent (GLM-5.3) with tools bash(cat only) + read; they read ./paper.txt.
evaluate_rubric.py: solver x n_attempts (weak = Qwen3.5-4B, strong = Qwen3.8-27B-FP8) -> judge (GLM-5.3) per attempt,
   binary per criterion -> score = clip(sum(w_i * I_i) / sum_{w_i>0} w_i, 0, 1) -> report text + report.json.
harness: records every round from the actual tool events; computes the harness-verified acceptance from the reports
   (guardrail against the agent "claiming" acceptance); runs the end-of-loop quality verifier on accepted items.
```

## 2. Module ownership and interfaces (fixed — do not change signatures without updating this file)

### 2.1 `autodata/config.py` (DONE, read it)
`AppConfig`, `ModelEndpoint`, `AcceptancePreset` (+ `PRESETS`: `prose_s31` default, `deployed_c1`), `EvalConfig`,
`RunConfig`, `load_config(path)`. `configs/cs_default.yaml` is the reference config.
`AcceptancePreset` forbids unknown fields and bounds score/gap thresholds to [0, 1]; merged overrides are validated
when loading `AppConfig`. `ModelEndpoint.seed: int | None = None` defaults through `AppConfig.endpoint(role)` to
`RunConfig.seed` when unset. Per-request sampling reproducibility is best-effort (server scheduling and kernels can
still vary); explicit request seeds override endpoint seeds, and legacy `extra_body.seed` overrides endpoint defaults.

### 2.2 `autodata/llm/client.py`  (owner: subagent A)
```python
@dataclass
class ToolCall: id: str; name: str; arguments: dict[str, Any]; raw_arguments: str
@dataclass
class ChatResult: content: str | None; reasoning: str | None; tool_calls: list[ToolCall]; finish_reason: str | None;
                  usage: dict[str, int]; raw: dict[str, Any]; latency_s: float
class LLMClient:
    def __init__(self, endpoint: ModelEndpoint, *, name: str = "", log_dir: Path | None = None): ...
    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None, tool_choice: str | None = None,
                   max_tokens: int | None = None, temperature: float | None = None,
                   response_format: dict | None = None, extra_body: dict | None = None,
                   seed: int | None = None) -> ChatResult
    def chat_sync(self, ...same...) -> ChatResult          # for the evaluate_rubric CLI (threads)
    usage_totals: dict[str, int]                            # prompt_tokens / completion_tokens / calls / errors
    @staticmethod
    def assistant_message(result: ChatResult) -> dict       # history message: role=assistant, content, reasoning_content (if any),
                                                            # tool_calls=[{"id","type":"function","function":{"name","arguments":<json str>}}]
```
Behaviour: uses `openai.AsyncOpenAI(base_url, api_key, timeout=endpoint.timeout_s, max_retries=0)`; sampling params from
the endpoint (temperature/top_p/max_tokens/presence_penalty natively; `top_k`, `min_p`, `repetition_penalty`,
`chat_template_kwargs` and `endpoint.extra_body` merged into `extra_body`); per-endpoint `asyncio.Semaphore(max_concurrency)`;
retry with exponential backoff (1s→60s, jitter) on connection errors, timeouts, HTTP 408/409/429/5xx, and on empty
responses whose `finish_reason` is not `length` —
up to `endpoint.max_retries`; parse `tool_calls` (function.arguments JSON → dict; if invalid JSON keep {} and raw string);
reasoning = `message.reasoning_content` or `message.reasoning` (vLLM variants); `finish_reason`; token usage; optional
per-call JSONL logging to `log_dir` (request meta + response) when set. `chat_sync` = `asyncio.run` wrapper safe to call
from threads (use a private event loop per call). `ChatResult.usage` sums all returned token counts across attempts,
including discarded empty completions; `usage_totals` includes those attempts too. Exhausted-call exceptions carry
`_autodata_usage` so the agent can account for spend even when no completion is returned. `raw` remains the final
response's unmodified payload.
Shared module-level helpers (also used by the synchronous evaluator):
```python
def retryable(exc: Exception) -> bool: ...
def backoff(attempt: int) -> float: ...  # delay only; caller sleeps
def build_request_kwargs(endpoint: ModelEndpoint, *, chat_template_kwargs: dict | None = None,
                         seed: int | None = None) -> dict: ...
```
The request builder returns fresh dictionaries and merges template options without mutating the endpoint.

### 2.3 `autodata/harness/tools.py` + `autodata/harness/agent.py`  (owner: subagent A)
```python
@dataclass
class Tool: name: str; description: str; parameters: dict; handler: Callable[[dict], Awaitable[str]]
    def to_openai(self) -> dict     # {"type":"function","function":{"name","description","parameters"}}
class Workspace:
    def __init__(self, root: Path, *, virtual_root: str = "/workspace/project"): ...
    def resolve(self, path: str) -> Path     # relative paths and virtual_root-prefixed paths resolve under root; anything
                                             # escaping root raises PermissionError
    def read_text(self, path: str, *, max_chars: int | None = None) -> str
    def write_text(self, path: str, content: str) -> Path   # creates parent dirs
def make_read_tool(ws) -> Tool        # read(filePath) -> content (error string if missing)
def make_write_tool(ws, *, allow: Callable[[str], bool] | None = None) -> Tool       # write(filePath, content) -> "Wrote N bytes to <path>"
def make_bash_tool(ws, *, evaluate_rubric_runner: Callable[[list[str]], Awaitable[str]] | None = None,
                   file_commands=("cat","ls","head","tail","wc","pwd"), max_output_chars=400_000) -> Tool
     # bash(command, timeout?) — allowed forms only:
     #   "cd /workspace/project && uv run python3 .opencode/tools/evaluate_rubric.py <args>"  (also without the cd/uv prefix,
     #   also "python3"/"python") -> evaluate_rubric_runner(argv_after_script) ; return its output verbatim
     #   "<file_command> <args...>" (single command, no pipes/redirects/;/&&/||/`$()`) -> run with subprocess in ws.root,
     #   args must resolve inside the workspace; output truncated to max_output_chars with a "[truncated]" marker
     #   anything else -> "Error: command not permitted in this sandbox. Allowed: ..." (never raises)
def make_task_tool(subagent_runner: Callable[[str, str, str], Awaitable[str]], allowed_types: list[str]) -> Tool
     # task(description, prompt, subagent_type) -> subagent_runner(subagent_type, description, prompt)
     #   unknown subagent_type -> error string listing allowed types

@dataclass
class AgentEvent: step: int; kind: Literal["llm","tool_call","tool_result","elide","final","error"]; data: dict; ts: float
@dataclass
class AgentResult: final_text: str; steps_used: int; stop_reason: Literal["final","max_steps","length","error"];
                   usage: dict[str,int]; transcript_path: Path | None; error: str | None = None
class Agent:
    def __init__(self, *, name: str, system_prompt: str, tools: list[Tool], llm: LLMClient, max_steps: int,
                 transcript_path: Path | None = None, event_hook: Callable[[AgentEvent], None] | None = None,
                 tool_result_max_chars: int = 400_000, context_budget_chars: int | None = None,
                 context_budget_tokens: int | None = None, keep_recent_tool_results: int = 6): ...
    async def run(self, task_prompt: str) -> AgentResult
```
Agent loop: messages = [system, user(task_prompt)]; each step: `llm.chat(messages, tools=[...])`; append
`LLMClient.assistant_message(result)`; if `result.tool_calls`: execute sequentially (handler exceptions → error string,
never crash), append `{"role":"tool","tool_call_id":id,"content":str}` for each; continue. If no tool calls → final
(`final_text = content or ""`). `finish_reason == "length"` with no tool calls → stop_reason "length". Steps exhausted →
"max_steps" (final_text = last assistant content or ""). Every message and event appended to the JSONL transcript as it
happens (so partial runs are inspectable). Event hook receives tool_call/tool_result events (name, arguments, result text).
Context accounting includes content, unique reasoning/reasoning_content text, and JSON tool-call arguments. On overflow,
first strip reasoning fields from assistant turns older than the most recent `keep_recent_tool_results` assistant turns,
then elide old tool results while preserving the most recent `keep_recent_tool_results` tool messages. Each change emits
an `elide` event; tool-call IDs and message ordering stay intact, and full original messages remain in the transcript.
If `context_budget_tokens` is set, it takes precedence over the character budget: use the last completion's billed
`usage["prompt_tokens"]`, plus the character delta since that request divided by four (subtracting elisions too).
Before prompt usage is available, estimate all text at four characters per token. Immutable/recent content may still
exceed the budget after all eligible elisions.

### 2.4 `autodata/cs/rubric.py`, `judge.py`, `solvers.py`, `evaluate_rubric.py`  (owner: subagent B)
- Shared `parsing.py` examines both fenced and unfenced JSON spans and returns the last valid top-level object in text
  order; fenced candidates win only when the spans coincide. QV verdicts accept bold, bullets and numbered lines,
  while a passing overall verdict still requires all four checks to pass.
- `rubric.py`: `RubricItem(criterion: str, weight: int, category: Literal["positive","negative"])`;
  `parse_rubric(obj) -> list[RubricItem]` (accepts weights like 8, "8", "+8", "-3"; category inferred from sign if
  missing; raises `RubricError` on: not a list, item missing criterion, zero weight, sign/category mismatch, <1 item);
  `score_response(items, satisfied: list[bool]) -> ScoreBreakdown(score: float, earned: int, penalty: int,
  max_positive: int, n_pos_satisfied, n_neg_triggered)` with score = clip((earned - penalty) / max_positive, 0, 1).
- `solvers.py`: `SyncClient` owns one synchronous OpenAI client, with SDK retries disabled;
  `chat(messages, *, timeout=None, chat_template_kwargs=None, seed=None)` uses the shared request builder. Solver/judge
  retry budgets remain evaluator-owned, using the same transport policy and backoff as `LLMClient`.
  `build_solver_messages(context, question, template_path) -> list[dict]` using `prompts/cs/solver_user.md`
  (`{context}`, `{question}` placeholders); `run_solver(client, context, question, ...) -> SolverAttempt(response_text,
  reasoning, finish_reason, usage, latency_s, error)`; response_text is the final answer only (reasoning stripped; if the
  model put `<think>` blocks inline, strip them). Empty/whitespace response → error "empty response" (retryable), except `finish_reason == "length"`:
  a truncated response with no final answer is returned without retry and scores zero without a judge request.
- `judge.py`: `build_judge_messages(context, question, rubric, response, system_prompt_path)`; user message lists the
  rubric as numbered lines `N. [+w positive] criterion` / `N. [-w negative] criterion`; `run_judge(...) ->
  JudgeResult(satisfied: list[bool], evidence: list[str], raw: str)`; parse the JSON object (tolerate fences/preamble);
  the `criteria` array must have exactly len(rubric) entries with matching 1-based `index`; otherwise retry up to
  `eval.judge_retries` then raise `JudgeError`. Judge sees context + question + rubric + response; NEVER the reference answer.
- `evaluate_rubric.py` CLI (`python -m autodata.cs.evaluate_rubric`, also console script `autodata-evaluate-rubric`):
  args `--input PATH --output-dir PATH --config PATH --timeout SECONDS [--weak-only | --strong-only]` (neither → both).
  `--config` = api_config.json: `{"weak_solver": ModelEndpoint, "strong_solver": ModelEndpoint, "judge": ModelEndpoint,
  "acceptance": AcceptancePreset dict, "eval": EvalConfig dict, "prompts_dir": path}`.
  `--input` = eval_input.json `{"context","question","rubric",["question_type"],["reference_answer"]}` (extra keys ignored).
  Runs the n_attempts solver attempts concurrently (threads), each followed by its judge call; a failed solver attempt is
  retried `eval.solver_retries` times; if ANY attempt still fails after retries → prints `SOLVER_ERROR: ...`, exit code 2
  (the main agent is told to retry). Judge failures → `JUDGE_ERROR`, exit 3. Malformed rubric → `RUBRIC_ERROR: <why>`,
  exit 4. Missing/invalid input → `INPUT_ERROR`, exit 5. Wall-clock guard `--timeout` per solver call.
  Output dir: `<output-dir>/run_<NNN>_<mode>/` (NNN = next free number) with `attempt_<solver>_<i>.json` (messages,
  response_text, reasoning, finish_reason, usage, judge raw output, satisfied[], breakdown), `report.json`, `report.txt`.
  `question_hash` = sha1 of json.dumps({context, question, rubric}) — `--strong-only` looks up the latest weak run with the
  same hash in the output dir, also requiring matching acceptance/eval settings, models, resolved prompts_dir and
  config_sha1. `_latest_weak(..., *, config=None, config_sha1=None, prompts_dir=None)` returns no source without all
  provenance inputs. If none exists or the source's weak stage did not pass, skip all strong requests, write
  `strong_attempts: []`, print `STRONG_SKIPPED: no passing weak result for this question (run --weak-only first)` and
  `ACCEPTANCE: FAILED (...)`, and exit 0. With no source also print `NO_WEAK_RESULT: run --weak-only first`; gap is omitted.
  Every report records config_sha1, models, prompts_dir, input_path and input_sha1. Strong-only reports additionally
  record weak_source_report and weak_source_run_dir (null without a source), retaining the source's weak model name.
  stdout = report.txt followed by a final line `REPORT_PATH: <abs path to report.json>`.
  Acceptance lines use the preset from the config (see `AcceptancePreset`): weak block prints `WEAK_PASSED (<criteria>)`
  or `WEAK_FAILED: TOO EASY (weak_avg 62.3% >= 50%)` (or the other reasons: max_weak, zero attempt); strong block
  prints `STRONG_PASSED`/`STRONG_FAILED: <reason>`, `GAP_PASSED`/`GAP_FAILED: <reason>`, and a final
  `ACCEPTANCE: ALL_SOLVER_CRITERIA_PASSED` or `ACCEPTANCE: FAILED (<reasons>)`. Report text format (weak-only example):
```
=== EVALUATE_RUBRIC REPORT (weak-only) ===
question_hash: 1a2b3c4d
rubric: 13 criteria (9 positive, 4 negative), positive weight total 48
weak solver: qwen3.5-4b x3
  attempt 1: 44.4%  (positive satisfied 4/9, negative triggered 1/4, finish=stop, 8123 completion tokens)
  attempt 2: 51.9%  (...)
  attempt 3: 48.0%  (...)
weak_avg: 48.1%  max_weak: 51.9%  min_weak: 44.4%
WEAK_PASSED (weak_avg 48.1% < 50%)
```
  `report.json`: {mode, question_hash, model names, per-attempt scores/breakdowns, weak_avg/max/min, strong_avg/max/min,
  gap, weak_passed, strong_passed, gap_passed, all_passed, failure_reasons[], acceptance preset, timestamps, run_dir}.

### 2.5 `autodata/data/s2_client.py`, `autodata/data/build_corpus.py`  (owner: subagent C)
Semantic Scholar API, key in env `S2_API_KEY` (loaded by env.sh). HARD RULE: one process-wide limiter, minimum
`S2_MIN_INTERVAL` seconds between ANY two requests (default 3.0; the nominal limit is 1 req/s but 429s were observed at
1.3 s spacing), exponential backoff on 429 (5 s, 10, 20, 40, 80, 160; jitter; max 8 tries), never parallel requests.
Endpoints: `GET https://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc_v2` (header `x-api-key`) →
{"release_id","README","files":[presigned S3 URLs]}; `POST https://api.semanticscholar.org/graph/v1/paper/batch?fields=
corpusId,title,abstract,year,publicationDate,s2FieldsOfStudy,externalIds,venue,citationCount` body {"ids":["CorpusId:<n>",...]}
(≤500 ids per call). Shard record schema (verified): `corpusid` (int), `title`, `authors` [str], `openaccessinfo`
{externalids{DOI,ArXiv,ACL,MAG,PubMedCentral,...}, license, url, status}, `body{text, annotations{paragraph,
section_header, bib_ref}}`, `bibliography{text, annotations}`. Body text median ~25k chars.
CLI `autodata-build-corpus --out data/corpus/<name>.jsonl --n-papers N --shard-indices 0,1 [--shard-seed S --n-shards K]
--min-year 2022 --field "Computer Science" --min-body-chars 8000 --max-body-chars 200000 --shard-cache data/s2orc_v2_shards`:
downloads the chosen shards (streaming to the cache dir, reuse if present, presigned URLs re-listed when expired), streams
records, batches corpus ids to `/paper/batch`, keeps records with `year >= min_year` AND `Computer Science` in
s2FieldsOfStudy categories AND a non-empty abstract AND body length in range, dedupes, stops at N. Output JSONL record:
{"paper_id": "s2_<corpusid>", "corpus_id", "title", "abstract", "year", "publication_date", "venue", "s2_fields",
"external_ids", "authors", "body_text", "n_body_chars", "shard", "release_id"}. Also `autodata.data.paper_text(record)
-> str` = "Title: ...\n\nAbstract: ...\n\n<body_text>" and `iter_corpus(path)`. Write progress + counts to stderr; resumable
(append-only; skip ids already in the output).

### 2.6 `autodata/cs/prompts.py`, `run_paper.py`, `pipeline.py`, `cot_baseline.py`, `stats.py`  (owner: orchestrator)
Uses everything above. Prompts live in `prompts/cs/*.md` (verbatim from the paper/README; main_agent.md has
`{{WEAK_CRITERIA_SHORT}}`, `{{STRONG_CRITERIA_SHORT}}`, `{{GAP_CRITERIA_SHORT}}`, `{{STRONG_CHECKLIST}}` placeholders
rendered from `AcceptancePreset`).
`pipeline.archive_workdir(workdir: Path, *, root: Path | None = None) -> None` archives an existing workspace under
`(root or workdir.parent)/_archive/<paper>.<YYYYmmdd-HHMMSS>`; same-second collisions get `.1`, `.2`, etc. Call while holding
`PaperLock(workdir)`. Locks live at `<run-root>/_locks/<paper>.lock`, use nonblocking `flock`, remain on disk after release,
and are acquired inside the paper concurrency semaphore before rechecking resume state or archiving. Every rerun archives,
including retries after errors. With retry_errors=True, only explicitly completed summaries without errors/error stop
reasons are resumable; keep-errors preserves the existing opt-out. Escaping per-paper exceptions write incomplete crash
summaries and do not abort the corpus; usage_agents.json is written in finally.
`stats` loads only direct paper workdirs, excluding `_...` and `.old.` directories, and deduplicates by latest finished_at.
`n_incomplete` aliases the existing incomplete count; `n_skipped_locked` counts distinct latest locked skips, including
corpus-log-only skips that cannot be written into another runner's workspace.

### 2.7 Serving (owner: serving subagent) — `serving/`
Three vLLM servers on the 4x B200 node, all co-located, started in this order:
1. GLM-5.3: `vllm serve nvidia/GLM-5.3-NVFP4 --served-model-name glm-5.3 --port 8000 --tensor-parallel-size 4
   --enable-expert-parallel --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.80 --max-model-len 400000
   --max-num-seqs 64 --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice
   --kv-offloading-size 400` (KV offload to CPU RAM is wanted; NO weight offload). Adjust only if it does not start.
2. Strong solver: `CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b --port 8001
   --tensor-parallel-size 2 --gpu-memory-utilization 0.14 --max-model-len 65536 --max-num-seqs 32 --reasoning-parser qwen3
   --language-model-only`
3. Weak solver: `CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3.5-4B --served-model-name qwen3.5-4b --port 8002
   --tensor-parallel-size 1 --gpu-memory-utilization 0.16 --max-model-len 65536 --max-num-seqs 48 --reasoning-parser qwen3
   --language-model-only` (a second replica on GPU 3 / port 8003 is optional).
Env: `source env.sh` (HF cache) ; logs to `logs/serve_*.log`. Health: `GET /v1/models`; smoke: one tool-call request to
GLM (tools=[{"type":"function",...}]) must return `tool_calls`; one Qwen request with
`extra_body={"chat_template_kwargs":{"enable_thinking":true}}` must return `reasoning_content`/`reasoning` + content.

## 3. Coding standards
Python 3.12, type hints, pydantic v2 for schemas, `orjson`/`json`, async (`asyncio`) for the harness, threads only inside
the evaluate_rubric CLI. No global state except the S2 limiter. Never log secrets. Every module gets pytest tests that
run without GPUs or network (fake OpenAI-compatible server = `tests/fake_openai_server.py`, stdlib `http.server` on a
free port, scripted responses; subagent A creates it, subagent B may extend it). Keep prompts verbatim; anything that
deviates from the paper must be a config knob with the paper's value as default and a comment citing the section.

## 4. Guardrails added after the first review (2026-09-27)
The paper notes agents "trying to cheat the goal" and that guardrails belong in the outer loop; the harness therefore never
trusts the agent for acceptance:
- Evaluator runs as an isolated interpreter (`python -I -m autodata.cs.evaluate_rubric`) with absolute, validated paths;
  `--config` must be the harness-written `.opencode/tools/api_config.json`, `--output-dir` must be `./eval_attempts`.
- The agent's `write` tool is allow-listed (`eval_input.json`, `output/*`, notes, top-level .md/.txt); the evaluator, its
  config and outputs, `paper.txt`, transcripts and anything importable (`*.py`, `*.pth`, `autodata*`) are read-only.
- Every evaluator report is verified for provenance (new run dir inside `eval_attempts`, exit code 0, no error, question
  hash of the actual `--input` file, acceptance preset and `n_attempts` equal to the run config) and the verdict is
  recomputed by the harness from the per-attempt `satisfied` lists with exact arithmetic; a mismatch blocks acceptance.
- Acceptance additionally requires a passing QV in the same round whose prompt contains the evaluated question (binding),
  and the accepted candidate is frozen for the final QV and the exported record. Contradictory/incomplete QV outputs fail.
- Round budget: the `task` tool refuses challenger calls beyond `run.max_rounds` and after acceptance.
- Main-agent history is compacted by removing old reasoning, then eliding old tool results beyond the context budget.
- Resume skips only completed papers without errors; every rerun archives old workspaces under `_archive/`. Per-paper
  flocks prevent two runners sharing a workspace and release on process death; summaries are written atomically with a
  config fingerprint.
