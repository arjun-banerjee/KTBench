# KTBench Harness Integration Plan

## Goal

Replace the hand-rolled tool loop in `src/ktbench/llm/agent.py` with thin adapters that plug KTBench's existing tools (`tools/tools.py`) into established frontier agent harnesses. The harness manages the LLM turn loop; KTBench supplies the domain logic.

---

## Framework Survey

Four harnesses evaluated against three criteria: tool definition simplicity, multi-provider support, and production readiness.

| | Inspect AI (AISI) | smolagents (HF) | LangGraph | AutoGen v0.4 |
|---|---|---|---|---|
| Tool interface | `@tool` + async def | `Tool` subclass | `@tool` / `StructuredTool` | async function |
| Lines per tool | ~12 | ~10 | ~5 | ~4 |
| Loop management | `use_tools()` solver | internal | `ToolNode` graph | `agent.run()` |
| Multi-provider | OpenAI, Anthropic, Gemini, Grok, Mistral | 100+ via LiteLLM | widest (any ChatModel) | OpenAI native; others via ext |
| Status | AISI production | Research-grade | Production | v0.4 active |
| Best fit | Eval benchmarking | Rapid prototyping | Flexible production | Multi-agent |

**Decision: support all four, in priority order: Inspect AI → LangGraph → smolagents → AutoGen.**

Inspect AI is the most natural fit because KTBench is a benchmark and Inspect AI is built for exactly that use case (tasks, solvers, scores). LangGraph is the strongest general-purpose production choice. smolagents and AutoGen are secondary targets.

---

## Key Design Constraint

Every harness expects tools to return `str`, not a custom type. Our `ToolResult` (which has `.output: str` and `.metadata: dict`) must be serialized at the tool boundary. Proposed serializer:

```python
import json

def _fmt(result: ToolResult) -> str:
    if not result.metadata:
        return result.output
    return result.output + "\n\n[meta] " + json.dumps(result.metadata, default=str)
```

`ToolContext` construction (same across all integrations):
```python
ctx = ToolContext(problem=problem, device=device, global_seed=seed, n_timing_trials=20)
```

---

## File Layout

```
KTBench/
└── integrations/
    ├── inspect_ai.py    # Inspect AI solver + @tool wrappers
    ├── langgraph.py     # LangGraph ReAct agent factory
    ├── smolagents.py    # smolagents Tool subclasses + agent factory
    └── autogen.py       # AutoGen v0.4 async wrappers + AssistantAgent factory
```

Each file is self-contained: `from tools.tools import get_tools, ToolContext, ToolResult` plus the harness-specific imports. No inter-file dependencies.

---

## Implementation Plan

### Priority 1 — Inspect AI (`integrations/inspect_ai.py`)

Inspect AI maps directly onto KTBench's structure: a `Task` wraps a `Problem`, a `Solver` chain drives the agent, and `Score` records the final score. The tool-calling loop is the `use_tools()` solver.

**Steps:**

1. Write `_make_tool(ktbench_tool, ctx)` — converts a `Tool` instance into an Inspect `ToolDef` using `ToolDef(tool=async_fn, name=..., description=..., parameters=json_schema_from_input_schema)`.

2. Write `ktbench_task(problem, device, model_name, ...)` — returns an Inspect `Task`:
   ```python
   Task(
       dataset=[Sample(input=build_prompt(problem))],
       solver=[use_tools(*inspect_tools), generate()],
       scorer=match(),   # custom scorer reading submit_kernel metadata
   )
   ```

3. Write a custom `Scorer` that reads the `submit_kernel` tool call metadata from the `TaskState` message history and extracts `final_score`, `sol_score`, `stress_pass_rate`.

4. Write a `run_ktbench_eval(problems, model, device)` entry point that calls `eval(tasks, model=model)`.

**Additions required to Inspect AI itself:** none — `use_tools()` + `generate()` is the standard pattern.

**Install:** `pip install inspect-ai`

---

### Priority 2 — LangGraph (`integrations/langgraph.py`)

LangGraph's `create_react_agent` gives a working agent in ~5 lines. Use `StructuredTool.from_function` to get precise control over tool schemas without relying on docstring parsing.

**Steps:**

1. Write `_make_lc_tool(ktbench_tool, ctx)` — wraps each `Tool.execute()` in a `StructuredTool`:
   ```python
   StructuredTool.from_function(
       func=lambda **kw: _fmt(ktbench_tool.execute(ctx, **kw)),
       name=ktbench_tool.name,
       description=ktbench_tool.description,
       args_schema=_pydantic_from_schema(ktbench_tool.input_schema),
   )
   ```

2. Write `make_agent(problem, model, device, ...)` factory:
   ```python
   llm = ChatOpenAI(model=model, ...)  # or ChatAnthropic, ChatGoogle, etc.
   lc_tools = [_make_lc_tool(t, ctx) for t in get_tools()]
   return create_react_agent(llm, lc_tools)
   ```

3. Write `run_agent(problem, model, device, ...)` that calls `agent.invoke({"messages": [("user", build_prompt(problem))]})` and extracts the `submit_kernel` result from the message history.

**Provider selection:** pass `ChatOpenAI(base_url=..., api_key=...)` for Azure/Grok; use `ChatAnthropic` for Anthropic. The `make_agent` factory should accept a pre-constructed `ChatModel`.

**Install:** `pip install langchain-core langgraph langchain-openai`

---

### Priority 3 — smolagents (`integrations/smolagents.py`)

smolagents is the fastest path to prototype. The `Tool` subclass maps almost 1:1 onto KTBench's `Tool` ABC — same `name`, `description`, `inputs` dict, and `forward()` method. Use `ToolCallingAgent` (not `CodeAgent`) to preserve structured tool calls.

**Steps:**

1. Write `_make_smol_tool(ktbench_tool, ctx)` — generates a `smolagents.Tool` subclass dynamically:
   ```python
   class _Adapter(smolagents.Tool):
       name = ktbench_tool.name
       description = ktbench_tool.description
       inputs = {k: {"type": "string", "description": v["description"]}
                 for k, v in ktbench_tool.input_schema["properties"].items()}
       output_type = "string"
       def forward(self, **kw): return _fmt(ktbench_tool.execute(ctx, **kw))
   ```

2. Write `make_agent(problem, model_id, device, ...)` factory:
   ```python
   model = LiteLLMModel(model_id=model_id)  # handles OpenAI, Azure, Grok, Anthropic
   tools = [_make_smol_tool(t, ctx) for t in get_tools()]
   return ToolCallingAgent(tools=tools, model=model)
   ```

3. `agent.run(build_prompt(problem))` returns the final text; parse `submit_kernel` metadata from the agent's step log.

**Install:** `pip install smolagents[litellm]`

---

### Priority 4 — AutoGen v0.4 (`integrations/autogen.py`)

AutoGen is most useful if KTBench later needs a multi-agent setup (e.g., separate agents for "write kernel" and "evaluate kernel"). For single-agent use, it's the most verbose of the four.

**Steps:**

1. Write `_make_autogen_tool(ktbench_tool, ctx)` — returns an async function:
   ```python
   async def _fn(**kw: str) -> str:
       return _fmt(ktbench_tool.execute(ctx, **kw))
   _fn.__name__ = ktbench_tool.name
   _fn.__doc__ = ktbench_tool.description
   ```

2. Write `make_agent(problem, model, device, ...)` factory:
   ```python
   client = OpenAIChatCompletionClient(model=model)
   tools = [_make_autogen_tool(t, ctx) for t in get_tools()]
   return AssistantAgent(
       name="ktbench_agent",
       model_client=client,
       tools=tools,
       system_message=SYSTEM_PROMPT_TOOLS,
   )
   ```

3. `asyncio.run(agent.run(task=build_prompt(problem)))` — extract final score from `TaskResult` messages.

**Install:** `pip install autogen-agentchat autogen-ext[openai]`

---

## Shared Infrastructure (add to `integrations/`)

`integrations/__init__.py` — common helpers all four integrations reuse:

```python
import json
from tools.tools import ToolResult

def fmt_result(result: ToolResult) -> str:
    if not result.metadata:
        return result.output
    return result.output + "\n\n[meta] " + json.dumps(result.metadata, default=str)

def extract_final_score(messages) -> dict | None:
    """Scan a message history for the last submit_kernel [meta] block."""
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or str(msg)
        if "submit_kernel" in content and "[meta]" in content:
            try:
                meta_str = content.split("[meta]")[-1].strip()
                return json.loads(meta_str)
            except json.JSONDecodeError:
                pass
    return None
```

---

## pyproject.toml Optional Groups

Add one optional group per harness so users only install what they need:

```toml
[project.optional-dependencies]
inspect = ["inspect-ai>=0.3"]
langgraph = ["langgraph>=0.2", "langchain-core>=0.3", "langchain-openai>=0.2"]
smolagents = ["smolagents[litellm]>=1.0"]
autogen = ["autogen-agentchat>=0.4", "autogen-ext[openai]>=0.4"]
```

Install example: `pip install -e ".[triton,llm,langgraph]"`

---

## What to Do with `tools/` and `llm/agent.py`

- **`tools/tools.py`** — keep as-is. It is harness-agnostic domain logic. All four integrations import from it.
- **`llm/agent.py`** (`TranslationAgent`) — keep the single-shot `generate()` path (it's useful for quick evals and testing). Remove or deprecate the hand-rolled `_tool_loop_responses` and `_tool_loop_chat` once at least one harness integration is working. The harness owns the loop; we own the tools.
- **`scripts/run_agent.py`** — add a `--harness` flag: `--harness inspect|langgraph|smolagents|autogen`. Currently defaults to the hand-rolled loop; long-term defaults to inspect.

---

## Recommended Execution Order

1. **Implement `integrations/inspect_ai.py`** — most aligned with benchmark use case; forces clean Task/Score separation.
2. **Add `integrations/langgraph.py`** — best provider coverage; most likely to be used by external contributors.
3. **Test both with the `softmax_h200_to_triton` problem** — end-to-end with a real LLM call.
4. **Add `integrations/smolagents.py`** — low effort, good for HF-ecosystem users.
5. **Add `integrations/autogen.py`** — only if multi-agent scenarios become needed.
6. **Deprecate `llm/agent.py` tool loop** once step 1+2 are validated.
