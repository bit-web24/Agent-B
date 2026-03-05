# Architecture

## Overview

`Agent-B` implements a **Hybrid State Machine** — a control loop where:

- A **directed graph** (the Transition Table) defines what can happen
- **State handlers** define what does happen at each node
- The **Engine** is the runner that connects them — it has no business logic of its own

```
                         ┌───────────────────────────────┐
                         │         PUBLIC API             │
                         │      AgentBuilder              │
                         └──────────────┬────────────────┘
                                        │ builds
                         ┌──────────────▼────────────────┐
                         │         AgentEngine            │
                         │  owns: memory, tools, llm,     │
                         │        transitions, handlers   │
                         │  runs: the state machine loop  │
                         └──────┬──────────────┬──────────┘
                                │              │
                    ┌───────────▼──┐   ┌───────▼──────────┐
                    │  Transition  │   │  State Handlers   │
                    │    Table     │   │  (states/*.rs)    │
                    │              │   │                   │
                    │ HashMap<     │   │  IdleState        │
                    │  (State,     │   │  PlanningState    │
                    │   Event),    │   │  ActingState      │
                    │  State>      │   │  ParallelActing   │
                    └──────────────┘   │  WaitingForHuman  │
                                       │  ObservingState   │
                                       │  ReflectingState  │
                                       │  DoneState        │
                                       │  ErrorState       │
                                       └──────┬────────────┘
                                              │ uses
                          ┌───────────────────┼───────────────┐
                          ▼                   ▼               ▼
                   ┌────────────┐      ┌───────────┐  ┌──────────────┐
                   │AgentMemory │      │ToolRegistry│  │  LlmCaller   │
                   │            │      │            │  │              │
                   │ task       │      │ register() │  │ OpenAiCaller │
                   │ history    │      │ execute()  │  │ Anthropic..  │
                   │ trace      │      │ schemas()  │  │ MockLlmCaller│
                   └────────────┘      └───────────┘  └──────────────┘
```

---

## State Machine Diagram

```
                  ┌──────┐
                  │ IDLE │
                  └──┬───┘
                     │ Start
                     ▼
            ┌──────────────────┐
       ┌───▶│    PLANNING      │◀──────────────────────────┐
       │    └────────┬─────────┘                           │
       │             │                                     │
       │    ┌────────┴──────────────────────┐             │
       │    │            │                  │             │
       │    ▼ ToolCall   ▼ ParallelCalls    ▼ FinalAns   │
       │ ┌────────┐  ┌───────────────┐   ┌──────┐       │
       │ │ ACTING │  │PARALLEL ACTING│   │ DONE │       │
       │ └───┬────┘  └──────┬────────┘   └──────┘       │
       │     │ToolSuccess    │                            │
       │     ▼               ▼                            │
       │ ┌───────────┐  NeedsReflection  ┌─────────────┐ │
       │ │ OBSERVING │─────────────────▶│  REFLECTING  │ │
       │ └─────┬─────┘                   └──────┬───────┘ │
       │       │ Continue                       │ Done     │
       └───────┘                                └──────────┘

MaxSteps / FatalError from any state ──▶ ERROR (terminal)
```

---

## The Engine Loop

```
loop:
  1. Is current state terminal (Done | Error | custom terminal)?
     YES → return Ok(final_answer) or Err(AgentFailed)

  2. Look up handler for current state name
     MISSING → return Err(NoHandlerForState)

  3. event = handler.handle(&mut memory, &tools, &llm, output_tx)

  4. next_state = transition_table[(current_state, event)]
     MISSING → return Err(InvalidTransition)

  5. current_state = next_state

  6. iterations++ ; if > safety_cap → return Err(SafetyCapExceeded)
```

---

## Layer Responsibilities

### AgentBuilder
- Fluent API for ergonomic construction
- Wires together memory, tools, LLM caller, transition table, and handlers
- Supports provider shortcuts (`.openai()`, `.anthropic()`, `.groq()`, `.ollama()`)
- Supports structured output via `.output_schema()`
- Supports custom state graphs via `.state()`, `.transition()`, `.terminal_state()`

### AgentEngine
- Owns all components
- Runs the loop above
- Exposes `trace()` for post-run inspection and `current_state()` for state queries

### AgentMemory
- The agent's entire world-state, passed by `&mut` to every handler
- Contains task, history, tool state, config (including `output_schema`), and trace

### TransitionTable
- A `HashMap<(State, Event), State>` built once at startup
- The single source of truth for what is legal
- Extensible with `.transition()` on the builder

### State Handlers
- Implement the async `AgentState` trait
- Each receives `(&mut AgentMemory, &Arc<ToolRegistry>, &dyn AsyncLlmCaller, Option<output_tx>)`
- Returns an `Event` — never panics, never returns nothing

### ToolRegistry
- A map of tool name → (schema, function)
- `execute()` returns `Result<String, String>` — tool failures are data

### LlmCaller
- `AsyncLlmCaller` trait with `call_async()` and `call_stream_async()`
- When `output_schema` is configured, OpenAI uses `json_object` mode and Anthropic uses a synthetic tool
- Returns `LlmResponse` variants: `ToolCall`, `ParallelToolCalls`, `FinalAnswer`, `Structured`

---

## File Structure

```
Agent-B/
├── src/
│   ├── lib.rs          ← Public API re-exports (includes OutputSchema)
│   ├── types.rs        ← State, Event, ToolCall, LlmResponse, AgentConfig, OutputSchema
│   ├── memory.rs       ← AgentMemory
│   ├── events.rs       ← Event type definitions
│   ├── transitions.rs  ← build_transition_table()
│   ├── tools.rs        ← ToolRegistry, Tool builder
│   ├── engine.rs       ← AgentEngine — the loop
│   ├── builder.rs      ← AgentBuilder — fluent API
│   ├── trace.rs        ← Trace, TraceEntry
│   ├── error.rs        ← AgentError
│   ├── budget.rs       ← TokenBudget, TokenUsage
│   ├── human.rs        ← HIP (ApprovalPolicy, HumanDecision)
│   ├── checkpoint.rs   ← CheckpointStore implementations
│   ├── mcp.rs          ← MCP server integration
│   ├── states/
│   │   ├── mod.rs      ← AgentState trait + re-exports
│   │   ├── idle.rs     ← IdleState
│   │   ├── planning.rs ← PlanningState (LLM calls, structured output handling)
│   │   ├── acting.rs   ← ActingState (tool execution)
│   │   ├── parallel_acting.rs ← ParallelActingState
│   │   ├── waiting_for_human.rs ← WaitingForHumanState
│   │   ├── observing.rs ← ObservingState
│   │   ├── reflecting.rs ← ReflectingState
│   │   ├── done.rs     ← DoneState
│   │   └── error.rs    ← ErrorState
│   └── llm/
│       ├── mod.rs      ← AsyncLlmCaller trait
│       ├── openai.rs   ← OpenAiCaller (json_object mode for structured output)
│       ├── anthropic.rs ← AnthropicCaller (synthetic tool for structured output)
│       └── mock.rs     ← MockLlmCaller (for testing)
├── examples/
│   ├── basic_agent.rs
│   ├── multi_tool_agent.rs
│   ├── anthropic_agent.rs
│   └── streaming_agent.rs
├── tests/
│   ├── integration_tests.rs  ← 40+ integration tests
│   ├── parallel_tool_test.rs
│   ├── persistence_test.rs
│   └── subagent_test.rs
└── docs/
```
