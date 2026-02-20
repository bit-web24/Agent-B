# Architecture

## Overview

`agentsm-rs` implements a **Hybrid State Machine** — a control loop where:

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
                    │  State>      │   │  ObservingState   │
                    └──────────────┘   │  ReflectingState  │
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
       │    │                               │             │
       │    ▼ LlmToolCall                   ▼ LlmFinalAns │
       │ ┌────────┐                      ┌──────┐         │
       │ │ ACTING │                      │ DONE │         │
       │ └───┬────┘                      └──────┘         │
       │     │ ToolSuccess/ToolFailure                     │
       │     ▼                                            │
       │ ┌───────────┐  NeedsReflection  ┌─────────────┐ │
       │ │ OBSERVING │─────────────────▶ │  REFLECTING  │ │
       │ └─────┬─────┘                   └──────┬───────┘ │
       │       │ Continue                       │ ReflectDone
       └───────┘                                └──────────┘

MaxSteps / FatalError from any state ──▶ ERROR (terminal)
```

---

## The Engine Loop

The engine loop in `AgentEngine::run()` is intentionally simple — **no business logic, ever**:

```
loop:
  1. Is current state terminal (Done | Error)?
     YES → return Ok(final_answer) or Err(AgentFailed)
  
  2. Look up handler for current state name
     MISSING → return Err(NoHandlerForState)
  
  3. event = handler.handle(&mut memory, &tools, &llm)
  
  4. next_state = transition_table[(current_state, event)]
     MISSING → return Err(InvalidTransition)
  
  5. current_state = next_state
  
  6. iterations++ ; if > safety_cap → return Err(SafetyCapExceeded)
```

That is literally the entire engine. All intelligence lives in the state handlers.

---

## Layer Responsibilities

### AgentBuilder
- Fluent API for ergonomic construction
- Wires together memory, tools, LLM caller, transition table, and handlers
- Returns a ready-to-run `AgentEngine`

### AgentEngine
- Owns all components
- Runs the loop above
- Exposes `trace()` for post-run inspection
- Exposes `current_state()` for inspection after `run()`

### AgentMemory
- The agent's entire world-state, passed by `&mut` to every handler
- Contains: task, history, current_tool_call, last_observation, final_answer, error, config, trace

### TransitionTable
- A `HashMap<(State, Event), State>` built once at startup
- The single source of truth for what is legal
- Cannot be modified after construction

### State Handlers
- Implement the `AgentState` trait
- Each receives `(&mut AgentMemory, &ToolRegistry, &dyn LlmCaller)`
- Returns an `Event` — never panics, never returns nothing

### ToolRegistry
- A map of tool name → (schema, function)
- `execute()` returns `Result<String, String>` — tool failures are data, not panics

### LlmCaller
- A trait with a single `call()` method
- `OpenAiCaller`, `AnthropicCaller`, and `MockLlmCaller` are provided
- Any `AsyncLlmCaller` can be wrapped into a sync `LlmCaller` with `LlmCallerExt`

---

## Data Flow: Planning → Acting → Observing

```
PlanningState::handle()
  ├── calls llm.call(memory, tools, model)
  │     └── LlmCaller builds messages from memory.build_messages()
  │     └── LlmCaller builds tool schemas from tools.schemas()
  │     └── LlmCaller sends to LLM API
  │     └── Returns LlmResponse::ToolCall { tool, confidence }
  │
  ├── validates confidence, blacklist
  ├── sets memory.current_tool_call = Some(tool)
  └── returns Event::LlmToolCall

        ↓ engine transitions to Acting

ActingState::handle()
  ├── reads memory.current_tool_call
  ├── calls tools.execute(name, args) → Result<String, String>
  ├── sets memory.last_observation = "SUCCESS: ..." or "ERROR: ..."
  └── returns Event::ToolSuccess or Event::ToolFailure

        ↓ engine transitions to Observing

ObservingState::handle()
  ├── takes memory.current_tool_call
  ├── takes memory.last_observation
  ├── pushes HistoryEntry { tool, observation, step, success }
  ├── clears current_tool_call and last_observation
  └── returns Event::Continue (or NeedsReflection)

        ↓ engine transitions back to Planning
```

---

## File Structure

```
agentsm-rs/
├── src/
│   ├── lib.rs          ← Public API re-exports
│   ├── types.rs        ← State, Event, ToolCall, HistoryEntry, LlmResponse, AgentConfig
│   ├── memory.rs       ← AgentMemory — the agent's world-state
│   ├── events.rs       ← Event enum — all possible state transition triggers
│   ├── transitions.rs  ← build_transition_table() — the complete graph
│   ├── tools.rs        ← ToolRegistry — registration, schema, execution
│   ├── engine.rs       ← AgentEngine — the loop
│   ├── builder.rs      ← AgentBuilder — fluent construction API
│   ├── trace.rs        ← Trace, TraceEntry — event-sourcing log
│   ├── error.rs        ← AgentError — all error variants
│   ├── states/
│   │   ├── mod.rs      ← AgentState trait + re-exports
│   │   ├── idle.rs     ← IdleState
│   │   ├── planning.rs ← PlanningState (LLM calls, model selection)
│   │   ├── acting.rs   ← ActingState (tool execution)
│   │   ├── observing.rs← ObservingState (commits history, triggers reflection)
│   │   ├── reflecting.rs←ReflectingState (history compression)
│   │   ├── done.rs     ← DoneState (terminal, logs completion)
│   │   └── error.rs    ← ErrorState (terminal, logs failure)
│   └── llm/
│       ├── mod.rs      ← LlmCaller trait, AsyncLlmCaller, LlmCallerExt/SyncWrapper
│       ├── openai.rs   ← OpenAiCaller (async-openai based)
│       ├── anthropic.rs← AnthropicCaller (raw reqwest HTTP)
│       └── mock.rs     ← MockLlmCaller (for testing)
├── examples/
│   ├── basic_agent.rs       ← Minimal OpenAI agent
│   ├── multi_tool_agent.rs  ← Multi-tool with blacklisting
│   └── anthropic_agent.rs   ← Anthropic/Claude agent
├── tests/
│   └── integration_tests.rs ← 14 integration tests
└── docs/                    ← You are here
```
