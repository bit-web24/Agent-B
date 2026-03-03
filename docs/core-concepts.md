# Core Concepts

This document explains the fundamental building blocks of `agentsm-rs`.

---

## States

States are the nodes in the agent's control graph. Each state has a single job:

| State | Job | Can Call LLM? | Can Execute Tools? |
|---|---|---|---|
| `Idle` | Log start, emit `Start` | No | No |
| `Planning` | Call LLM, decide next action | **Yes** | No |
| `Acting` | Execute the tool the LLM chose | No | **Yes** |
| `ParallelActing`| Run multiple tools simultaneously | No | **Yes** |
| `WaitingForHuman`| Pause for manual approval | No | No |
| `Observing` | Commit result to history | No | No |
| `Reflecting` | Compress history | Optional | No |
| `Done` | Log completion (terminal) | No | No |
| `Error` | Log failure (terminal) | No | No |

### The `State` Type

`State` is a **string-based newtype** — you can define any custom state:

```rust
// Built-in states use constructors:
State::idle()      // "Idle"
State::planning()  // "Planning"
State::acting()    // "Acting"
State::done()      // "Done"
State::error()     // "Error"

// Custom states use ::new():
State::new("Researching")
State::new("MyCustomState")
```

`State::done()` and `State::error()` are **terminal** — `is_terminal()` returns `true`, and the engine exits immediately when it reaches either. Custom terminal states can be registered via `.terminal_state("name")`.

### The `AgentState` Trait

Every state implements this trait:

```rust
#[async_trait]
pub trait AgentState: Send + Sync {
    fn name(&self) -> &'static str;

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &Arc<ToolRegistry>,
        llm:       &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event;
}
```

The `handle()` method **must always return an `Event`**. It must never panic. Errors should be written into `memory` and expressed as failure events. The `output_tx` channel is used to emit streaming events.

---

## Events

Events are the edges in the control graph — they drive transitions between states.

`Event` is a **string-based newtype** with built-in constructors:

```rust
// Built-in events:
Event::start()
Event::llm_tool_call()
Event::llm_parallel_tool_calls()
Event::llm_final_answer()
Event::max_steps()
Event::low_confidence()
Event::answer_too_short()
Event::tool_blacklisted()
Event::human_approval_required()
Event::fatal_error()
Event::human_approved()
Event::human_rejected()
Event::human_modified()
Event::tool_success()
Event::tool_failure()
Event::continue_event()
Event::needs_reflection()
Event::reflect_done()

// Custom events:
Event::new("NeedsResearch")
Event::new("ResearchDone")
```

### Key Philosophy: Tool Failures Are Data

`ToolFailure` is NOT a crash. When a tool returns an error, `ActingState` prefixes the result with `"ERROR: ..."`, stores it in `memory.last_observation`, and returns `Event::tool_failure()`. The engine transitions to `Observing`, which commits the error as a `HistoryEntry`. On the next `Planning` cycle, the LLM sees the error in its history and can decide how to recover.

---

## AgentMemory

`AgentMemory` is the agent's entire world-state, passed by `&mut` reference to every state handler.

```rust
pub struct AgentMemory {
    // Task definition
    pub task:               String,
    pub task_type:          String,         // "research" | "calculation" | "default"
    pub system_prompt:      String,

    // Execution counters
    pub step:               usize,
    pub retry_count:        usize,
    pub confidence_score:   f64,
    pub total_usage:        TokenUsage,     // Accumulated token consumption
    pub budget:             Option<TokenBudget>,

    // Tool call lifecycle
    pub current_tool_call:  Option<ToolCall>,
    pub pending_tool_calls: Vec<ToolCall>,      // For parallel execution
    pub parallel_results:   Vec<ToolResult>,
    pub last_observation:   Option<String>,
    pub pending_approval:   Option<HumanApprovalRequest>,

    // Results
    pub history:            Vec<HistoryEntry>,
    pub final_answer:       Option<String>,
    pub error:              Option<String>,

    // Configuration
    pub config:             AgentConfig,
    pub blacklisted_tools:  HashSet<String>,

    // Human-in-the-Loop
    pub approval_policy:    ApprovalPolicy,
    pub approval_callback:  Option<ApprovalCallback>,

    // Observability
    pub trace:              Trace,
}
```

### Ownership Rules (Invariants)

| Field | Written by | Cleared by |
|---|---|---|
| `step` | `PlanningState` only | Never |
| `current_tool_call` | `PlanningState` | `ObservingState` |
| `last_observation` | `ActingState` | `ObservingState` |
| `final_answer` | `PlanningState` | Never |
| `error` | `PlanningState`, `ActingState` | Never |
| `history` | `ObservingState` (push only) | `ReflectingState` (compress) |
| `trace` | Any state via `memory.log()` | Never |

---

## Transitions

The `TransitionTable` is a `HashMap<(State, Event), State>`. It is built once at startup and **never mutated**.

```rust
pub type TransitionTable = HashMap<(State, Event), State>;
```

Every legal transition is explicitly declared in `build_transition_table()`:

```
// IDLE
(Idle, Start) → Planning

// PLANNING
(Planning, LlmToolCall)           → Acting
(Planning, LlmParallelToolCalls)  → ParallelActing
(Planning, LlmFinalAnswer)        → Done
(Planning, MaxSteps)              → Error
(Planning, LowConfidence)         → Reflecting
(Planning, AnswerTooShort)        → Planning
(Planning, ToolBlacklisted)       → Planning
(Planning, HumanApprovalRequired) → WaitingForHuman
(Planning, FatalError)            → Error

// WAITING FOR HUMAN
(WaitingForHuman, HumanApproved)  → Acting
(WaitingForHuman, HumanRejected)  → Observing
(WaitingForHuman, HumanModified)  → Acting

// ACTING / PARALLEL ACTING
(Acting,         ToolSuccess)     → Observing
(Acting,         ToolFailure)     → Observing
(ParallelActing, ToolSuccess)     → Observing
(ParallelActing, ToolFailure)     → Observing

// OBSERVING
(Observing,  Continue)            → Planning
(Observing,  NeedsReflection)     → Reflecting

// REFLECTING
(Reflecting, ReflectDone)         → Planning
```

Custom transitions can be added via `.transition("FromState", "OnEvent", "ToState")`.

---

## AgentConfig

Controls the agent's behavior limits:

```rust
pub struct AgentConfig {
    pub max_steps:             usize,   // Hard cap on Planning cycles (default: 15)
    pub max_retries:           usize,   // Low-confidence retry budget (default: 3)
    pub confidence_threshold:  f64,     // Below this → LowConfidence event (default: 0.4)
    pub reflect_every_n_steps: usize,   // Compress history every N steps (default: 5)
    pub min_answer_length:     usize,   // Shorter answers → AnswerTooShort event (default: 5)
    pub parallel_tools:        bool,    // Enable/disable parallel execution (default: true)
    pub models:                HashMap<String, String>, // task_type → model name
    pub output_schema:         Option<OutputSchema>,    // Structured output schema
}
```

### Model Selection

"Resolution priority: `config.models[task_type]` → `config.models["default"]` → `""` (LlmCaller decides).

### Structured Output

When `output_schema` is set, the LLM is instructed to return JSON conforming to the schema. OpenAI uses `json_object` response format; Anthropic uses a synthetic tool.

---

## HistoryEntry and the Trace

### HistoryEntry

```rust
pub struct HistoryEntry {
    pub step:        usize,
    pub tool:        ToolCall,
    pub observation: String,    // "SUCCESS: ..." or "ERROR: ..."
    pub success:     bool,
}
```

### Trace

```rust
pub struct TraceEntry {
    pub step:      usize,
    pub state:     String,
    pub event:     String,
    pub data:      String,
    pub timestamp: DateTime<Utc>,
}
```

Access after run:

```rust
engine.trace().print();                // pretty table to stdout
engine.trace().to_json();              // JSON string
engine.trace().for_state("Planning");  // filter by state
engine.trace().len();                  // total entry count
```
