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

### The `State` Enum

```rust
pub enum State {
    Idle,
    Planning,
    Acting,
    ParallelActing,
    WaitingForHuman,
    Observing,
    Reflecting,
    Done,
    Error,
}
```

`State::Done` and `State::Error` are **terminal** — `is_terminal()` returns `true` for both, and the engine exits immediately when it reaches either.

### The `AgentState` Trait

Every state implements this trait:

```rust
#[async_trait]
pub trait AgentState: Send + Sync {
    fn name(&self) -> &'static str;

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &ToolRegistry,
        llm:       &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event;
}
```

The `handle()` method **must always return an `Event`**. It must never panic. Errors should be written into `memory` and expressed as failure events.

---

## Events

Events are the edges in the control graph — they drive transitions between states.

```rust
pub enum Event {
    // Lifecycle
    Start,

    // Planning outcomes
    LlmToolCall,             // LLM requested a tool call
    LlmParallelToolCalls,   // LLM requested multiple tool calls
    LlmFinalAnswer,          // LLM produced a final answer
    MaxSteps,                // step counter hit the limit
    LowConfidence,           // confidence below threshold, retry budget remaining
    AnswerTooShort,          // final answer too short
    ToolBlacklisted,         // LLM requested a blacklisted tool
    HumanApprovalRequired,   // Approval required by policy
    FatalError,              // unrecoverable LLM error

    // Human connection
    HumanApproved,           // User approved action
    HumanRejected,           // User rejected action
    HumanModified,           // User changed tool arguments

    // Acting outcomes
    ToolSuccess,             // tool executed successfully
    ToolFailure,             // tool raised an error (becomes observation data)

    // Observing outcomes
    Continue,                // normal flow back to Planning
    NeedsReflection,         // step count triggers history compression

    // Reflecting outcomes
    ReflectDone,             // compression complete, back to Planning
}
```

### Key Philosophy: Tool Failures Are Data

`ToolFailure` is NOT a crash. When a tool returns an error, `ActingState` prefixes the result with `"ERROR: ..."`, stores it in `memory.last_observation`, and returns `Event::ToolFailure`. The engine transitions to `Observing`, which commits the error as a `HistoryEntry`. On the next `Planning` cycle, the LLM sees the error in its history and can decide how to recover.

---

## AgentMemory

`AgentMemory` is the agent's entire world-state, passed by `&mut` reference to every state handler.

```rust
pub struct AgentMemory {
    // Task definition
    pub task:               String,         // The original task
    pub task_type:          String,         // "research" | "calculation" | "default"
    pub system_prompt:      String,         // Prepended to every LLM call

    // Execution counters
    pub step:               usize,          // Incremented by PlanningState only
    pub retry_count:        usize,          // Low-confidence retries used
    pub confidence_score:   f64,            // Last LLM confidence value
    pub total_usage:        TokenUsage,     // Accumulated token consumption
    pub budget:             Option<TokenBudget>, // Resource limits

    // Tool call lifecycle
    pub current_tool_call:  Option<ToolCall>,   // Set by Planning, cleared by Observing
    pub pending_tool_calls: Vec<ToolCall>,      // For parallel execution
    pub parallel_results:   Vec<ToolResult>,    // Collected results from parallel runs
    pub last_observation:   Option<String>,     // Set by Acting, cleared by Observing
    pub pending_approval:   Option<HumanApprovalRequest>, // For HITL

    // Results
    pub history:            Vec<HistoryEntry>,  // Append-only completed tool calls
    pub final_answer:       Option<String>,     // Set by Planning on final answer
    pub error:              Option<String>,     // Set on unrecoverable errors

    // Configuration
    pub config:             AgentConfig,
    pub blacklisted_tools:  HashSet<String>,

    // Observability
    pub trace:              Trace,              // Append-only event log
}
```

### Ownership Rules (Invariants)

These are absolute. Violating them is a bug:

| Field | Written by | Cleared by |
|---|---|---|
| `step` | `PlanningState` only | Never |
| `current_tool_call` | `PlanningState` | `ObservingState` |
| `last_observation` | `ActingState` | `ObservingState` |
| `final_answer` | `PlanningState` | Never |
| `error` | `PlanningState`, `ActingState` | Never |
| `history` | `ObservingState` (push only) | `ReflectingState` (compress) |
| `trace` | Any state via `memory.log()` | Never |

### Building LLM Messages

`memory.build_messages()` constructs the full messages array for the LLM from the current state:

1. System message (if `system_prompt` is non-empty)
2. History entries as alternating assistant/user messages
3. The current task as the final user message

---

## Transitions

The `TransitionTable` is a `HashMap<(State, Event), State>`. It is built once at startup and **never mutated**.

```rust
pub type TransitionTable = HashMap<(State, Event), State>;
```

Every legal transition is explicitly declared in `build_transition_table()`:

```rust
// IDLE
(Idle,       Start)           → Planning

// PLANNING
(Planning,   LlmToolCall)             → Acting
(Planning,   LlmParallelToolCalls)   → ParallelActing
(Planning,   LlmFinalAnswer)          → Done
(Planning,   MaxSteps)                → Error
(Planning,   LowConfidence)           → Reflecting
(Planning,   AnswerTooShort)          → Planning
(Planning,   ToolBlacklisted)         → Planning
(Planning,   HumanApprovalRequired)   → WaitingForHuman
(Planning,   FatalError)              → Error

// WAITING FOR HUMAN
(WaitingForHuman, HumanApproved)      → Acting
(WaitingForHuman, HumanRejected)      → Observing
(WaitingForHuman, HumanModified)      → Acting

// ACTING / PARALLEL ACTING
(Acting,         ToolSuccess)         → Observing
(Acting,         ToolFailure)         → Observing
(ParallelActing, ToolSuccess)         → Observing
(ParallelActing, ToolFailure)         → Observing

// OBSERVING
(Observing,  Continue)        → Planning
(Observing,  NeedsReflection) → Reflecting

// REFLECTING
(Reflecting, ReflectDone)     → Planning
```

If the engine encounters a `(State, Event)` pair not in this table, it immediately returns `Err(AgentError::InvalidTransition)`.

---

## AgentConfig

Controls the agent's behavior limits:

```rust
pub struct AgentConfig {
    pub max_steps:             usize,  // Hard cap on Planning cycles
    pub max_retries:           usize,  // Low-confidence retry budget
    pub confidence_threshold:  f64,    // Below this → LowConfidence event
    pub reflect_every_n_steps: usize,  // Compress history every N steps
    pub min_answer_length:     usize,  // Shorter answers → AnswerTooShort event
    pub parallel_tools:        bool,   // Enable/disable parallel execution
}
```

### Model Selection

Model names are **not hardcoded** anywhere in the library. `PlanningState` reads them from `memory.config.models` at runtime:

**Resolution priority:**
1. `config.models[task_type]` — exact task-type match (e.g. `"calculation"`)
2. `config.models["default"]` — generic fallback
3. `""` — empty string, lets the `LlmCaller` use its own internal default

**Setting models via the builder:**
```rust
AgentBuilder::new("task")
    .model("gpt-4o")                            // sets "default" key
    .model_for("calculation", "gpt-4o-mini")    // sets "calculation" key
    .model_for("research",    "gpt-4o")         // sets "research" key
```

**Setting models via `AgentConfig`:**
```rust
AgentConfig {
    models: [
        ("default".to_string(),     "llama3.2".to_string()),
        ("calculation".to_string(), "qwen2.5-coder:7b".to_string()),
    ].into(),
    ..Default::default()
}
```

The `task_type` value you set on the builder is the lookup key. You can add any key names you like — the library doesn’t restrict the set of task type strings.

---

## HistoryEntry and the Trace

### HistoryEntry

Committed by `ObservingState` after each tool call cycle:

```rust
pub struct HistoryEntry {
    pub step:        usize,     // Which planning step this happened in
    pub tool:        ToolCall,  // What tool was called with what args
    pub observation: String,    // "SUCCESS: ..." or "ERROR: ..."
    pub success:     bool,      // Whether the tool succeeded
}
```

History is included in every subsequent LLM call via `memory.build_messages()`.

### Trace

The `Trace` is a complete, append-only event log of every state handler's significant actions:

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
engine.trace().print();           // pretty table to stdout
engine.trace().to_json();         // JSON string
engine.trace().for_state("Planning"); // filter by state
engine.trace().len();             // total entry count
```
