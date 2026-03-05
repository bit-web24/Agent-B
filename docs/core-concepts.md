# Core Concepts

This document explains the fundamental building blocks of `Agent-B`.

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

---

## Callbacks/Hooks

Hooks provide real-time observability during agent execution. Implement the `AgentHooks` trait to receive notifications at key lifecycle points.

### The `AgentHooks` Trait

```rust
pub trait AgentHooks: Send + Sync {
    fn on_agent_start(&self, task: &str, memory: &AgentMemory) {}
    fn on_agent_end(&self, result: Result<&str, &AgentError>, memory: &AgentMemory) {}
    fn on_state_enter(&self, state: &str, memory: &AgentMemory) {}
    fn on_state_exit(&self, state: &str, event: &str, memory: &AgentMemory) {}
    fn on_llm_start(&self, model: &str, memory: &AgentMemory) {}
    fn on_llm_end(&self, model: &str, response: &LlmResponse, memory: &AgentMemory) {}
    fn on_llm_error(&self, model: &str, error: &str, memory: &AgentMemory) {}
    fn on_tool_start(&self, name: &str, args: &HashMap<String, Value>, memory: &AgentMemory) {}
    fn on_tool_end(&self, name: &str, result: &str, success: bool, memory: &AgentMemory) {}
}
```

### Built-in Implementations

| Type | Purpose |
|---|---|
| `NoopHooks` | Zero-cost default — does nothing |
| `PrintHooks` | ANSI-colored output to stdout for development |
| `CompositeHooks` | Chains multiple hooks with panic isolation |

### Usage

```rust
let engine = AgentBuilder::new("task")
    .on_hook(Arc::new(PrintHooks))
    .on_hook(Arc::new(my_custom_hook))
    .build()?;
```

---

## Prompt Templates

Templates allow variable substitution in system prompts using `{variable}` syntax.

```rust
let tpl = PromptTemplate::new("You are a {role}. Focus on {topic}.")
    .var("role", "Rust expert")
    .default_var("topic", "performance");

assert_eq!(tpl.render().unwrap(), "You are a Rust expert. Focus on performance.");
```

### Features

- **`{name}`** — replaced by bound value
- **`{{` / `}}`** — escaped literal braces
- **Defaults** — `.default_var()` used when no explicit value is bound
- **Runtime overrides** — `render_with(&extra)` for highest-priority values
- **Strict mode** — `.strict(true)` makes unresolved variables an error
- **Lenient mode** (default) — unresolved variables kept as `{name}` in output

### Builder Integration

```rust
let engine = AgentBuilder::new("task")
    .prompt_template(
        PromptTemplate::new("You are a {role}. Focus on {topic}.")
            .var("role", "security auditor")
            .default_var("topic", "web security")
    )
    .build()?;
```

---

## LLM Response Caching

The cache avoids duplicate LLM API calls by storing responses keyed by SHA-256 hash of the messages + model string.

### How It Works

1. Before each LLM call, `PlanningState` computes `cache_key(messages, model)`.
2. On **cache hit**, the response is returned immediately — no API call.
3. On **cache miss**, the LLM is called and the response is stored.

### Built-in Implementations

| Type | Purpose |
|---|---|
| `NoopCache` | Caching disabled (default) |
| `InMemoryCache` | Thread-safe in-memory cache with TTL expiration and LRU eviction |

### Usage

```rust
use std::time::Duration;

let cache = Arc::new(InMemoryCache::new(
    100,                        // max entries
    Duration::from_secs(300),   // TTL: 5 minutes
));

let engine = AgentBuilder::new("task")
    .cache(cache.clone())
    .build()?;

// After run, check stats:
let stats = cache.stats();
println!("Hits: {}, Misses: {}, Rate: {:.1}%", stats.hits, stats.misses, stats.hit_rate() * 100.0);
```

---

## Conversation Memory Strategies

Memory strategies control how conversation history is prepared before being sent to the LLM, helping manage context window limits.

### Built-in Strategies

| Strategy | Behaviour |
|---|---|
| `FullMemory` | Pass complete history unchanged (default) |
| `SlidingWindowMemory` | Keep system prompt + last N messages |
| `SummaryMemory` | Keep system prompt + rolling summary + last N messages |

### Usage

```rust
// Sliding window: keep last 10 messages
let engine = AgentBuilder::new("task")
    .memory_strategy(Arc::new(SlidingWindowMemory::new(10)))
    .build()?;

// Summary memory: rolling summary + last 4 messages
let strategy = Arc::new(SummaryMemory::new(4));
strategy.set_summary("User asked about Rust. We discussed ownership.");

let engine = AgentBuilder::new("task")
    .memory_strategy(strategy)
    .build()?;
```

The strategy is applied at the end of `build_messages()`, transforming the full message list before it reaches the LLM.

---

## Advanced Concepts

The framework natively supports 8 advanced AI patterns beyond standard state execution:

1. **Execution Contracts**: `Guard`, `Invariant`, and `PostCondition` closures.
2. **Adaptive Model Routing**: Dynamically hot-swap LLMs mid-execution.
3. **Agent Introspection**: Background telemetry for anomaly detection.
4. **Self-Healing Policies**: Auto-recover from execution/parsing failures.
5. **Plan-and-Execute**: Step-by-step pre-planning and dynamic revision.
6. **Agent Forking**: Multi-path speculative verification.
7. **Deterministic Replay**: Micro-state logging and NDJSON trace replay.
8. **Tool Composition**: Runtime synthesis of tool pipelines.

For an in-depth guide on using these capabilities, refer to [Advanced Features](advanced.md).
