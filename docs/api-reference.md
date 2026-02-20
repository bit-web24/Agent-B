# API Reference

Complete public API surface of `agentsm-rs`. All types are in the `agentsm` crate root unless noted.

---

## Builder

### `AgentBuilder`

The primary entry point. Use this to construct `AgentEngine`.

```rust
impl AgentBuilder {
    /// Create a builder with the given task description.
    pub fn new(task: impl Into<String>) -> Self

    /// Set the task type for model selection routing.
    pub fn task_type(self, t: impl Into<String>) -> Self

    /// Set the system prompt prepended to every LLM call.
    pub fn system_prompt(self, p: impl Into<String>) -> Self

    // ── Provider shortcuts ───────────────────────────────────────────────

    /// Set the LLM caller explicitly (escape hatch for custom providers).
    pub fn llm(self, llm: Box<dyn LlmCaller>) -> Self

    /// Use OpenAI API. Pass API key or "" to read OPENAI_API_KEY from env.
    pub fn openai(self, api_key: impl Into<String>) -> Self

    /// Use Groq API (OpenAI-compatible). Pass Groq API key.
    pub fn groq(self, api_key: impl Into<String>) -> Self

    /// Use local Ollama. Pass base URL or "" for http://localhost:11434/v1.
    pub fn ollama(self, base_url: impl Into<String>) -> Self

    /// Use Anthropic API. Pass API key or "" to read ANTHROPIC_API_KEY from env.
    pub fn anthropic(self, api_key: impl Into<String>) -> Self

    // ── Retry policy ─────────────────────────────────────────────────────

    /// Wrap LLM caller with exponential-backoff retry (1s, 2s, 4s, … cap 30s).
    /// Auth errors (401/403) are never retried.
    pub fn retry_on_error(self, n: u32) -> Self

    // ── Configuration ────────────────────────────────────────────────────

    pub fn config(self, config: AgentConfig) -> Self
    pub fn max_steps(self, n: usize) -> Self
    pub fn model(self, model: impl Into<String>) -> Self          // sets "default" key
    pub fn model_for(self, task_type, model) -> Self              // per-task-type
    pub fn models(self, models: HashMap<String, String>) -> Self  // full map

    // ── Tool registration ────────────────────────────────────────────────

    /// Register a raw tool (name, description, JSON Schema, function).
    pub fn tool(self, name, description, schema: Value, func: ToolFn) -> Self

    /// Register a tool built with the Tool builder.
    pub fn add_tool(self, tool: Tool) -> Self

    /// Prevent the agent from using the named tool.
    pub fn blacklist_tool(self, name: impl Into<String>) -> Self

    // ── Build ────────────────────────────────────────────────────────────

    pub fn build(self) -> Result<AgentEngine, AgentError>
    pub fn build_with_handlers(self, extra: HashMap<&'static str, Box<dyn AgentState>>) -> Result<AgentEngine, AgentError>
}
```

---

## Engine

### `AgentEngine`

```rust
impl AgentEngine {
    /// Construct directly (prefer AgentBuilder for most use cases).
    pub fn new(
        memory:      AgentMemory,
        tools:       ToolRegistry,
        llm:         Box<dyn LlmCaller>,
        transitions: TransitionTable,
        handlers:    HashMap<&'static str, Box<dyn AgentState>>,
    ) -> Self

    /// Run the agent to completion. Blocks until Done or Error.
    /// Returns: Ok(final_answer) | Err(AgentError)
    pub fn run(&mut self) -> Result<String, AgentError>

    /// Access the full execution trace.
    pub fn trace(&self) -> &Trace

    /// Current state (useful for inspection after run()).
    pub fn current_state(&self) -> &State

    // Public fields (accessible directly):
    pub memory: AgentMemory
    pub tools:  ToolRegistry
    pub llm:    Box<dyn LlmCaller>
}
```

---

## Types

### `State`

```rust
pub enum State {
    Idle, Planning, Acting, Observing, Reflecting, Done, Error,
}

impl State {
    /// Returns true for Done and Error — engine exits immediately.
    pub fn is_terminal(&self) -> bool

    /// Returns the state name as a static str (same as handler map key).
    pub fn as_str(&self) -> &'static str
}
```

### `Event`

```rust
pub enum Event {
    Start,
    LlmToolCall, LlmFinalAnswer, MaxSteps, LowConfidence,
    AnswerTooShort, ToolBlacklisted, FatalError,
    ToolSuccess, ToolFailure,
    Continue, NeedsReflection,
    ReflectDone,
}
```

### `LlmResponse`

Returned by every `LlmCaller::call()`:

```rust
pub enum LlmResponse {
    ToolCall {
        tool:       ToolCall,   // The requested tool
        confidence: f64,        // 0.0–1.0
    },
    FinalAnswer {
        content: String,        // The complete response text
    },
}
```

### `ToolCall`

```rust
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, serde_json::Value>,
}
```

### `HistoryEntry`

```rust
pub struct HistoryEntry {
    pub step:        usize,     // Planning step number this happened in
    pub tool:        ToolCall,
    pub observation: String,    // "SUCCESS: ..." or "ERROR: ..."
    pub success:     bool,
}
```

### `AgentConfig`

```rust
pub struct AgentConfig {
    pub max_steps:             usize,                  // Default: 15
    pub max_retries:           usize,                  // Default: 3
    pub confidence_threshold:  f64,                    // Default: 0.4
    pub reflect_every_n_steps: usize,                  // Default: 5 (0 = disabled)
    pub min_answer_length:     usize,                  // Default: 20
    pub models:                HashMap<String, String>, // Default: empty
}

impl Default for AgentConfig { ... }  // uses the defaults above
```

---

## Memory

### `AgentMemory`

```rust
impl AgentMemory {
    pub fn new(task: impl Into<String>) -> Self

    // Builder-style constructors
    pub fn with_task_type(self, task_type: impl Into<String>) -> Self
    pub fn with_system_prompt(self, prompt: impl Into<String>) -> Self
    pub fn with_config(self, config: AgentConfig) -> Self

    /// Add a tool to the blacklist.
    pub fn blacklist_tool(&mut self, tool_name: impl Into<String>)

    /// Record a trace entry. Called by state handlers.
    pub fn log(&mut self, state: &str, event: &str, data: &str)

    /// Build the [system, history..., task] messages array for LLM calls.
    pub fn build_messages(&self) -> Vec<serde_json::Value>

    // All fields are public — see core-concepts.md for the full list
}
```

---

## Tools

### `ToolRegistry`

```rust
/// ToolFn type alias
pub type ToolFn = Box<dyn Fn(&HashMap<String, Value>) -> Result<String, String> + Send + Sync>;

impl ToolRegistry {
    pub fn new() -> Self
    pub fn register(&mut self, name, description, schema: Value, func: ToolFn)
    pub fn register_tool(&mut self, tool: Tool)  // ergonomic alternative
    pub fn execute(&self, name: &str, args: &HashMap<String, Value>) -> Result<String, String>
    pub fn has(&self, name: &str) -> bool
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn schemas(&self) -> Vec<ToolSchema>
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSchema {
    pub name:         String,
    pub description:  String,
    pub input_schema: serde_json::Value,
}
```

### `Tool` (Builder)

```rust
impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self
    pub fn param(self, name, param_type, description) -> Self      // required parameter
    pub fn param_opt(self, name, param_type, description) -> Self   // optional parameter
    pub fn call<F>(self, f: F) -> Self                             // attach implementation
}

// Usage:
Tool::new("search", "Search the web")
    .param("query", "string", "The search query")
    .param_opt("limit", "integer", "Max results")
    .call(|args| Ok("results".to_string()))
```

---

## LLM Providers

### `LlmCaller` Trait (sync)

```rust
pub trait LlmCaller: Send + Sync {
    fn call(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String>;
}
```

### `AsyncLlmCaller` Trait

```rust
#[async_trait]
pub trait AsyncLlmCaller: Send + Sync {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String>;
}
```

### `LlmCallerExt` (aka `SyncWrapper`)

Wraps any `AsyncLlmCaller` into a `LlmCaller`:

```rust
pub struct SyncWrapper<T: AsyncLlmCaller>(pub T);
pub use SyncWrapper as LlmCallerExt;

// Usage:
let llm: Box<dyn LlmCaller> = Box::new(LlmCallerExt(OpenAiCaller::new()));
```

### `OpenAiCaller`

```rust
impl OpenAiCaller {
    pub fn new() -> Self                                              // reads OPENAI_API_KEY
    pub fn with_base_url(api_base: impl Into<String>, api_key: impl Into<String>) -> Self
}
// Implements AsyncLlmCaller
```

### `AnthropicCaller`

```rust
impl AnthropicCaller {
    pub fn new(api_key: impl Into<String>) -> Self
    pub fn from_env() -> Result<Self, String>                         // reads ANTHROPIC_API_KEY
}
// Implements AsyncLlmCaller
```

### `MockLlmCaller`

```rust
impl MockLlmCaller {
    pub fn new(responses: Vec<LlmResponse>) -> Self
    pub fn call_count(&self) -> usize
    pub fn model_for_call(&self, n: usize) -> Option<String>
}
// Implements LlmCaller directly (no wrapping needed)
```

### `RetryingLlmCaller`

```rust
impl RetryingLlmCaller {
    /// Wrap any LlmCaller with retry. n = max retry attempts.
    pub fn new(inner: Box<dyn LlmCaller>, max_retries: u32) -> Self
}
// Implements LlmCaller. Auth errors (401/403) are never retried.
// Back-off: 1s → 2s → 4s → … cap 30s.
```

---

## States

### `AgentState` Trait

```rust
pub trait AgentState: Send + Sync {
    fn name(&self) -> &'static str;
    fn handle(&self, memory: &mut AgentMemory, tools: &ToolRegistry, llm: &dyn LlmCaller) -> Event;
}
```

### Provided Implementations

| Type | Module | Name string |
|---|---|---|
| `IdleState` | `agentsm::states` | `"Idle"` |
| `PlanningState` | `agentsm::states` | `"Planning"` |
| `ActingState` | `agentsm::states` | `"Acting"` |
| `ObservingState` | `agentsm::states` | `"Observing"` |
| `ReflectingState` | `agentsm::states` | `"Reflecting"` |
| `DoneState` | `agentsm::states` | `"Done"` |
| `ErrorState` | `agentsm::states` | `"Error"` |

---

## Transitions

```rust
pub type TransitionTable = HashMap<(State, Event), State>;

/// Build the default, complete transition table.
pub fn build_transition_table() -> TransitionTable

/// Check if a (state, event) pair is legal.
pub fn is_valid_transition(table: &TransitionTable, state: &State, event: &Event) -> bool
```

---

## Trace

```rust
pub struct TraceEntry {
    pub step:      usize,
    pub state:     String,
    pub event:     String,
    pub data:      String,
    pub timestamp: DateTime<Utc>,
}

impl Trace {
    pub fn new() -> Self
    pub fn record(&mut self, entry: TraceEntry)
    pub fn entries(&self) -> &[TraceEntry]
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn for_state(&self, state: &str) -> Vec<&TraceEntry>  // filter by state name
    pub fn to_json(&self) -> String                            // pretty-printed JSON
    pub fn print(&self)                                        // table to stdout
}
```

---

## Errors

```rust
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Agent failed: {0}")]
    AgentFailed(String),                    // agent reached Error state

    #[error("Invalid transition: {from} + {event} not in transition table")]
    InvalidTransition { from: State, event: Event },

    #[error("No handler registered for state: {0}")]
    NoHandlerForState(String),

    #[error("Safety cap exceeded after {0} iterations")]
    SafetyCapExceeded(usize),              // loop ran > max_steps * 3 iterations

    #[error("LLM caller error: {0}")]
    LlmError(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Build error: {0}")]
    BuildError(String),                    // e.g. missing .llm()
}
```

---

## Crate Root Re-exports

Everything you need is available directly from `agentsm::`:

```rust
use agentsm::{
    AgentBuilder,
    AgentEngine,
    AgentMemory,
    AgentConfig,
    AgentError,
    State,
    Event,
    LlmResponse,
    ToolCall,
    HistoryEntry,
    ToolRegistry,
    ToolFn,
    Tool,               // Tool builder
    LlmCaller,
    LlmCallerExt,
    RetryingLlmCaller,  // retry wrapper
    TraceEntry,
    Trace,
};
```
