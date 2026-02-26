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

    /// Set the LLM caller explicitly.
    pub fn llm(self, llm: Arc<dyn AsyncLlmCaller>) -> Self

    // ── Feature: Parallelization ─────────────────────────────────────────

    /// Enable/disable parallel tool execution (default: true).
    pub fn parallel_tools(self, enabled: bool) -> Self

    // ── Feature: Human-in-the-Loop ──────────────────────────────────────

    /// Set the approval policy for high-risk tools.
    pub fn approval_policy(self, policy: ApprovalPolicy) -> Self

    /// Set a callback for handling approval requests.
    pub fn on_approval<F>(self, f: F) -> Self
    where F: Fn(HumanApprovalRequest) -> Result<ApprovalAction, String> + Send + Sync + 'static

    // ── Feature: Persistence & Checkpointing ─────────────────────────────

    /// Set the persistent storage for checkpoints.
    pub fn checkpoint_store(self, store: Arc<dyn CheckpointStore>) -> Self

    /// Set a unique session ID for this run.
    pub fn session_id(self, id: impl Into<String>) -> Self

    /// Resume a run from the last checkpoint in the store.
    pub fn resume(self, session_id: impl Into<String>) -> Self

    // ── Feature: Token Budgeting ─────────────────────────────────────────

    /// Set a simple total token limit for the session.
    pub fn max_tokens(self, n: usize) -> Self

    /// Set complex resource limits.
    pub fn token_budget(self, budget: TokenBudget) -> Self

    // ── Feature: Sub-Agents ──────────────────────────────────────────────

    /// Convert this builder's configuration into a tool.
    pub fn as_tool(self, name: impl Into<String>, description: impl Into<String>) -> Tool

    /// Add a sub-agent as a tool.
    pub fn add_subagent(self, tool: Tool) -> Self

    // ── Builder completion ───────────────────────────────────────────────

    pub fn build(self) -> Result<AgentEngine, AgentError>
}
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

    /// Run the agent to completion.
    /// Returns: Ok(final_answer) | Err(AgentError)
    pub async fn run(&mut self) -> Result<String, AgentError>

    /// Run the agent with real-time streaming output.
    pub fn run_streaming(&mut self) -> BoxStream<'_, AgentOutput>

    /// Access the full execution trace.
    pub fn trace(&self) -> &Trace

    // ...
    pub llm:    Box<dyn AsyncLlmCaller>
}
}
```

---

## Types

### `State`

```rust
pub enum State {
    Idle, Planning, Acting, ParallelActing, WaitingForHuman, Observing, Reflecting, Done, Error,
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
    LlmToolCall, LlmParallelToolCalls, LlmFinalAnswer, MaxSteps, LowConfidence,
    AnswerTooShort, ToolBlacklisted, FatalError,
    HumanApprovalRequired, HumanApproved, HumanRejected, HumanModified,
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

### `AgentOutput` (Streaming)

```rust
pub enum AgentOutput {
    StateStarted(State),
    LlmToken(String),
    ToolCallStarted { name: String, args: HashMap<String, Value> },
    ToolCallFinished { name: String, result: String, success: bool },
    ToolCallDelta { name: Option<String>, args_json: String },
    Action(String),
    FinalAnswer(String),
    Error(String),
}
```

### `LlmStreamChunk`

```rust
pub enum LlmStreamChunk {
    Content(String),
    ToolCallDelta { name: Option<String>, args_json: String },
    Done(LlmResponse),
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
    pub max_steps:             usize,
    pub max_retries:           usize,
    pub confidence_threshold:  f64,
    pub reflect_every_n_steps: usize,
    pub min_answer_length:     usize,
    pub parallel_tools:        bool,
    pub models:                HashMap<String, String>,
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
pub type ToolFn = Arc<dyn Fn(&HashMap<String, Value>) -> Result<String, String> + Send + Sync>;

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

    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
    ) -> BoxStream<'a, Result<LlmStreamChunk, String>>;
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
    ToolResult,
    HistoryEntry,
    ToolRegistry,
    ToolFn,
    Tool,
    LlmCaller,
    LlmCallerExt,
    RetryingLlmCaller,
    TraceEntry,
    Trace,
    TokenBudget,
    TokenUsage,
    ApprovalPolicy,
    ApprovalAction,
    HumanApprovalRequest,
};
```
