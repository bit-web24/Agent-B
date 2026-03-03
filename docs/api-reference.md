# API Reference

Complete public API surface of `agentsm-rs`. All types are in the `agentsm` crate root unless noted.

---

## Builder

### `AgentBuilder`

```rust
impl AgentBuilder {
    pub fn new(task: impl Into<String>) -> Self

    // ── Core ──────────────────────────────────────────────────────────────
    pub fn task_type(self, t: impl Into<String>) -> Self
    pub fn system_prompt(self, p: impl Into<String>) -> Self
    pub fn llm(self, llm: Arc<dyn AsyncLlmCaller>) -> Self
    pub fn model(self, name: impl Into<String>) -> Self
    pub fn model_for(self, task_type: impl Into<String>, model: impl Into<String>) -> Self
    pub fn max_steps(self, n: usize) -> Self
    pub fn config(self, config: AgentConfig) -> Self
    pub fn retry_on_error(self, n: u32) -> Self

    // ── Provider shortcuts ────────────────────────────────────────────────
    pub fn openai(self, api_key: impl Into<String>) -> Self
    pub fn anthropic(self, api_key: impl Into<String>) -> Self
    pub fn ollama(self, base_url: impl Into<String>) -> Self
    pub fn groq(self, api_key: impl Into<String>) -> Self

    // ── Structured Output ─────────────────────────────────────────────────
    pub fn output_schema(self, name: impl Into<String>, schema: serde_json::Value) -> Self
    pub fn output_schema_with_desc(self, name: impl Into<String>,
        description: impl Into<String>, schema: serde_json::Value) -> Self

    // ── Tools ─────────────────────────────────────────────────────────────
    pub fn add_tool(self, tool: Tool) -> Self
    pub fn tool(self, name, description, schema, func) -> Self
    pub fn blacklist_tool(self, name: impl Into<String>) -> Self
    pub fn parallel_tools(self, enabled: bool) -> Self

    // ── Human-in-the-Loop ─────────────────────────────────────────────────
    pub fn approval_policy(self, policy: ApprovalPolicy) -> Self
    pub fn on_approval<F>(self, f: F) -> Self

    // ── Persistence ───────────────────────────────────────────────────────
    pub fn checkpoint_store(self, store: Arc<dyn CheckpointStore>) -> Self
    pub fn session_id(self, id: impl Into<String>) -> Self
    pub async fn resume(self, session_id: impl Into<String>) -> Self

    // ── Budgeting ─────────────────────────────────────────────────────────
    pub fn max_tokens(self, n: usize) -> Self
    pub fn token_budget(self, budget: TokenBudget) -> Self

    // ── Sub-Agents ────────────────────────────────────────────────────────
    pub fn as_tool(self, name: impl Into<String>, description: impl Into<String>) -> Tool
    pub fn add_subagent(self, name, desc, builder: AgentBuilder) -> Self

    // ── MCP ───────────────────────────────────────────────────────────────
    pub fn mcp_server(self, command: &str, args: &[String]) -> Self

    // ── Custom State Graphs ───────────────────────────────────────────────
    pub fn state(self, name: &'static str, handler: Arc<dyn AgentState>) -> Self
    pub fn transition(self, from: &str, event: &str, to: &str) -> Self
    pub fn terminal_state(self, name: &str) -> Self

    // ── Build ─────────────────────────────────────────────────────────────
    pub fn build(self) -> Result<AgentEngine, AgentError>
}
```

---

## Engine

### `AgentEngine`

```rust
impl AgentEngine {
    pub async fn run(&mut self) -> Result<String, AgentError>
    pub fn run_streaming(&mut self) -> BoxStream<'_, AgentOutput>
    pub fn trace(&self) -> &Trace
    pub fn current_state(&self) -> &State
    pub memory: AgentMemory       // public field
}
```

---

## Types

### `State`

String-based newtype with built-in constructors:

```rust
State::idle()       // "Idle"
State::planning()   // "Planning"
State::acting()     // "Acting"
State::done()       // "Done"
State::error()      // "Error"
State::new("Custom") // any custom state

state.is_terminal() -> bool   // true for Done, Error, and custom terminal states
```

### `Event`

String-based newtype with built-in constructors:

```rust
Event::start()                    Event::tool_success()
Event::llm_tool_call()            Event::tool_failure()
Event::llm_parallel_tool_calls()  Event::continue_event()
Event::llm_final_answer()         Event::needs_reflection()
Event::max_steps()                Event::reflect_done()
Event::low_confidence()           Event::human_approval_required()
Event::answer_too_short()         Event::human_approved()
Event::tool_blacklisted()         Event::human_rejected()
Event::fatal_error()              Event::human_modified()
Event::new("Custom")              // any custom event
```

### `LlmResponse`

Returned by every `AsyncLlmCaller::call_async()`:

```rust
pub enum LlmResponse {
    ToolCall {
        tool:       ToolCall,
        confidence: f64,
        usage:      Option<TokenUsage>,
    },
    ParallelToolCalls {
        tools:      Vec<ToolCall>,
        confidence: f64,
        usage:      Option<TokenUsage>,
    },
    FinalAnswer {
        content:    String,
        usage:      Option<TokenUsage>,
    },
    Structured {
        data:       serde_json::Value,
        usage:      Option<TokenUsage>,
    },
}
```

### `OutputSchema`

```rust
pub struct OutputSchema {
    pub name:        String,
    pub description: Option<String>,
    pub schema:      serde_json::Value,
}
```

### `ToolCall`

```rust
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, serde_json::Value>,
    pub id:   Option<String>,
}
```

### `AgentConfig`

```rust
pub struct AgentConfig {
    pub max_steps:             usize,                    // default: 15
    pub max_retries:           usize,                    // default: 3
    pub confidence_threshold:  f64,                      // default: 0.4
    pub reflect_every_n_steps: usize,                    // default: 5
    pub min_answer_length:     usize,                    // default: 5
    pub parallel_tools:        bool,                     // default: true
    pub models:                HashMap<String, String>,  // default: empty
    pub output_schema:         Option<OutputSchema>,     // default: None
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

---

## LLM Callers

### `AsyncLlmCaller` Trait

```rust
#[async_trait]
pub trait AsyncLlmCaller: Send + Sync {
    async fn call_async(
        &self,
        memory:    &AgentMemory,
        tools:     &ToolRegistry,
        model:     &str,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Result<LlmResponse, String>;

    fn call_stream_async<'a>(
        &'a self,
        memory:    &'a AgentMemory,
        tools:     &'a ToolRegistry,
        model:     &'a str,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> BoxStream<'a, Result<LlmStreamChunk, String>>;
}
```

### `OpenAiCaller`

```rust
impl OpenAiCaller {
    pub fn new() -> Self                              // reads OPENAI_API_KEY
    pub fn with_base_url(base: impl Into<String>, key: impl Into<String>) -> Self
}
```

### `AnthropicCaller`

```rust
impl AnthropicCaller {
    pub fn new(api_key: impl Into<String>) -> Self
    pub fn from_env() -> Result<Self, String>        // reads ANTHROPIC_API_KEY
}
```

### `MockLlmCaller`

```rust
impl MockLlmCaller {
    pub fn new(responses: Vec<LlmResponse>) -> Self
    pub fn call_count(&self) -> usize
    pub fn model_for_call(&self, n: usize) -> Option<String>
}
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
        tools:     &Arc<ToolRegistry>,
        llm:       &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event;
}
```

### Built-in States

| Type | Name string |
|---|---|
| `IdleState` | `"Idle"` |
| `PlanningState` | `"Planning"` |
| `ActingState` | `"Acting"` |
| `ParallelActingState` | `"ParallelActing"` |
| `WaitingForHumanState` | `"WaitingForHuman"` |
| `ObservingState` | `"Observing"` |
| `ReflectingState` | `"Reflecting"` |
| `DoneState` | `"Done"` |
| `ErrorState` | `"Error"` |

---

## Errors

```rust
#[derive(Debug, Error)]
pub enum AgentError {
    AgentFailed(String),
    InvalidTransition { from: State, event: Event },
    NoHandlerForState(String),
    SafetyCapExceeded(usize),
    LlmError(String),
    ToolError(String),
    MemoryError(String),
    BuildError(String),
}
```

---

## Crate Root Re-exports

```rust
use agentsm::{
    AgentBuilder, AgentEngine, AgentConfig, AgentError,
    State, Event,
    LlmResponse, ToolCall, ToolResult, HistoryEntry,
    ToolRegistry, Tool,
    AgentOutput, LlmStreamChunk, OutputSchema,
    TokenBudget, TokenUsage,
};
```
