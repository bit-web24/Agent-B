# LLM Providers

`agentsm-rs` supports OpenAI, Anthropic, any OpenAI-compatible API, and custom providers through a simple trait.

---

## Provider Shortcuts (Recommended)

`AgentBuilder` has one-liner methods for the most common providers:

```rust
// OpenAI (reads OPENAI_API_KEY from env if empty string)
AgentBuilder::new("task").openai("").model("gpt-4o")

// Anthropic / Claude
AgentBuilder::new("task").anthropic("").model("claude-sonnet-4-6")

// Groq (ultra-fast inference)
AgentBuilder::new("task").groq("gsk_...").model("llama-3.3-70b-versatile")

// Ollama (local, default http://localhost:11434/v1)
AgentBuilder::new("task").ollama("").model("llama3.2")

// Any OpenAI-compatible API (escape hatch)
use agentsm::llm::OpenAiCaller;
AgentBuilder::new("task")
    .llm(Arc::new(OpenAiCaller::with_base_url(
        "https://api.together.xyz/v1",
        std::env::var("TOGETHER_API_KEY").unwrap(),
    )))
    .model("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
```

---

## Built-in Retry Policy

Transient provider errors (HTTP 429, 503, timeouts) are common. Enable automatic retry with exponential back-off:

```rust
AgentBuilder::new("task")
    .groq("gsk_...")
    .model("llama-3.3-70b-versatile")
    .retry_on_error(3)   // 1s → 2s → 4s back-off, up to 3 retries
```

**Retry rules:**
- Any `Err(String)` from the LLM caller is retried
- **Auth errors are never retried** (401, 403, "unauthorized", "invalid api key")
- Back-off: 1s → 2s → 4s → … capped at 30s
- After all retries exhausted: `Err("LLM failed after N retries — last error: ...")`

---

## Anthropic Provider

Uses the Anthropic Messages API directly via `reqwest` — no community SDK dependency.

```rust
// From environment variable
AgentBuilder::new("task").anthropic("").model("claude-sonnet-4-6")

// Explicit key
AgentBuilder::new("task").anthropic("sk-ant-api03-...").model("claude-sonnet-4-6")
```

### Anthropic Model Strings

```rust
AgentBuilder::new("task")
    .model("claude-sonnet-4-6")                             // default
    .model_for("research",    "claude-opus-4-6")            // best for research
    .model_for("calculation", "claude-haiku-4-5-20251001")  // cheapest for math
```

---

## Mock Provider (for Testing)

`MockLlmCaller` returns pre-programmed responses in sequence. No network calls.

```rust
use agentsm::llm::MockLlmCaller;
use agentsm::LlmResponse;

let mock = Arc::new(MockLlmCaller::new(vec![
    LlmResponse::ToolCall {
        tool: ToolCall { name: "search".into(), args: HashMap::new(), id: None },
        confidence: 0.95,
        usage: None,
    },
    LlmResponse::FinalAnswer {
        content: "Rust is a systems programming language.".into(),
        usage: None,
    },
]));

let engine = AgentBuilder::new("task").llm(mock).build()?;
```

---

## Custom LLM Provider

Implement `AsyncLlmCaller` for your provider:

```rust
#[async_trait]
pub trait AsyncLlmCaller: Send + Sync {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Result<LlmResponse, String>;

    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> BoxStream<'a, Result<LlmStreamChunk, String>>;
}
```

---

## The `LlmResponse` Type

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
        content: String,
        usage:   Option<TokenUsage>,
    },
    Structured {
        data:  serde_json::Value,
        usage: Option<TokenUsage>,
    },
}
```

The `confidence` field is used by `PlanningState` to decide whether to trigger reflection. Most built-in callers return `1.0`. The `Structured` variant is returned when `output_schema` is configured.
