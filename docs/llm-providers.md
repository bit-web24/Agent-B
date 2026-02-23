# LLM Providers

`agentsm-rs` supports OpenAI, Anthropic, any OpenAI-compatible API, and custom providers through a simple trait.

---

## Provider Shortcuts (Recommended)

`AgentBuilder` has one-liner methods for the most common providers:

```rust
// OpenAI (reads OPENAI_API_KEY from env if empty string)
AgentBuilder::new("task").openai("sk-...").model("gpt-4o")

// Anthropic / Claude
AgentBuilder::new("task").anthropic("sk-ant-...").model("claude-sonnet-4-6")

// Groq (ultra-fast inference)
AgentBuilder::new("task").groq("gsk_...").model("llama-3.3-70b-versatile")

// Ollama (local, default http://localhost:11434/v1)
AgentBuilder::new("task").ollama("").model("llama3.2")

// Any OpenAI-compatible API (escape hatch)
use agentsm::llm::{OpenAiCaller, LlmCallerExt};
AgentBuilder::new("task")
    .llm(Box::new(LlmCallerExt(OpenAiCaller::with_base_url(
        "https://api.together.xyz/v1",
        std::env::var("TOGETHER_API_KEY").unwrap(),
    ))))
    .model("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
```

---

## Built-in Retry Policy

Transient provider errors (HTTP 429, 503, timeouts, `tool_use_failed`) are common. Enable automatic retry with exponential back-off:

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

// Together AI
let llm = Box::new(LlmCallerExt(
    OpenAiCaller::with_base_url(
        "https://api.together.xyz/v1",
        std::env::var("TOGETHER_API_KEY").unwrap(),
    )
));

// Ollama (local)
let llm = Box::new(LlmCallerExt(
    OpenAiCaller::with_base_url(
        "http://localhost:11434/v1",
        "ollama",  // Ollama ignores the key
    )
));

// Fireworks AI
let llm = Box::new(LlmCallerExt(
    OpenAiCaller::with_base_url(
        "https://api.fireworks.ai/inference/v1",
        std::env::var("FIREWORKS_API_KEY").unwrap(),
    )
));
```

Then pass the appropriate model string in `task_type` routing — or override `PlanningState::select_model()` in a custom state.

### Model Strings for OpenAI-Compatible Providers

| Provider | Example model string |
|---|---|
| OpenAI | `"gpt-4o"`, `"gpt-4o-mini"`, `"o1"` |
| Groq | `"llama-3.3-70b-versatile"`, `"mixtral-8x7b-32768"` |
| Together | `"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"` |
| Ollama | `"llama3.2"`, `"qwen2.5-coder:7b"` |
| Fireworks | `"accounts/fireworks/models/llama-v3p3-70b-instruct"` |

---

## Anthropic Provider

Uses the Anthropic Messages API directly via `reqwest` — no community SDK dependency.

### From Environment Variable

```rust
use agentsm::llm::{AnthropicCaller, LlmCallerExt};

let caller = AnthropicCaller::from_env()
    .map_err(|e| anyhow::anyhow!(e))?;
let llm = Box::new(LlmCallerExt(caller));
```

Reads `ANTHROPIC_API_KEY` from the environment.

### Explicit Key

```rust
let caller = AnthropicCaller::new("sk-ant-api03-...");
let llm = Box::new(LlmCallerExt(caller));
```

### Anthropic Model Strings

Model names are **not hardcoded** in the library — set them via the builder:

```rust
// Single model for all tasks
AnthropicBuilder::new("task")
    .model("claude-opus-4-6")           // highest quality
    .model("claude-sonnet-4-6")         // balanced  
    .model("claude-haiku-4-5-20251001") // fast/cheap

// Different models per task type
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
use agentsm::{LlmResponse, ToolCall};
use std::collections::HashMap;

let mock = MockLlmCaller::new(vec![
    // First call: LLM requests a tool
    LlmResponse::ToolCall {
        tool: ToolCall {
            name: "search".to_string(),
            args: HashMap::from([
                ("query".to_string(), serde_json::json!("Rust programming")),
            ]),
        },
        confidence: 0.95,
    },
    // Second call: LLM produces the final answer
    LlmResponse::FinalAnswer {
        content: "Rust is a systems programming language focused on safety and performance.".to_string(),
    },
]);

// Inspect after run
println!("LLM was called {} times", mock.call_count());
println!("First call used model: {:?}", mock.model_for_call(0));
```

Since `AgentBuilder::llm()` takes ownership, if you need to inspect `call_count` after running, track invocations via the trace instead:

```rust
let planning_steps = engine.trace()
    .entries()
    .iter()
    .filter(|e| e.state == "Planning" && e.event == "STEP_START")
    .count();
```

---

## Custom LLM Provider

Implement `AsyncLlmCaller` for your provider:

```rust
use agentsm::llm::{AsyncLlmCaller, LlmCallerExt};
use agentsm::memory::AgentMemory;
use agentsm::tools::ToolRegistry;
use agentsm::types::{LlmResponse, LlmStreamChunk};
use async_trait::async_trait;
use futures::stream::BoxStream;

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

```rust
pub struct MyCustomCaller {
    client: reqwest::Client,
    api_key: String,
}

#[async_trait]
impl AsyncLlmCaller for MyCustomCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        let messages = memory.build_messages();       // Vec<serde_json::Value>
        let schemas  = tools.schemas();               // Vec<ToolSchema>

        // ... call your API ...

        Ok(LlmResponse::FinalAnswer {
            content: "response from my API".to_string(),
        })
    }
}

// Wrap for use with the engine
let llm = Box::new(LlmCallerExt(MyCustomCaller {
    client:  reqwest::Client::new(),
    api_key: "...".to_string(),
}));
```

### LlmCaller Contract

Your implementation **must**:
- Call `memory.build_messages()` to get the message history
- Call `tools.schemas()` to get the tool definitions
- Return `Ok(LlmResponse::ToolCall { ... })` when the model wants to use a tool
- Return `Ok(LlmResponse::FinalAnswer { ... })` for a final response
- Return `Err(String)` **only** for unrecoverable failures (network down, auth failed)
- Never panic

---

## The `LlmResponse` Type

```rust
pub enum LlmResponse {
    ToolCall {
        tool:       ToolCall,   // name + args (HashMap<String, serde_json::Value>)
        confidence: f64,        // 0.0–1.0; use 1.0 if your API doesn't provide this
    },
    FinalAnswer {
        content: String,        // The complete answer text
    },
}
```

The `confidence` field is used by `PlanningState` to decide whether to trigger reflection (`LowConfidence` event). If your provider doesn't return a confidence score, always pass `1.0` — this disables low-confidence reflection for that caller.
