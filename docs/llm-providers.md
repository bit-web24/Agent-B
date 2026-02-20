# LLM Providers

`agentsm-rs` supports OpenAI, Anthropic, any OpenAI-compatible API, and custom providers through a simple trait.

---

## Trait Architecture

```
LlmCaller (sync, used by engine)
    │
    ├── Implemented directly by:
    │     └── MockLlmCaller (for testing)
    │
    └── Implemented via SyncWrapper (LlmCallerExt) over:
          └── AsyncLlmCaller (async)
                ├── OpenAiCaller
                └── AnthropicCaller
```

The engine uses `&dyn LlmCaller` (sync). Async callers are wrapped using:

```rust
let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));
//                 ^^^^^^^^^^^^ wraps AsyncLlmCaller → LlmCaller
```

---

## OpenAI Provider

### Standard OpenAI

Uses `OPENAI_API_KEY` environment variable automatically:

```rust
use agentsm::llm::{OpenAiCaller, LlmCallerExt};

let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));
```

### Custom Base URL (OpenAI-Compatible APIs)

Works with any API that follows the OpenAI chat completions format:

```rust
// Groq
let llm = Box::new(LlmCallerExt(
    OpenAiCaller::with_base_url(
        "https://api.groq.com/openai/v1",
        std::env::var("GROQ_API_KEY").unwrap(),
    )
));

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

### Claude Model Strings

Pass via `task_type` routing (see [Core Concepts](./core-concepts.md#model-selection-by-task_type)) or use a custom `PlanningState`:

| Tier | Model string |
|---|---|
| Research (highest quality) | `"claude-opus-4-6"` |
| Default (balanced) | `"claude-sonnet-4-6"` |
| Calculation (fast/cheap) | `"claude-haiku-4-5-20251001"` |

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
use agentsm::types::LlmResponse;
use async_trait::async_trait;

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
