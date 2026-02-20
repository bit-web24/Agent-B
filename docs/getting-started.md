# Getting Started

## Installation

Add `agentsm` to your `Cargo.toml`:

```toml
[dependencies]
agentsm-rs = { path = "../agentsm-rs" }  # local path
# OR once published:
# agentsm = "0.1"

tokio = { version = "1", features = ["full"] }
serde_json = "1"
anyhow = "1"
```

> **Note:** `agentsm-rs` uses `rustls` for TLS — no OpenSSL installation needed.

---

## Your First Agent in 5 Minutes

The complete minimal example — an agent that uses a search tool to answer a question:

```rust
use agentsm::{AgentBuilder, Tool};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build the agent — one line per concern
    let mut engine = AgentBuilder::new("What is the capital of Japan?")
        .openai("sk-...")       // or .groq("gsk_...") or .ollama("") or .anthropic("sk-ant-...")
        .model("gpt-4o")
        .retry_on_error(2)      // auto-retry on transient LLM failures
        .add_tool(
            Tool::new("search", "Search the web for factual information.")
                .param("query", "string", "The search query")
                .call(|args| {
                    let query = args["query"].as_str().unwrap_or("");
                    Ok(format!("Search results for '{}': Tokyo is the capital of Japan.", query))
                })
        )
        .build()?;

    // Run to completion
    match engine.run() {
        Ok(answer) => println!("Answer: {}", answer),
        Err(e)     => eprintln!("Failed: {}", e),
    }
    Ok(())
}
```

Run it:

```bash
OPENAI_API_KEY=sk-... cargo run
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For OpenAI/compatible providers | API key from platform.openai.com |
| `ANTHROPIC_API_KEY` | For Anthropic/Claude | API key from console.anthropic.com |
| `RUST_LOG` | Optional | Log level: `error`, `warn`, `info`, `debug`, `trace` |

Enable logging in your binary:

```rust
tracing_subscriber::fmt::init();
```

Then run:

```bash
RUST_LOG=info OPENAI_API_KEY=sk-... cargo run
RUST_LOG=debug OPENAI_API_KEY=sk-... cargo run  # verbose — shows every LLM call
```

---

## Project Layout

```
your-project/
├── Cargo.toml
└── src/
    └── main.rs   ← your agent lives here
```

The simplest binary looks like this:

```rust
use agentsm::AgentBuilder;
use agentsm::llm::{OpenAiCaller, LlmCallerExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut engine = AgentBuilder::new("Your task here")
        .llm(Box::new(LlmCallerExt(OpenAiCaller::new())))
        .build()?;

    match engine.run() {
        Ok(answer) => println!("{}", answer),
        Err(e)     => eprintln!("{}", e),
    }
    Ok(())
}
```

---

## Next Steps

- **[Architecture](./architecture.md)** — Understand how the state machine works
- **[LLM Providers](./llm-providers.md)** — Use Anthropic, Groq, Ollama, or any OpenAI-compatible API
- **[Tool System](./tool-system.md)** — Register tools with rich schemas
- **[Examples](./examples.md)** — Full annotated examples
