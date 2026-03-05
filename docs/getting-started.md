# Getting Started

## Installation

Add `agent_b` to your `Cargo.toml`:

```toml
[dependencies]
Agent-B = { path = "../Agent-B" }  # local path
# OR once published:
# agent_b = "0.1"

tokio = { version = "1", features = ["full"] }
serde_json = "1"
anyhow = "1"
```

> **Note:** `Agent-B` uses `rustls` for TLS — no OpenSSL installation needed.

---

## Your First Agent in 5 Minutes

```rust
use agent_b::{AgentBuilder, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut engine = AgentBuilder::new("What is the capital of Japan?")
        .openai("")             // reads OPENAI_API_KEY from env
        .model("gpt-4o")
        .retry_on_error(2)
        .add_tool(
            Tool::new("search", "Search the web for factual information.")
                .param("query", "string", "The search query")
                .call(|args| {
                    let query = args["query"].as_str().unwrap_or("");
                    Ok(format!("Results for '{}': Tokyo is the capital of Japan.", query))
                })
        )
        .build()?;

    match engine.run().await {
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

---

## Next Steps

- **[Architecture](./architecture.md)** — Understand the state machine
- **[LLM Providers](./llm-providers.md)** — Use Anthropic, Groq, Ollama
- **[Tool System](./tool-system.md)** — Register tools with rich schemas
- **[Configuration](./configuration.md)** — Tune behavior and structured output
- **[Examples](./examples.md)** — Full annotated examples
