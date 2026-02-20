# Examples

All examples are in the `examples/` directory and compile with `cargo build --examples`. Running them requires real API keys.

```bash
# Build all examples (no keys needed)
cargo build --examples

# Run (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... cargo run --example basic_agent
OPENAI_API_KEY=sk-... cargo run --example multi_tool_agent

# Run Anthropic (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_agent

# With verbose logging
RUST_LOG=debug OPENAI_API_KEY=sk-... cargo run --example basic_agent
```

---

## `basic_agent.rs` — Minimal OpenAI Agent

**[Source](../examples/basic_agent.rs)**

The simplest complete agent. Demonstrates:
- `AgentBuilder` fluent API
- Single tool registration
- `LlmCallerExt` wrapping
- Running to completion and printing the trace

```rust
use agentsm::{AgentBuilder, AgentConfig};
use agentsm::llm::{OpenAiCaller, LlmCallerExt};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));

    let mut engine = AgentBuilder::new(
            "What is the capital of France and what is its population?"
        )
        .task_type("research")          // → gpt-4o tier
        .system_prompt("You are a helpful research assistant. \
                        Use the search tool to find information.")
        .llm(llm)
        .config(AgentConfig {
            max_steps: 10,
            ..Default::default()
        })
        .tool(
            "search",
            "Search the web for current information.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" }
                },
                "required": ["query"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                // In production: call a real search API (Serper, Tavily, Brave Search)
                Ok(format!("Results for '{}': Paris is the capital of France, \
                            population ~2.1M city / ~12M metro.", query))
            }),
        )
        .build()?;

    match engine.run() {
        Ok(answer) => {
            println!("ANSWER:\n{}\n", answer);
            engine.trace().print();
        }
        Err(e) => eprintln!("Failed: {}", e),
    }
    Ok(())
}
```

**Typical execution trace:**

```
step   state          event                        data
────────────────────────────────────────────────────────────────────────────────
0      Idle           AGENT_STARTED                task='What is the capital...'
1      Planning       STEP_START                   step=1/10
1      Planning       LLM_TOOL_CALL                tool='search' confidence=1.00
1      Acting         TOOL_EXECUTE                 tool='search' args={"query": "..."}
1      Acting         TOOL_SUCCESS                 Results for '...'
1      Observing      HISTORY_COMMIT               step=1 success=true len=1
2      Planning       STEP_START                   step=2/10
2      Planning       LLM_FINAL_ANSWER             Paris is the capital of France...
```

---

## `multi_tool_agent.rs` — Multiple Tools with Blacklisting

**[Source](../examples/multi_tool_agent.rs)**

Demonstrates:
- `task_type("calculation")` → cheaper model tier
- Multiple tool registration (calculator, weather)
- A registered-but-blacklisted tool (search)
- Custom `AgentConfig`
- A minimal expression evaluator

```rust
AgentBuilder::new("Calculate 137 * 48 and get the weather in London")
    .task_type("calculation")       // → gpt-4o-mini tier
    .system_prompt("You are a precise assistant. Use calculator for math, \
                    weather for weather. Never guess.")
    .llm(llm)
    .config(AgentConfig {
        max_steps: 8,
        max_retries: 2,
        reflect_every_n_steps: 4,
        ..Default::default()
    })
    // Tool 1: Calculator
    .tool("calculator", "Evaluate math expressions...", calc_schema, calc_fn)
    // Tool 2: Weather
    .tool("weather", "Get current weather for a city...", weather_schema, weather_fn)
    // Tool 3: Registered but forbidden
    .tool("search", "Search the web...", search_schema, search_fn)
    .blacklist_tool("search")   // LLM will not be allowed to use this
    .build()?
```

**What happens when the LLM tries the blacklisted tool:**

```
Planning  TOOL_BLACKLISTED  Requested blacklisted tool: search
Planning  STEP_START        step=2/8      ← Planning retries, gets another chance
```

---

## `anthropic_agent.rs` — Claude via Anthropic API

**[Source](../examples/anthropic_agent.rs)**

Demonstrates:
- `AnthropicCaller::from_env()` — reads `ANTHROPIC_API_KEY`
- `task_type("research")` → `claude-opus-4-6` (highest quality)
- A knowledge base lookup tool with structured responses

```rust
use agentsm::llm::{AnthropicCaller, LlmCallerExt};

let anthropic = AnthropicCaller::from_env()
    .map_err(|e| anyhow::anyhow!("{}", e))?;
let llm = Box::new(LlmCallerExt(anthropic));

AgentBuilder::new("What are the key design principles of Rust?")
    .task_type("research")
    .system_prompt("You are an expert systems programmer. \
                    Use the knowledge_base tool for technical information.")
    .llm(llm)
    .max_steps(8)
    .tool("knowledge_base", "Retrieve technical documentation...", schema, kb_fn)
    .build()?
```

---

## Building Your Own Agent from Scratch

Here's a template for a production-ready agent:

```rust
use agentsm::{AgentBuilder, AgentConfig};
use agentsm::llm::{OpenAiCaller, LlmCallerExt};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // ── Configure ─────────────────────────────────────────────────────────
    let task = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the meaning of life?".to_string());

    let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));

    let config = AgentConfig {
        max_steps:             15,
        max_retries:           3,
        confidence_threshold:  0.4,
        reflect_every_n_steps: 5,
        min_answer_length:     30,
    };

    // ── Build ──────────────────────────────────────────────────────────────
    let mut engine = AgentBuilder::new(task)
        .task_type("research")
        .system_prompt("You are a helpful assistant. Be thorough and accurate.")
        .llm(llm)
        .config(config)
        .tool(
            "search",
            "Search the web for current information about any topic.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for" }
                },
                "required": ["query"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                // Replace with a real search API call:
                // - Tavily: https://tavily.com
                // - Serper: https://serper.dev
                // - Brave Search: https://brave.com/search/api/
                let query = args["query"].as_str().unwrap_or("");
                call_search_api(query)
            }),
        )
        .build()?;

    // ── Run ────────────────────────────────────────────────────────────────
    match engine.run() {
        Ok(answer) => {
            println!("\n=== ANSWER ===\n{}\n", answer);
            println!("=== TRACE ===");
            engine.trace().print();
            println!("\nCompleted in {} steps.", engine.memory.step);
        }
        Err(e) => {
            eprintln!("Agent failed: {}", e);
            eprintln!("Error in memory: {:?}", engine.memory.error);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn call_search_api(query: &str) -> Result<String, String> {
    // Placeholder — integrate your preferred search API here
    Ok(format!("Mock results for: {}", query))
}
```
