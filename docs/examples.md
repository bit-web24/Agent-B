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

# Streaming
OPENAI_API_KEY=sk-... cargo run --example streaming_agent
```

---

## `basic_agent.rs` — Minimal Agent

Uses the `AgentBuilder` fluent API with `.openai("")` (reads `OPENAI_API_KEY` from env):

```rust
let mut engine = AgentBuilder::new("What is the capital of France?")
    .openai("")
    .model("gpt-4o")
    .system_prompt("You are a helpful assistant. Use tools when needed.")
    .max_steps(10)
    .add_tool(
        Tool::new("search", "Search the web for information")
            .param("query", "string", "The search query")
            .call(|args| {
                let q = args["query"].as_str().unwrap_or("");
                Ok(format!("Results for '{}': Paris is the capital of France.", q))
            })
    )
    .build()?;

match engine.run().await {
    Ok(answer) => {
        println!("ANSWER:\n{}\n", answer);
        engine.trace().print();
    }
    Err(e) => eprintln!("Failed: {}", e),
}
```

---

## `anthropic_agent.rs` — Claude via Anthropic

```rust
let mut engine = AgentBuilder::new("What are Rust's key design principles?")
    .anthropic("")
    .model("claude-sonnet-4-6")
    .system_prompt("You are an expert systems programmer.")
    .max_steps(8)
    .add_tool(
        Tool::new("knowledge_base", "Look up technical documentation")
            .param("topic", "string", "Topic to look up")
            .call(|args| {
                let topic = args["topic"].as_str().unwrap_or("");
                Ok(format!("Info about {}: ...", topic))
            })
    )
    .build()?;
```

---

## `streaming_agent.rs` — Real-time Streaming

```rust
use futures::StreamExt;

let mut stream = engine.run_streaming();

while let Some(output) = stream.next().await {
    match output {
        AgentOutput::LlmToken(t)     => print!("{}", t),
        AgentOutput::FinalAnswer(a)  => println!("\n\nAnswer: {}", a),
        AgentOutput::Error(e)        => eprintln!("Error: {}", e),
        AgentOutput::StateStarted(s) => println!("\n[{}]", s),
        _ => {}
    }
}
```

---

## Structured Output Example

```rust
use serde_json::json;

let mut engine = AgentBuilder::new("Extract info about the Rust programming language")
    .openai("")
    .model("gpt-4o")
    .output_schema("language_info", json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "year": { "type": "integer" },
            "paradigms": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["name", "year", "paradigms"]
    }))
    .build()?;

let answer = engine.run().await?;
let data: serde_json::Value = serde_json::from_str(&answer)?;
println!("Language: {}", data["name"]);
println!("Year: {}", data["year"]);
```

---

## Sub-Agent Example

```rust
let researcher = AgentBuilder::new("research")
    .openai("")
    .add_tool(search_tool);

let mut engine = AgentBuilder::new("Research and summarize Rust's history")
    .openai("")
    .add_subagent("researcher", "A research specialist", researcher)
    .build()?;

engine.run().await?;
```
