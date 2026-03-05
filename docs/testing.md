# Testing Guide

`Agent-B` is designed for testability from the ground up. The `MockLlmCaller` enables full agent runs without any network calls.

---

## Running Tests

```bash
cargo test                                    # All tests
cargo test --test integration_tests           # Only integration tests
cargo test test_structured_output             # A specific test
cargo test -- --nocapture                     # With output printed
```

---

## The MockLlmCaller

`MockLlmCaller` returns pre-programmed `LlmResponse` values in sequence. Wrap in `Arc` for use with the builder:

```rust
use agent_b::llm::MockLlmCaller;
use agent_b::{LlmResponse, ToolCall};
use std::sync::Arc;
use std::collections::HashMap;

let mock = Arc::new(MockLlmCaller::new(vec![
    LlmResponse::ToolCall {
        tool: ToolCall { name: "search".into(), args: HashMap::new(), id: None },
        confidence: 1.0,
        usage: None,
    },
    LlmResponse::FinalAnswer {
        content: "Rust is a systems programming language.".into(),
        usage: None,
    },
]));

let mut engine = AgentBuilder::new("What is Rust?")
    .llm(mock)
    .add_tool(
        Tool::new("search", "search tool")
            .param("query", "string", "search query")
            .call(|_| Ok("Rust is a systems language...".to_string()))
    )
    .build()
    .expect("should succeed");

let result = engine.run().await;
assert!(result.is_ok());
```

---

## Testing Structured Output

```rust
use serde_json::json;

#[tokio::test]
async fn test_structured_output() {
    let data = json!({"name": "Rust", "year": 2010});

    let mock = Arc::new(MockLlmCaller::new(vec![
        LlmResponse::Structured { data: data.clone(), usage: None },
    ]));

    let mut engine = AgentBuilder::new("Extract language info")
        .llm(mock)
        .output_schema("language_info", json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "year": { "type": "integer" }
            }
        }))
        .build()
        .expect("should succeed");

    let result = engine.run().await;
    assert!(result.is_ok());

    let answer = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&answer).unwrap();
    assert_eq!(parsed["name"], "Rust");
    assert_eq!(parsed["year"], 2010);
}
```

---

## Test Helpers

```rust
fn tool_call_resp(name: &str) -> LlmResponse {
    LlmResponse::ToolCall {
        tool: ToolCall { name: name.into(), args: HashMap::new(), id: None },
        confidence: 1.0,
        usage: None,
    }
}

fn final_answer_resp(content: &str) -> LlmResponse {
    LlmResponse::FinalAnswer { content: content.into(), usage: None }
}
```

---

## Testing Full Agent Runs

```rust
#[tokio::test]
async fn test_agent_two_tool_calls() {
    let mock = Arc::new(MockLlmCaller::new(vec![
        tool_call_resp("search"),
        tool_call_resp("search"),
        final_answer_resp("The complete answer is 42."),
    ]));

    let mut engine = AgentBuilder::new("test task")
        .llm(mock)
        .add_tool(Tool::new("search", "search tool")
            .param("q", "string", "query")
            .call(|_| Ok("result".into())))
        .build().unwrap();

    let result = engine.run().await;
    assert!(result.is_ok());
    assert_eq!(engine.memory.history.len(), 2);
}
```

---

## What to Test

- [x] **Happy path:** tool call → success → final answer
- [x] **Structured output:** schema → JSON response
- [x] **Multi-step:** 2+ tool calls before answer
- [x] **Tool failure:** `Err` in tool → agent continues
- [x] **Blacklisted tool:** → retry without executing
- [x] **Max steps:** agent stops after N cycles
- [x] **Parallel tools:** multiple simultaneous calls
- [x] **Sub-agents:** delegation and result passing
- [x] **Checkpointing:** save and resume sessions
- [x] **Trace:** non-empty after run
