# Testing Guide

`agentsm-rs` is designed for testability from the ground up. The `MockLlmCaller` enables full agent runs without any network calls.

---

## Running Tests

```bash
# All tests
cargo test

# Only integration tests
cargo test --test integration_tests

# A specific test
cargo test test_full_run_with_mock_llm

# With output printed
cargo test -- --nocapture
```

---

## The MockLlmCaller

`MockLlmCaller` returns pre-programmed `LlmResponse` values in sequence. It implements `LlmCaller` directly (not `AsyncLlmCaller`), so no wrapping is needed:

```rust
use agentsm::llm::MockLlmCaller;
use agentsm::{LlmResponse, ToolCall};
use std::collections::HashMap;

let mock = MockLlmCaller::new(vec![
    // First LLM call: request a tool
    LlmResponse::ToolCall {
        tool: ToolCall {
            name: "search".to_string(),
            args: HashMap::from([
                ("query".to_string(), serde_json::json!("Rust programming language")),
            ]),
        },
        confidence: 1.0,
    },
    // Second LLM call: produce the final answer
    LlmResponse::FinalAnswer {
        content: "Rust is a systems programming language focused on safety, \
                  speed, and concurrency without garbage collection.".to_string(),
    },
]);

let mut engine = AgentBuilder::new("What is Rust?")
    .llm(Box::new(mock))       // ← no LlmCallerExt wrapper needed
    .tool("search", "search tool", json!({ "type": "object", "properties": {} }),
          Box::new(|_| Ok("Rust is a systems language...".to_string())))
    .build()
    .expect("builder should succeed");

let result = engine.run();
assert!(result.is_ok());
```

---

## Test Helper Functions

Copy these into your test file to reduce boilerplate:

```rust
use agentsm::{AgentBuilder, LlmResponse, ToolCall, ToolRegistry, AgentMemory};
use agentsm::llm::MockLlmCaller;
use serde_json::json;
use std::collections::HashMap;

fn test_memory() -> AgentMemory {
    AgentMemory::new("test task")
}

fn test_tools() -> ToolRegistry {
    ToolRegistry::new()
}

fn tool_call_resp(name: &str) -> LlmResponse {
    LlmResponse::ToolCall {
        tool: ToolCall { name: name.to_string(), args: HashMap::new() },
        confidence: 1.0,
    }
}

fn final_answer_resp(content: &str) -> LlmResponse {
    LlmResponse::FinalAnswer { content: content.to_string() }
}

fn mock_llm(responses: Vec<LlmResponse>) -> MockLlmCaller {
    MockLlmCaller::new(responses)
}

/// Build a complete engine with a dummy tool registered
fn engine_with_mock(responses: Vec<LlmResponse>) -> agentsm::AgentEngine {
    AgentBuilder::new("test task")
        .llm(Box::new(mock_llm(responses)))
        .tool(
            "dummy",
            "A test tool",
            json!({ "type": "object", "properties": {} }),
            Box::new(|_| Ok("dummy result".to_string())),
        )
        .build()
        .expect("builder should succeed")
}
```

---

## Testing Individual State Handlers

States can be tested in complete isolation by calling `handle()` directly:

```rust
use agentsm::states::{AgentState, PlanningState, ActingState, ObservingState};
use agentsm::{Event, ToolCall};

#[test]
fn test_planning_increments_step() {
    let mut memory = test_memory();
    let tools = test_tools();
    let llm = mock_llm(vec![final_answer_resp("A long enough answer to pass the minimum check.")]);

    assert_eq!(memory.step, 0);

    let state = PlanningState;
    state.handle(&mut memory, &tools, &llm);

    assert_eq!(memory.step, 1, "Planning must increment step exactly once");
}

#[test]
fn test_acting_with_missing_tool_call() {
    let mut memory = test_memory();
    // current_tool_call is None — acting without planning first
    let tools = test_tools();
    let llm   = mock_llm(vec![]);

    let event = ActingState.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::FatalError);
    assert!(memory.error.is_some());
}

#[test]
fn test_observing_commits_then_clears() {
    let mut memory = test_memory();
    memory.step = 1;
    memory.config.reflect_every_n_steps = 0; // disable reflection
    memory.current_tool_call = Some(ToolCall {
        name: "search".to_string(),
        args: HashMap::new(),
    });
    memory.last_observation = Some("SUCCESS: found data".to_string());

    let event = ObservingState.handle(&mut memory, &test_tools(), &mock_llm(vec![]));

    assert_eq!(event, Event::Continue);
    assert_eq!(memory.history.len(), 1);
    assert!(memory.current_tool_call.is_none());
    assert!(memory.last_observation.is_none());
}
```

---

## Testing Full Agent Runs

For end-to-end tests, program the mock with the exact sequence of responses you expect:

```rust
#[test]
fn test_agent_with_two_tool_calls() {
    let mut engine = engine_with_mock(vec![
        tool_call_resp("dummy"),    // step 1: tool call
        tool_call_resp("dummy"),    // step 2: another tool call
        final_answer_resp("After two searches, the complete answer is: 42 is the answer."),
    ]);

    let result = engine.run();
    assert!(result.is_ok());

    let answer = result.unwrap();
    assert!(answer.contains("42"));

    // Verify history
    assert_eq!(engine.memory.history.len(), 2, "Two tool calls should create two history entries");
    assert!(engine.memory.history.iter().all(|h| h.success));

    // Verify trace
    assert!(engine.trace().len() > 0);
    assert!(!engine.trace().for_state("Acting").is_empty());
}
```

---

## Testing the Transition Table

Verify your custom transitions without running the full engine:

```rust
use agentsm::transitions::build_transition_table;
use agentsm::{State, Event};

#[test]
fn test_transition_table_coverage() {
    let table = build_transition_table();

    // Standard expected transitions
    assert_eq!(table.get(&(State::Idle,      Event::Start)),          Some(&State::Planning));
    assert_eq!(table.get(&(State::Planning,  Event::LlmToolCall)),    Some(&State::Acting));
    assert_eq!(table.get(&(State::Planning,  Event::LlmFinalAnswer)), Some(&State::Done));
    assert_eq!(table.get(&(State::Acting,    Event::ToolSuccess)),    Some(&State::Observing));
    assert_eq!(table.get(&(State::Acting,    Event::ToolFailure)),    Some(&State::Observing));
    assert_eq!(table.get(&(State::Observing, Event::Continue)),       Some(&State::Planning));

    // Terminal states have no outgoing transitions
    assert!(table.get(&(State::Done,  Event::Start)).is_none());
    assert!(table.get(&(State::Error, Event::Start)).is_none());
}
```

---

## Testing Tool Registration and Execution

```rust
use agentsm::ToolRegistry;

#[test]
fn test_tool_registry() {
    let mut registry = ToolRegistry::new();

    registry.register(
        "add",
        "Add two numbers",
        serde_json::json!({ "type": "object", "properties": {} }),
        Box::new(|args| {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(format!("{}", a + b))
        }),
    );

    assert!(registry.has("add"));
    assert!(!registry.has("subtract"));
    assert_eq!(registry.len(), 1);

    // Test execution
    let args = [
        ("a".to_string(), serde_json::json!(3.0)),
        ("b".to_string(), serde_json::json!(4.0)),
    ].into();
    let result = registry.execute("add", &args);
    assert_eq!(result, Ok("7".to_string()));

    // Unknown tool → Err, no panic
    let unknown = registry.execute("nonexistent", &std::collections::HashMap::new());
    assert!(unknown.is_err());
}
```

---

## Testing the Builder

```rust
#[test]
fn test_builder_requires_llm() {
    let result = AgentBuilder::new("test").build();
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(matches!(err, agentsm::AgentError::BuildError(_)));
}

#[test]
fn test_builder_succeeds_with_llm() {
    let result = AgentBuilder::new("test")
        .llm(Box::new(mock_llm(vec![])))
        .build();
    assert!(result.is_ok());
}
```

---

## What to Test

Coverage checklist for a well-tested agent:

- [ ] **Happy path:** tool call → tool success → final answer
- [ ] **Multi-step:** 2+ tool calls before final answer
- [ ] **Tool failure:** tool returns `Err` → agent continues, not crashes
- [ ] **Unknown tool:** LLM requests unregistered tool → `ToolFailure`
- [ ] **Blacklist:** LLM requests blacklisted tool → `ToolBlacklisted`, retries
- [ ] **Max steps:** agent stops after N planning cycles
- [ ] **No LLM:** builder without `.llm()` returns `BuildError`
- [ ] **Trace:** after run, trace is non-empty with expected state entries
- [ ] **History:** after tool calls, `memory.history` has entries with correct fields
- [ ] **Reflection:** at step % N == 0, `NeedsReflection` fires
