# Error Handling

---

## Philosophy: Failure as Data

A core design principle of `agentsm-rs` is that **most failures are data, not exceptions**.

When a tool call fails, the error message becomes a `last_observation` — the LLM sees it on the next Planning cycle and can:
- Try a different tool
- Revise the query
- Acknowledge the error in its final answer

Only **unrecoverable library-level failures** surface as `AgentError`:

| Situation | Handling |
|---|---|
| Tool returns `Err(...)` | → `"ERROR: ..."` observation, agent continues |
| LLM API timeout / HTTP 5xx | → `FatalError`, agent transitions to `Error` state |
| Max steps exceeded | → `MaxSteps` event, agent transitions to `Error` state |
| Bug: no transition exists | → `Err(InvalidTransition)` returned from `run()` |

---

## `AgentError` Variants

### `AgentFailed(String)`

The agent reached the `Error` terminal state. The string is `memory.error`.

**Causes:**
- `MaxSteps`: planning steps exhausted — `"Max steps N exceeded"`
- `FatalError`: LLM call returned an unrecoverable error — `"LLM error: ..."`

```rust
match engine.run() {
    Ok(answer) => println!("{}", answer),
    Err(AgentError::AgentFailed(msg)) => {
        println!("Agent failed: {}", msg);
        // Inspect memory for more context
        println!("Steps used: {}", engine.memory.step);
        println!("History: {} entries", engine.memory.history.len());
    }
    Err(other) => eprintln!("Library error: {}", other),
}
```

---

### `InvalidTransition { from, event }`

The engine tried to look up `(from_state, event)` in the transition table and it wasn't there.

**Causes:** Implementation bug (a state returned an event that doesn't exist in the table for that state), or a custom state emitting unsupported events without a matching custom transition.

This should never occur with the default state machine.

```rust
match engine.run() {
    Err(AgentError::InvalidTransition { from, event }) => {
        eprintln!("BUG: no transition for ({}, {})", from, event);
    }
    _ => {}
}
```

---

### `NoHandlerForState(String)`

The engine looked up a handler for the current state name in the handler map and found nothing.

**Causes:** Custom handler map missing a registration for a state that is reachable via the transition table.

---

### `SafetyCapExceeded(usize)`

The engine loop ran more than `max_steps * 3` total iterations.

**Causes:** A cycle in the transition table combined with states that keep emitting looping events without incrementing the step counter. For example: a custom transition where `(Error, Start) → Planning` alongside a PlanningState that always returns `FatalError`.

The default transition table has no such cycles.

---

### `BuildError(String)`

`AgentBuilder::build()` failed. Currently only caused by not calling `.llm()` before `.build()`.

```rust
let result = AgentBuilder::new("task").build();  // no .llm()
assert!(matches!(result, Err(AgentError::BuildError(_))));
```

---

## Tool Error Handling in Practice

### The Tool Function Contract

```rust
type ToolFn = Box<dyn Fn(&HashMap<String, serde_json::Value>) -> Result<String, String>>;
```

- `Ok(String)` → stored as `"SUCCESS: <your string>"` in `memory.last_observation`
- `Err(String)` → stored as `"ERROR: <your string>"` in `memory.last_observation`
- **Panics** → NOT caught — will crash the agent

**Always return errors, never panic:**

```rust
// ❌ Will crash the agent process
Box::new(|args| {
    let val = args["required_field"].as_str().unwrap(); // panics if missing
    Ok(val.to_string())
})

// ✅ Returns error gracefully — agent can recover
Box::new(|args| {
    let val = args.get("required_field")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: required_field")?;
    Ok(val.to_string())
})
```

### Writing Descriptive Tool Errors

Since tool errors become LLM observations, make them informative and actionable:

```rust
// ❌ Unhelpful — LLM doesn't know what to do
Err("error".to_string())

// ✅ Actionable — LLM can try a different query
Err(format!("HTTP 404: URL '{}' not found. Try a different URL or use the search tool instead.", url))

// ✅ Rate limit — LLM can wait (or try a fallback)
Err("Rate limit exceeded (429). Try again in 60 seconds or use an alternative data source.".to_string())
```

---

## LLM Error Handling

`LlmCaller::call()` returns `Result<LlmResponse, String>`. When it returns `Err(...)`:

1. `PlanningState` stores the error in `memory.error`
2. Returns `Event::FatalError`
3. Engine transitions to `Error` state
4. `run()` returns `Err(AgentError::AgentFailed("LLM error: ..."))`

### Retrying Transient LLM Errors

For transient errors (network timeout, rate limit), implement retry logic inside your `AsyncLlmCaller`:

```rust
#[async_trait]
impl AsyncLlmCaller for RetryingOpenAiCaller {
    async fn call_async(&self, memory: &AgentMemory, tools: &ToolRegistry, model: &str) -> Result<LlmResponse, String> {
        let mut last_err = String::new();
        for attempt in 1..=3 {
            match self.inner.call_async(memory, tools, model).await {
                Ok(resp) => return Ok(resp),
                Err(e) if e.contains("429") || e.contains("timeout") => {
                    last_err = e;
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),  // non-retryable — fail immediately
            }
        }
        Err(format!("Failed after 3 retries: {}", last_err))
    }
}
```

---

## Inspecting Failure State After `run()`

Even after `Err(...)`, the engine's memory is intact and inspectable:

```rust
let result = engine.run();

if result.is_err() {
    // What error was recorded?
    eprintln!("memory.error: {:?}", engine.memory.error);

    // How far did we get?
    eprintln!("Steps completed: {}", engine.memory.step);

    // What was the last tool attempted?
    eprintln!("Last tool: {:?}", engine.memory.current_tool_call);

    // Full history up to failure
    for h in &engine.memory.history {
        eprintln!("[step {}] {} → {} ({})",
            h.step, h.tool.name, &h.observation[..50.min(h.observation.len())],
            if h.success { "ok" } else { "fail" });
    }

    // Full trace for diagnostics
    engine.trace().print();
}
```
