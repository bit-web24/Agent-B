# Error Handling

---

## Philosophy: Failure as Data

A core design principle of `agentsm-rs` is that **most failures are data, not exceptions**.

When a tool call fails, the error message becomes a `last_observation` — the LLM sees it on the next Planning cycle and can try a different approach.

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

```rust
match engine.run().await {
    Ok(answer) => println!("{}", answer),
    Err(AgentError::AgentFailed(msg)) => {
        println!("Agent failed: {}", msg);
        println!("Steps used: {}", engine.memory.step);
    }
    Err(other) => eprintln!("Library error: {}", other),
}
```

### `InvalidTransition { from, event }`

A `(State, Event)` pair not in the transition table. Indicates a bug in custom state handlers.

### `NoHandlerForState(String)`

No handler registered for the current state name.

### `SafetyCapExceeded(usize)`

Engine loop exceeded `max_steps * 3` total iterations (prevents infinite loops).

### `BuildError(String)`

`AgentBuilder::build()` failed (e.g., missing `.llm()` or provider shortcut).

---

## Tool Error Handling

```rust
// Tool function signature:
Arc<dyn Fn(&HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync>
```

- `Ok(String)` → stored as `"SUCCESS: <string>"` in observation
- `Err(String)` → stored as `"ERROR: <string>"` in observation
- **Panics** → NOT caught, will crash the agent

```rust
// ✅ Return descriptive errors — let the LLM recover
Tool::new("fetch_url", "Fetch a URL")
    .param("url", "string", "URL to fetch")
    .call(|args| {
        let url = args.get("url").and_then(|v| v.as_str())
            .ok_or("Missing required field: url")?;
        if url.starts_with("file://") {
            return Err(format!("Cannot fetch local URLs: {}", url));
        }
        Ok("Page content here".into())
    })
```

---

## LLM Error Handling

When `AsyncLlmCaller::call_async()` returns `Err(...)`:
1. `PlanningState` stores the error in `memory.error`
2. Returns `Event::fatal_error()`
3. Engine transitions to `Error` → `run()` returns `Err(AgentFailed(...))`

Use `.retry_on_error(n)` for transient errors:

```rust
AgentBuilder::new("task")
    .openai("")
    .retry_on_error(3)   // 1s → 2s → 4s back-off
    .build()?
```

---

## Inspecting Failure State

```rust
let result = engine.run().await;

if result.is_err() {
    eprintln!("Error: {:?}", engine.memory.error);
    eprintln!("Steps: {}", engine.memory.step);
    eprintln!("Last tool: {:?}", engine.memory.current_tool_call);
    engine.trace().print();
}
```
