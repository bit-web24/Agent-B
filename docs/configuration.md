# Configuration

## AgentConfig

All behavioral limits are set through `AgentConfig`:

```rust
pub struct AgentConfig {
    pub max_steps:             usize,  // Hard cap on Planning cycles
    pub max_retries:           usize,  // Low-confidence retry budget
    pub confidence_threshold:  f64,    // Confidence floor before reflection
    pub reflect_every_n_steps: usize,  // Periodic history compression interval
    pub min_answer_length:     usize,  // Minimum chars for a valid final answer
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps:             15,
            max_retries:           3,
            confidence_threshold:  0.4,
            reflect_every_n_steps: 5,
            min_answer_length:     20,
        }
    }
}
```

---

## Setting Configuration

### Via Builder — Convenience Methods

```rust
AgentBuilder::new("task")
    .llm(llm)
    .max_steps(10)        // quick shortcut for max_steps only
    .build()?
```

### Via Builder — Full Config

```rust
use agentsm::AgentConfig;

AgentBuilder::new("task")
    .llm(llm)
    .config(AgentConfig {
        max_steps:             20,
        max_retries:           5,
        confidence_threshold:  0.3,
        reflect_every_n_steps: 8,
        min_answer_length:     50,
    })
    .build()?
```

### Via Memory After Construction

```rust
engine.memory.config.max_steps = 20;
engine.memory.config.confidence_threshold = 0.3;
```

---

## Configuration Fields in Depth

### `max_steps`

**Default: 15**

The maximum number of Planning cycles. Increment happens at the **start** of each `PlanningState::handle()` call. When `step >= max_steps`:

1. `memory.error` is set to `"Max steps N exceeded"`
2. `Event::MaxSteps` is emitted
3. Engine transitions to `Error` state

```rust
// For simple tasks with a clear answer
.config(AgentConfig { max_steps: 5, ..Default::default() })

// For deep research tasks needing many tool calls
.config(AgentConfig { max_steps: 30, ..Default::default() })
```

> **Safety Valve:** The engine also has a hard safety cap at `max_steps * 3` total loop iterations (counting all state transitions, not just Planning). This prevents infinite loops caused by implementation bugs.

---

### `max_retries`

**Default: 3**

When `PlanningState` receives a tool call with confidence below `confidence_threshold`, it increments `memory.retry_count` and emits `Event::LowConfidence`. This triggers a `Reflecting` cycle, after which Planning tries again.

Once `retry_count >= max_retries`, low confidence is ignored — the tool call is accepted anyway.

```rust
// Zero retries — never trigger reflection for low confidence
.config(AgentConfig { max_retries: 0, ..Default::default() })

// More patience for uncertain LLM responses
.config(AgentConfig { max_retries: 5, ..Default::default() })
```

---

### `confidence_threshold`

**Default: 0.4**

The minimum confidence score required to accept a tool call without triggering reflection. `confidence` comes from the `LlmResponse::ToolCall { confidence, .. }` returned by the LLM caller.

Most built-in callers return `confidence: 1.0` since OpenAI and Anthropic APIs don't natively provide confidence scores. This field is primarily useful when building custom LLM callers that compute confidence from response metadata (e.g., token log-probabilities).

```rust
// Only validate confidence if you have a caller that actually reports it
.config(AgentConfig { confidence_threshold: 0.7, ..Default::default() })

// Disable confidence check entirely
.config(AgentConfig { confidence_threshold: 0.0, ..Default::default() })
```

---

### `reflect_every_n_steps`

**Default: 5**

After every N completed tool calls (tracked by `memory.step`), `ObservingState` emits `Event::NeedsReflection` instead of `Event::Continue`. This triggers `ReflectingState`, which compresses all history into a single summary entry.

**Why this matters:** LLM context windows are finite. Without compression, a long-running agent accumulates more and more history in every request, eventually exceeding the context limit.

```rust
// Disable periodic reflection entirely
.config(AgentConfig { reflect_every_n_steps: 0, ..Default::default() })

// Compress more frequently (smaller context window)
.config(AgentConfig { reflect_every_n_steps: 3, ..Default::default() })

// Compress less frequently (large context window, want full history)
.config(AgentConfig { reflect_every_n_steps: 10, ..Default::default() })
```

**Reflection trigger condition:** `memory.step % reflect_every_n_steps == 0 && reflect_every_n_steps > 0`

---

### `min_answer_length`

**Default: 20**

If the LLM produces a `FinalAnswer` shorter than this in characters, `PlanningState` emits `Event::AnswerTooShort`, which loops back to `Planning` (the LLM gets another chance to produce a complete answer).

This prevents the agent from returning trivially short answers like `"Yes."` or `"Paris."` when a more complete response is expected.

```rust
// For Q&A agents where short answers are fine
.config(AgentConfig { min_answer_length: 5, ..Default::default() })

// For report-writing agents expecting long-form output
.config(AgentConfig { min_answer_length: 200, ..Default::default() })
```

---

## System Prompt

The system prompt is prepended to every LLM call as a `"system"` role message:

```rust
AgentBuilder::new("task")
    .llm(llm)
    .system_prompt("You are a financial analyst. Always cite your sources. \
                    Prefer using the SEC filing tool over web search for company data.")
    .build()?
```

**Best practices:**
- Define the agent's persona and expertise
- Clarify which tools to prefer for which situations
- Set expectations for output format and quality
- Keep it concise — the system prompt is sent with every request

---

## Task Type

Controls which model tier `PlanningState` selects:

```rust
AgentBuilder::new("task")
    .llm(llm)
    .task_type("research")     // → claude-opus-4-6 / gpt-4o
    .task_type("calculation")  // → claude-haiku / gpt-4o-mini
    .task_type("default")      // → claude-sonnet / gpt-4o (the default)
    .build()?
```

---

## Configuration Recipes

### Fast, Cheap Agent (simple tasks)

```rust
AgentConfig {
    max_steps:             5,
    max_retries:           1,
    confidence_threshold:  0.0,  // skip confidence check
    reflect_every_n_steps: 0,    // no reflection
    min_answer_length:     10,
}
```
+ `.task_type("calculation")`  → cheapest model tier

### Deep Research Agent (complex tasks)

```rust
AgentConfig {
    max_steps:             30,
    max_retries:           3,
    confidence_threshold:  0.4,
    reflect_every_n_steps: 5,
    min_answer_length:     100,
}
```
+ `.task_type("research")` → highest quality model tier

### Long-Running Agent (many tool calls)

```rust
AgentConfig {
    max_steps:             50,
    max_retries:           3,
    confidence_threshold:  0.4,
    reflect_every_n_steps: 8,   // reflect less often, trust large context window
    min_answer_length:     50,
}
```
