# Configuration

## AgentConfig

All behavioral limits are set through `AgentConfig`:

```rust
pub struct AgentConfig {
    pub max_steps:             usize,   // Hard cap on Planning cycles
    pub max_retries:           usize,   // Low-confidence retry budget
    pub confidence_threshold:  f64,     // Confidence floor before reflection
    pub reflect_every_n_steps: usize,   // Periodic history compression interval
    pub min_answer_length:     usize,   // Minimum chars for a valid final answer
    pub parallel_tools:        bool,    // Enable/disable parallel execution
    pub models: HashMap<String, String>, // task_type → model name
    pub output_schema: Option<OutputSchema>, // Structured output schema
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps:             15,
            max_retries:           3,
            confidence_threshold:  0.4,
            reflect_every_n_steps: 5,
            min_answer_length:     5,
            parallel_tools:        true,
            models:                HashMap::new(),
            output_schema:         None,
        }
    }
}
```

---

## Setting Configuration

### Via Builder — Convenience Methods

```rust
AgentBuilder::new("task")
    .openai("")
    .max_steps(10)
    .parallel_tools(false)
    .build()?
```

### Via Builder — Full Config

```rust
use agent_b::AgentConfig;

AgentBuilder::new("task")
    .openai("")
    .config(AgentConfig {
        max_steps:             20,
        max_retries:           5,
        confidence_threshold:  0.3,
        reflect_every_n_steps: 8,
        min_answer_length:     50,
        ..Default::default()
    })
    .build()?
```

---

## Configuration Fields

### `max_steps` (default: 15)

The maximum number of Planning cycles. When `step >= max_steps`, `Event::MaxSteps` → `Error` state.

### `max_retries` (default: 3)

Low-confidence retry budget. When confidence < threshold and retries remain, `LowConfidence` event triggers `Reflecting`.

### `confidence_threshold` (default: 0.4)

Minimum confidence score to accept a tool call without reflection. Most built-in callers return `1.0`, so this is mainly useful with custom callers.

### `reflect_every_n_steps` (default: 5)

After every N tool calls, `ObservingState` triggers history compression via `Reflecting`. Set to 0 to disable.

### `min_answer_length` (default: 5)

Minimum character length for a final answer. Shorter answers trigger `AnswerTooShort` which loops back to `Planning`. **Skipped for structured output** (`LlmResponse::Structured`).

### `parallel_tools` (default: true)

Enable/disable parallel tool execution for multi-tool-call LLM responses.

### `output_schema` (default: None)

When set, the LLM is instructed to return JSON conforming to this schema:

```rust
AgentBuilder::new("Extract person info")
    .output_schema(schema)
    .build()?
```

## Advanced Configurations

For detailed explanations of the following advanced configuration capabilities, refer to [docs/advanced.md](advanced.md).

- **`fork_strategy(ForkConfig)`**: Branch the agent into parallel universes when confident reasoning fails.
- **`routing_policy(RoutingPolicy)`**: Dynamically switch the active LLM based on specific runtime triggers.
- **`self_healing(HealingPolicy)`**: Trap errors and LLM hallucinations before they become fatal.
- **`introspection(IntrospectionEngine)`**: Run background telemetry detectors to flag anomalies to the agent.
- **`replay_recording(ReplayRecording)`**: Record every micro-state transition into an NDJSON file for debugging.
- **`planning_mode(PlanningMode)`**: Create formal step-by-step plans before executing tools.
- **`tool_composition(CompositionConfig)`**: Allow the agent to synthesize new tools from primitives.
- **`invariant(name, closure)`**: Halt the agent immediately if a core safety property is violated.
    .openai("")
    .output_schema("person", json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    }))
    .build()?;
```

---

## Token Budgeting

Limit the agent's resource consumption:

```rust
AgentBuilder::new("task")
    .openai("")
    .max_tokens(5000)       // Simple total token limit
    .build()?
```

When exceeded, `Event::FatalError` → `Error` state.

---

## System Prompt

Prepended to every LLM call as a `"system"` role message:

```rust
AgentBuilder::new("task")
    .openai("")
    .system_prompt("You are a financial analyst. Always cite your sources.")
    .build()?
```

---

## Models

### Via builder (recommended)

```rust
AgentBuilder::new("task")
    .model("gpt-4o")                           // sets ["default"]
    .model_for("calculation", "gpt-4o-mini")   // cheaper for math
    .model_for("research",    "gpt-4o")        // best for research
```

### Resolution order

1. `models[task_type]` — exact match
2. `models["default"]` — generic fallback
3. `""` — empty string (LlmCaller decides)

---

## Configuration Recipes

### Fast, Cheap Agent
```rust
AgentConfig { max_steps: 5, max_retries: 1, confidence_threshold: 0.0,
              reflect_every_n_steps: 0, min_answer_length: 5, ..Default::default() }
```

### Deep Research Agent
```rust
AgentConfig { max_steps: 30, max_retries: 3, confidence_threshold: 0.4,
              reflect_every_n_steps: 5, min_answer_length: 100, ..Default::default() }
```
