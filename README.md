# agentsm-rs

> **A production-grade, zero-framework Agentic AI library for Rust built on the Hybrid State Machine pattern.**
>
> Transition Table owns the graph. State Traits own the behavior. You own everything.

---

## Features

| Feature | Description |
|---|---|
| **Hybrid State Machine** | 9 built-in states with a fully extensible transition table |
| **LLM Providers** | OpenAI, Anthropic (Claude), and any OpenAI-compatible API (Groq, Ollama, Together, etc.) |
| **Structured Output** | Force LLM to return JSON conforming to a user-defined schema |
| **Streaming** | Real-time token streaming with `run_streaming()` |
| **Parallel Tool Execution** | Execute multiple tool calls concurrently via `tokio::spawn` |
| **Human-in-the-Loop (HIP)** | Approval workflows with `AlwaysAsk`, `NeverAsk`, `AskAbove(RiskLevel)`, and `ToolBased` policies |
| **Checkpointing & Crash Recovery** | SQLite, File, and In-Memory checkpoint stores |
| **Token Budget Management** | Track and enforce session-wide token usage limits |
| **Sub-Agents as Tools** | Delegate tasks to specialized child agents recursively |
| **MCP (Model Context Protocol)** | Connect to MCP servers via stdio transport and use their tools |
| **Custom State Graphs** | Define your own states, events, and transitions (LangGraph-style) |
| **Retry with Back-off** | Automatic retry for transient LLM errors with exponential back-off |
| **Tool Blacklisting** | Prevent the agent from calling specific tools |
| **Full Trace** | Event-sourced execution log for observability and debugging |
| **Agent Forking** | Spatially explore multiple reasoning paths in parallel |
| **Adaptive Model Routing** | Switch models dynamically based on cost or confidence |
| **Self-Healing Policies** | Intercept errors and dynamically apply fallback actions |
| **Agent Introspection** | Anomaly detection monitors agent metrics behind the scenes |
| **Deterministic Replay** | Record state transitions to ndjson for perfect reproducibility |
| **Execution Contracts** | `Guard`, `Invariant`, and `PostCondition` closures for safety |
| **Plan-and-Execute** | Native step-by-step planning and dynamic replanning |
| **Tool Composition** | Synthesize new pipelined tools at runtime |

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
agentsm = { path = "." }
tokio = { version = "1", features = ["full"] }
```

### Minimal Agent (OpenAI)

```rust
use agentsm::{AgentBuilder, Tool};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut engine = AgentBuilder::new("What is the capital of France?")
        .openai("")  // reads OPENAI_API_KEY from env
        .model("gpt-4o")
        .system_prompt("You are a helpful assistant. Use tools when needed.")
        .add_tool(
            Tool::new("search", "Search for information")
                .param("query", "string", "The search query")
                .call(|args| {
                    let q = args["query"].as_str().unwrap_or("");
                    Ok(format!("Paris is the capital of France."))
                })
        )
        .build()?;

    let answer = engine.run().await?;
    println!("Answer: {}", answer);
    engine.trace().print();
    Ok(())
}
```

### Using Anthropic (Claude)

```rust
let mut engine = AgentBuilder::new("Explain Rust's ownership model")
    .anthropic("")  // reads ANTHROPIC_API_KEY from env
    .model("claude-sonnet-4-6")
    .build()?;
```

### Using Ollama / Groq / Any OpenAI-Compatible API

```rust
// Ollama (local)
let engine = AgentBuilder::new("task")
    .ollama("http://localhost:11434")
    .model("llama3.2")
    .build()?;

// Groq
let engine = AgentBuilder::new("task")
    .groq("gsk-your-key")
    .model("llama-3.3-70b-versatile")
    .build()?;
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PUBLIC API                            │
│              AgentBuilder  (builder.rs)                  │
└────────────────────────┬────────────────────────────────┘
                         │ builds
┌────────────────────────▼────────────────────────────────┐
│                   AgentEngine  (engine.rs)               │
│   - owns AgentMemory, Arc<ToolRegistry>                  │
│   - owns Arc<dyn AsyncLlmCaller>                         │
│   - owns TransitionTable + StateHandlerMap               │
│   - runs the loop: handle() → event → lookup → repeat    │
└──────┬──────────────────────┬──────────────────────────┘
       │                      │
       ▼                      ▼
┌─────────────┐    ┌──────────────────────┐
│ Transition  │    │   State Handlers     │
│   Table     │    │  (states/*.rs)       │
│             │    │                      │
│ HashMap<    │    │  Idle → Planning     │
│  (State,    │    │  Planning → Acting   │
│   Event),   │    │  Acting → Observing  │
│  State>     │    │  Observing → Plan/Ref│
│             │    │  Reflecting → Plan   │
└─────────────┘    │  Done / Error (term) │
                   │  WaitingForHuman     │
                   │  ParallelActing      │
                   └──────────────────────┘
```

### State Machine Flow

```
           ┌──────┐
           │ IDLE │
           └──┬───┘
              │ Start
              ▼
     ┌──────────────────┐
┌───▶│    PLANNING      │◀──────────────────────┐
│    └────────┬─────────┘                       │
│             │                                 │
│    ┌────────┴──────────────────┐              │
│    │                           │              │
│    ▼ LlmToolCall               ▼ LlmFinalAns │
│ ┌────────┐                  ┌──────┐         │
│ │ ACTING │                  │ DONE │         │
│ └───┬────┘                  └──────┘         │
│     │ ToolSuccess/ToolFailure                 │
│     ▼                                        │
│ ┌───────────┐ NeedsReflection ┌────────────┐ │
│ │ OBSERVING │────────────────▶│ REFLECTING │ │
│ └─────┬─────┘                 └──────┬─────┘ │
│       │ Continue                     │ Done   │
└───────┘                              └────────┘

MaxSteps / FatalError from any state ──▶ ERROR (terminal)
```

---

## Builder API Reference

### Core Configuration

| Method | Description |
|---|---|
| `AgentBuilder::new(task)` | Create a new builder with the given task |
| `.llm(Arc<dyn AsyncLlmCaller>)` | Set the LLM caller (required) |
| `.openai(api_key)` | Shorthand: OpenAI caller (empty string reads from env) |
| `.anthropic(api_key)` | Shorthand: Anthropic caller |
| `.ollama(base_url)` | Shorthand: Ollama caller |
| `.groq(api_key)` | Shorthand: Groq caller |
| `.model(name)` | Set default model name |
| `.model_for(task_type, model)` | Set model for a specific task type |
| `.system_prompt(prompt)` | Set the system prompt |
| `.task_type(type)` | Set the task type (for model routing) |
| `.max_steps(n)` | Maximum planning steps (default: 15) |
| `.retry_on_error(n)` | Enable retry wrapper with n retries |
| `.fork_strategy(strategy)` | Configure parallel branch speculation |
| `.routing_policy(policy)` | Dynamic model swapping |
| `.self_healing(policy)` | Auto-recover from tool/LLM failures |
| `.introspection(engine)` | Anomaly detection engine |
| `.replay_recording(mode)` | Enable state trace recording |
| `.planning_mode(mode)` | Pre-planning and sub-tasking |
| `.tool_composition(config)` | Runtime tool synthesis |
| `.invariant(name, fn)` | Execution invariants and contracts |

### Tool Registration

```rust
// Method 1: Fluent Tool builder (recommended)
.add_tool(
    Tool::new("calculator", "Adds two numbers")
        .param("a", "number", "First number")       // required param
        .param("b", "number", "Second number")       // required param
        .param_opt("precision", "integer", "Decimal places") // optional param
        .call(|args| {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok((a + b).to_string())
        })
)

// Method 2: Raw schema
.tool(
    "search",
    "Search the web",
    json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] }),
    Arc::new(|args| Ok(format!("Results for: {}", args["query"]))),
)
```

### Advanced Features

| Method | Description |
|---|---|
| `.blacklist_tool(name)` | Prevent the agent from calling a tool |
| `.parallel_tools(true)` | Enable parallel tool execution |
| `.approval_policy(policy)` | Set human-in-the-loop policy |
| `.on_approval(callback)` | Set the approval callback function |
| `.output_schema(name, schema)` | Request structured JSON output conforming to schema |
| `.checkpoint_store(store)` | Enable checkpointing with a store |
| `.session_id(id)` | Set session ID for checkpointing |
| `.resume(session_id).await?` | Resume from a checkpoint |
| `.max_tokens(n)` | Set token budget limit |
| `.mcp_server(cmd, args)` | Connect to an MCP server and register its tools |
| `.add_subagent(name, desc, builder)` | Register a sub-agent as a tool |
| `.state(name, handler)` | Register a custom state handler |
| `.transition(from, event, to)` | Add a custom transition |
| `.terminal_state(name)` | Register a custom terminal state |

---

## Streaming

```rust
use futures::StreamExt;

let mut stream = engine.run_streaming();

while let Some(output) = stream.next().await {
    match output {
        AgentOutput::LlmToken(token) => print!("{}", token),
        AgentOutput::StateStarted(state) => println!("\n[STATE] {}", state),
        AgentOutput::ToolCallStarted { name, args } => println!("[TOOL] {} {:?}", name, args),
        AgentOutput::ToolCallFinished { name, result, success } => {
            println!("[RESULT] {} (ok={}): {}", name, success, result);
        }
        AgentOutput::FinalAnswer(answer) => println!("\n✅ {}", answer),
        AgentOutput::Error(err) => eprintln!("❌ {}", err),
        _ => {}
    }
}
```

---

## Structured Output

Force the LLM to return JSON conforming to a user-defined schema:

```rust
let mut engine = AgentBuilder::new("Extract info about Rust")
    .openai("")
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
let parsed: serde_json::Value = serde_json::from_str(&answer)?;
assert_eq!(parsed["name"], "Rust");
```

**Provider implementations:**
- **OpenAI** → `response_format: json_object` + schema injected into system prompt
- **Anthropic** → Synthetic tool with schema as `input_schema`

---

## Parallel Tool Execution

When the LLM returns multiple tool calls in a single response, they execute concurrently:

```rust
let engine = AgentBuilder::new("Compare weather in NYC and London")
    .openai("")
    .parallel_tools(true)
    .add_tool(weather_tool)
    .build()?;
```

---

## Human-in-the-Loop (HIP)

Control which tool calls require human approval:

```rust
use agentsm::human::{ApprovalPolicy, RiskLevel, HumanDecision};

let engine = AgentBuilder::new("task")
    .openai("")
    .approval_policy(ApprovalPolicy::AlwaysAsk)
    // Or: ApprovalPolicy::AskAbove(RiskLevel::High)
    // Or: ApprovalPolicy::ToolBased(risk_map)
    // Or: ApprovalPolicy::NeverAsk (default)
    .on_approval(|req| {
        println!("Approve {}? (y/n)", req.tool_name);
        HumanDecision::Approved
        // Or: HumanDecision::Rejected("reason".into())
        // Or: HumanDecision::Modified { tool_name, tool_args }
    })
    .build()?;
```

---

## Checkpointing & Crash Recovery

```rust
use agentsm::checkpoint::{SqliteCheckpointStore, FileCheckpointStore, MemoryCheckpointStore};

// SQLite store
let store = Arc::new(SqliteCheckpointStore::new("checkpoints.db")?);

// Build with checkpointing
let engine = AgentBuilder::new("task")
    .openai("")
    .checkpoint_store(store.clone())
    .session_id("session-123")
    .build()?;

// Resume from checkpoint
let engine = AgentBuilder::new("dummy")
    .openai("")
    .checkpoint_store(store.clone())
    .resume("session-123").await?
    .build()?;
```

---

## Token Budget Management

```rust
let engine = AgentBuilder::new("task")
    .openai("")
    .max_tokens(10_000)  // session-wide budget
    .build()?;

// After run, check usage:
println!("Used: {} tokens", engine.memory.total_usage.total_tokens);
```

---

## Sub-Agents

Delegate complex tasks to specialized child agents:

```rust
let calculator = AgentBuilder::new("Calculate sum")
    .openai("")
    .add_tool(calc_tool);

let parent = AgentBuilder::new("Calculate 20 + 22")
    .openai("")
    .add_subagent("calculator", "A math specialist", calculator)
    .build()?;
```

---

## MCP (Model Context Protocol)

Connect to MCP servers and use their tools:

```rust
let engine = AgentBuilder::new("task")
    .openai("")
    .mcp_server("python3", &["math_server.py".into()])
    .build()?;
// All tools from the MCP server are now available to the agent
```

---

## Custom State Graphs

Define your own states, events, and transitions (LangGraph-style):

```rust
use agentsm::states::AgentState;

struct ResearchingState;

#[async_trait]
impl AgentState for ResearchingState {
    fn name(&self) -> &'static str { "Researching" }

    async fn handle(
        &self,
        memory: &mut AgentMemory,
        tools: &Arc<ToolRegistry>,
        llm: &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        // Your custom logic here
        memory.log("Researching", "DONE", "Research complete");
        Event::new("ResearchDone")
    }
}

let engine = AgentBuilder::new("task")
    .openai("")
    .state("Researching", Arc::new(ResearchingState))
    .transition("Planning", "NeedsResearch", "Researching")
    .transition("Researching", "ResearchDone", "Planning")
    .build()?;
```

---

## The Three Laws

1. **The Transition Table is the Single Source of Truth.** Every legal state transition lives in one `HashMap`. If a transition is not in the table, it cannot happen.

2. **Failure is Data, Not Exceptions.** Tool failures and LLM errors are fed back into the agent's context as observations. The agent decides what to do next.

3. **The Engine is Dumb.** `AgentEngine` knows nothing about what states do. It only wires the table + handlers. No business logic ever lives in the engine.

---

## Project Structure

```
agentsm-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Public API surface and re-exports
│   ├── types.rs         # State, Event, LlmResponse, AgentConfig, AgentOutput
│   ├── memory.rs        # AgentMemory — the agent's working state
│   ├── events.rs        # Event type with built-in constants
│   ├── transitions.rs   # TransitionTable — single source of truth
│   ├── tools.rs         # ToolRegistry, Tool builder, ToolFn
│   ├── engine.rs        # AgentEngine — the dumb loop
│   ├── trace.rs         # Event-sourced execution trace
│   ├── error.rs         # AgentError enum
│   ├── builder.rs       # AgentBuilder — the fluent API
│   ├── human.rs         # HIP: ApprovalPolicy, RiskLevel, HumanDecision
│   ├── budget.rs        # TokenBudget, TokenUsage
│   ├── checkpoint.rs    # CheckpointStore trait + SQLite/File/Memory impls
│   ├── states/
│   │   ├── mod.rs       # AgentState trait
│   │   ├── idle.rs
│   │   ├── planning.rs  # LLM call, confidence check, model routing
│   │   ├── acting.rs    # Single tool execution
│   │   ├── parallel_acting.rs  # Concurrent multi-tool execution
│   │   ├── observing.rs # History commit, reflection trigger
│   │   ├── reflecting.rs
│   │   ├── waiting_for_human.rs
│   │   ├── done.rs
│   │   └── error.rs
│   ├── llm/
│   │   ├── mod.rs       # AsyncLlmCaller trait, LlmCaller trait
│   │   ├── openai.rs    # OpenAI + compatible APIs
│   │   ├── anthropic.rs # Anthropic Claude (native reqwest)
│   │   ├── mock.rs      # MockLlmCaller for testing
│   │   └── retry.rs     # RetryingLlmCaller wrapper
│   └── mcp/
│       ├── mod.rs       # MCP bridge: bridge_mcp_tool()
│       ├── client.rs    # McpClient — JSON-RPC over stdio
│       ├── transport.rs # StdioTransport
│       └── types.rs     # JSON-RPC types, McpTool, CallToolResult
├── examples/
│   ├── basic_agent.rs
│   ├── multi_tool_agent.rs
│   ├── anthropic_agent.rs
│   └── streaming_agent.rs
└── tests/
    ├── integration_tests.rs   # 21 tests: full lifecycle, state handlers, retries, custom states
    ├── human_loop_test.rs     # 3 tests: approval, rejection, modification
    ├── parallel_tool_test.rs  # 1 test: concurrent execution timing
    ├── persistence_test.rs    # 3 tests: Memory, File, SQLite stores
    ├── subagent_test.rs       # 2 tests: delegation, nested sub-agents
    ├── budget_test.rs         # 2 tests: accumulation, enforcement
    └── mcp_integration.rs     # 1 test: MCP tool bridge
```

---

## Running Examples

```bash
# Basic agent with OpenAI
OPENAI_API_KEY=sk-... cargo run --example basic_agent

# Multi-tool agent
OPENAI_API_KEY=sk-... cargo run --example multi_tool_agent

# Anthropic agent
ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_agent

# Streaming agent
OPENAI_API_KEY=sk-... cargo run --example streaming_agent
```

## Running Tests

```bash
cargo test
```

All tests use `MockLlmCaller` — no API keys or network calls required.

---

## License

MIT
