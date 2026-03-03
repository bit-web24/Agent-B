# Advanced Usage

---

## Custom State Graphs (LangGraph-style)

States and Events are string-based, so you can define completely custom state machines:

### 1. Define a Custom State

```rust
use agentsm::states::AgentState;
use agentsm::{Event, AgentOutput};
use agentsm::memory::AgentMemory;
use agentsm::tools::ToolRegistry;
use agentsm::llm::AsyncLlmCaller;
use async_trait::async_trait;

struct ResearchingState;

#[async_trait]
impl AgentState for ResearchingState {
    fn name(&self) -> &'static str { "Researching" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &std::sync::Arc<ToolRegistry>,
        llm:       &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        memory.log("Researching", "RESEARCH_DONE", "Research complete");
        Event::new("ResearchDone")
    }
}
```

### 2. Register with the Builder

```rust
let engine = AgentBuilder::new("task")
    .openai("")
    .state("Researching", Arc::new(ResearchingState))
    .transition("Planning", "NeedsResearch", "Researching")
    .transition("Researching", "ResearchDone", "Planning")
    .terminal_state("CustomDone")  // optional: custom terminal states
    .build()?;
```

---

## Checkpointing & Persistence

Supports session persistence for long-running workflows or crash recovery.

### Using the SQLite Store

```rust
use agentsm::checkpoint::SqliteCheckpointStore;

let store = Arc::new(SqliteCheckpointStore::new("agents.db")?);

let mut engine = AgentBuilder::new("Long task")
    .openai("")
    .checkpoint_store(store.clone())
    .session_id("session-123")
    .build()?;

engine.run().await?;
```

### Resuming

```rust
let mut engine = AgentBuilder::new("dummy")
    .openai("")
    .checkpoint_store(store.clone())
    .resume("session-123").await?
    .build()?;
```

**Supported Stores:**
- `MemoryCheckpointStore`: Volatile, thread-safe (for tests)
- `FileCheckpointStore`: JSON files in a directory
- `SqliteCheckpointStore`: Production-grade persistence

---

## Human-in-the-Loop (HIP)

High-risk actions can require human approval via the state machine.

### Approval Policies

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

When approval is required, the agent transitions to `WaitingForHuman` state.

---

## Sub-Agents as Tools

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

The sub-agent runs to completion and its final answer becomes the tool observation for the parent.

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

## Accessing Memory After a Run

```rust
let result = engine.run().await;

println!("Answer: {:?}", engine.memory.final_answer);
println!("Steps: {}/{}", engine.memory.step, engine.memory.config.max_steps);
println!("Tokens used: {:?}", engine.memory.total_usage);

for entry in &engine.memory.history {
    println!("[step {}] {} → {} ({})",
        entry.step, entry.tool.name,
        &entry.observation[..entry.observation.len().min(50)],
        if entry.success { "✓" } else { "✗" });
}

engine.trace().print();
```
