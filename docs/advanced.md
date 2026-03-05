# Advanced Usage

---

## Custom State Graphs (LangGraph-style)

States and Events are string-based, so you can define completely custom state machines:

### 1. Define a Custom State

```rust
use agent_b::states::AgentState;
use agent_b::{Event, AgentOutput};
use agent_b::memory::AgentMemory;
use agent_b::tools::ToolRegistry;
use agent_b::llm::AsyncLlmCaller;
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
use agent_b::checkpoint::SqliteCheckpointStore;

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
use agent_b::human::{ApprovalPolicy, RiskLevel, HumanDecision};

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

---

## 8 New Advanced Features

### 1. Agent Forking (Speculative Execution)

Run multiple parallel agent branches exploring different paths. The best result is chosen automatically:

```rust
use agent_b::fork::{ForkConfig, MergeStrategy, StepEfficiencyScorer};

let config = ForkConfig::new(3) // 3 parallel branches
    .max_depth(10) // 10 steps max per branch
    .merge(MergeStrategy::MostConfident)
    .scorer(std::sync::Arc::new(StepEfficiencyScorer));

let engine = AgentBuilder::new("Complex task")
    .openai("gpt-4o")
    .fork_strategy(config)
    .build()?;
```

### 2. Adaptive Model Routing

Dynamically route requests to different models based on context, cost, or agent state:

```rust
use agent_b::routing::{RoutingPolicy, RoutingRule, condition};

let mut policy = RoutingPolicy::new("gpt-3.5-turbo"); // Default
// Route to GPT-4 for high confidence steps >= 5
policy.add_rule(RoutingRule {
    model: "gpt-4".into(),
    condition: condition::step_count_greater_than(5),
});

let engine = AgentBuilder::new("task")
    .openai("gpt-4o")
    .routing_policy(policy)
    .build()?;
```

### 3. Self-Healing Policies

Automatically pause, reset, or dial down the temperature when the agent hits failure loops:

```rust
use agent_b::healing::{HealingPolicy, Trigger, Action};

let mut healing = HealingPolicy::new();
healing.add_rule(
    Trigger::ToolFailureLoop { tool_name: "search".into(), max_failures: 3 },
    Action::PauseForHuman { message: "Search failed 3 times".into() }
);

let engine = AgentBuilder::new("task")
    .openai("gpt-4o")
    .self_healing(healing)
    .build()?;
```

### 4. Agent Introspection & Anomaly Detection

Monitors the agent's behavior for infinite loops, repetitive outputs, or idle tool usage:

```rust
use agent_b::introspection::{IntrospectionEngine, LoopDetector};

let mut insights = IntrospectionEngine::new();
insights.add_detector(LoopDetector { max_repetitions: 3 });

let engine = AgentBuilder::new("task")
    .openai("gpt-4o")
    .introspection(insights)
    .build()?;
```

### 5. Deterministic Replay

Record every action. Replay from checkpoints to debug or patch intermediate steps:

```rust
use agent_b::replay::ReplayRecording;

let mut engine = AgentBuilder::new("task")
    .openai("gpt-4o")
    .replay_recording(ReplayRecording::Full)
    .session_id("run-1")
    .build()?;

// Later: Load and patch a run
let ndjson = engine.memory.replay_recorder.to_ndjson();
```

### 6. Execution Contracts & Invariants

Enforce strict pre/post-conditions on state transitions. Perfect for safety-critical tasks:

```rust
let engine = AgentBuilder::new("task")
    .openai("gpt-4o")
    .invariant("Balance >= 0", |m| m.confidence_score >= 0.0) // Must always be true
    .postcondition("Acting", "Success", |m| m.history.last().unwrap().success)
    .build()?;
```

### 7. Explicit Plan-and-Execute

Force the agent to generate and manage an explicit step-by-step plan before execution:

```rust
use agent_b::plan::{PlanningMode, PlanRevisionTrigger};

let mode = PlanningMode::Explicit {
    max_plan_steps: 5,
    revision_triggers: vec![PlanRevisionTrigger::ToolFailure],
};

let engine = AgentBuilder::new("task")
    .openai("gpt-4o")
    .planning_mode(mode)
    .build()?;
```

### 8. Dynamic Tool Composition

Synthesize new composite tools at runtime by chaining existing primitive tools:

```rust
use agent_b::tool_synthesis::{CompositionConfig, CompositeToolSpec, ToolPipelineStep};

let config = CompositionConfig { max_composite_tools: 5, max_pipeline_depth: 3 };

let engine = AgentBuilder::new("task")
    .add_tool(search_tool)
    .add_tool(summarize_tool)
    .tool_composition(config)
    .build()?;

// The agent can now create a tool combining search -> summarize
```
