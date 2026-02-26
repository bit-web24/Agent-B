# Advanced Usage

---

## Custom State Handlers

You can replace or extend any default state by implementing the `AgentState` trait.

### Example: A State That Logs to an External System

```rust
use agentsm::states::AgentState;
use agentsm::{Event, AgentMemory, ToolRegistry, LlmCaller};
use agentsm::states::PlanningState;

/// Wraps PlanningState and adds external event publishing
pub struct ObservabilityPlanningState {
    inner:    PlanningState,
    event_tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl AgentState for ObservabilityPlanningState {
    fn name(&self) -> &'static str { "Planning" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        tools:  &ToolRegistry,
        llm:    &dyn LlmCaller,
    ) -> Event {
        let event = self.inner.handle(memory, tools, llm);

        // Publish to external system
        let _ = self.event_tx.send(format!(
            "step={} event={} task={}",
            memory.step, event, &memory.task[..50.min(memory.task.len())]
        ));

        event
    }
}
```

### Registering Custom Handlers

Use `AgentBuilder::build_with_handlers()` or construct `AgentEngine` directly:

```rust
use std::collections::HashMap;
use agentsm::states::AgentState;

// Option A: build_with_handlers()
let mut custom: HashMap<&'static str, Box<dyn AgentState>> = HashMap::new();
custom.insert("Planning", Box::new(my_custom_planning_state));

let engine = AgentBuilder::new("task")
    .llm(llm)
    .build_with_handlers(custom)?;

// Option B: AgentEngine::new() directly (maximum control)
use agentsm::{AgentEngine, AgentMemory, ToolRegistry};
use agentsm::transitions::build_transition_table;

let memory  = AgentMemory::new("task");
let tools   = ToolRegistry::new();
let mut handlers: HashMap<&'static str, Box<dyn AgentState>> = HashMap::new();
// register all 7 handlers...

let engine = AgentEngine::new(memory, tools, llm, build_transition_table(), handlers);
```

---

## Custom Transition Tables

Add new (State, Event) pairs to the default table, or build a completely custom one:

```rust
use agentsm::transitions::build_transition_table;
use agentsm::{State, Event};

// Extend the default table
let mut table = build_transition_table();

// Example: make AnswerTooShort go to Reflecting instead of back to Planning
table.insert((State::Planning, Event::AnswerTooShort), State::Reflecting);

// Build engine with the custom table
let engine = AgentEngine::new(memory, tools, llm, table, handlers);
```

### Adding New States and Events

To add a completely new state:

1. Create a new `enum` variant in a fork, or add it to `State` in `types.rs`
2. Implement `AgentState` for your new struct
3. Add the new event variants to `events.rs`
4. Add the transitions to your custom table
5. Register the handler in the handler map

---

## Multi-Agent Patterns

Each `AgentEngine` is a self-contained value. Coordinate multiple agents via channels:

```rust
use tokio::sync::mpsc;

async fn run_parallel_agents() -> anyhow::Result<()> {
    let (tx1, mut rx1) = mpsc::channel::<String>(1);
    let (tx2, mut rx2) = mpsc::channel::<String>(1);

    // Agent 1: Research
    let research_future = tokio::spawn(async move {
        let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));
        let mut engine = AgentBuilder::new("Research the current state of LLM benchmarks")
            .task_type("research")
            .llm(llm)
            .tool("search", "...", search_schema, search_fn)
            .build()
            .expect("build failed");
        let result = engine.run().unwrap_or_else(|e| e.to_string());
        tx1.send(result).await.ok();
    });

    // Agent 2: Summarizer (waits for research result)
    let research_result = rx1.recv().await.unwrap_or_default();

    let summarizer_future = tokio::spawn(async move {
        let task = format!("Summarize this research in 3 bullet points:\n{}", research_result);
        let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));
        let mut engine = AgentBuilder::new(task)
            .task_type("default")
            .llm(llm)
            .build()
            .expect("build failed");
        let result = engine.run().unwrap_or_else(|e| e.to_string());
        tx2.send(result).await.ok();
    });

    tokio::try_join!(research_future, summarizer_future)?;
    let summary = rx2.recv().await.unwrap_or_default();
    println!("Final summary:\n{}", summary);

    Ok(())
}
```

### Orchestrator Pattern

```rust
// One orchestrator agent decides which specialist agent to invoke
// Specialist agents are called as "tools" from the orchestrator's perspective

let orchestrator = AgentBuilder::new("Complex multi-step task")
    .llm(main_llm)
    .tool(
        "research_agent",
        "Delegate a research sub-task to a specialist research agent. \
         Returns a detailed research report.",
        json!({ "type": "object", "properties": {
            "sub_task": { "type": "string", "description": "The research question to answer" }
        }, "required": ["sub_task"] }),
        Box::new(|args| {
            let sub_task = args["sub_task"].as_str().unwrap_or("");
            // Spin up a sub-agent and run it synchronously
            let rt = tokio::runtime::Handle::current();
            let result = rt.block_on(async {
                // ... create and run sub-agent ...
                Ok::<String, AgentError>("research result".to_string())
            });
            result.map_err(|e| e.to_string())
        }),
    )
    .build()?;
```

---

## Accessing Memory After a Run

After `engine.run()`, inspect the full memory:

```rust
let result = engine.run();

// Final answer (if successful)
println!("Answer: {:?}", engine.memory.final_answer);

// Error message (if failed)
println!("Error: {:?}", engine.memory.error);

// Number of planning steps used
println!("Steps used: {}/{}", engine.memory.step, engine.memory.config.max_steps);

// Full history of tool calls
for entry in &engine.memory.history {
    println!("Step {}: {} → {} ({})",
        entry.step,
        entry.tool.name,
        &entry.observation[..entry.observation.len().min(50)],
        if entry.success { "✓" } else { "✗" }
    );
}

// Full trace as JSON
let trace_json = engine.trace().to_json();
std::fs::write("trace.json", trace_json).ok();
```

---

## Checkpointing & Persistence

`agentsm-rs` supports session persistence, allowing you to stop and resume agents across process restarts. This is critical for long-running workflows or crash recovery.

### Using the SQLite Store

```rust
use agentsm::persistence::SqliteCheckpointStore;

let store = SqliteCheckpointStore::new("agents.db").await?;

let mut engine = AgentBuilder::new("Long task")
    .openai(key)
    .checkpoint_store(Arc::new(store))
    .session_id("session-123") // Unique ID for this agent run
    .build()?;

engine.run().await?;
```

### Resuming an Agent

```rust
let mut engine = AgentBuilder::new("Long task") // Task must match (or will be updated from store)
    .openai(key)
    .checkpoint_store(Arc::new(store))
    .resume("session-123") // Load state from last checkpoint
    .build()?;
```

**Supported Stores:**
- `MemoryCheckpointStore`: Volatile, thread-safe (useful for tests).
- `FileCheckpointStore`: Simple JSON files in a directory.
- `SqliteCheckpointStore`: Robust, production-grade persistence.

---

## Human-in-the-Loop (HITL)

High-risk actions (spending money, deleting data) often require human approval. `agentsm-rs` integrates this directly into the state machine.

### Approval Policies

Define when a human must be consulted:

```rust
use agentsm::approval::ApprovalPolicy;

let policy = ApprovalPolicy::new()
    .require_for("delete_file")
    .require_for("send_transaction")
    .always_for_tool("expensive_research");

let mut engine = AgentBuilder::new("task")
    .openai(key)
    .approval_policy(policy)
    .on_approval(|request| {
        println!("Approval required for tool: {}", request.tool.name);
        println!("Arguments: {:?}", request.tool.args);
        // In a real app, you might send a notification and wait
        Ok(ApprovalAction::Approve)
    })
    .build()?;
```

### Approval Actions:
- `Approve`: Execution proceeds with the original arguments.
- `Reject`: Tool is skipped; the agent receives an "Access Denied" observation.
- `Modify`: arguments are updated by the human before execution.

When approval is required, the agent transitions to `WaitingForHuman` state.

---

## Custom Memory Extension

`AgentMemory` is a plain struct with all public fields. For advanced scenarios, wrap it:

```rust
pub struct MyAgentContext {
    pub memory:    AgentMemory,
    pub user_id:   String,
    pub session:   uuid::Uuid,
    pub cost_usd:  f64,        // accumulated API cost
}
```

Then pass `&mut context.memory` to custom state handlers that also take `&mut MyAgentContext`. This lets you track cross-cutting concerns without modifying the library.

---

## Replacing History Compression

The default `ReflectingState` uses a simple string summary. Override it for LLM-based compression:

```rust
pub struct LlmReflectingState;

impl AgentState for LlmReflectingState {
    fn name(&self) -> &'static str { "Reflecting" }

    fn handle(&self, memory: &mut AgentMemory, _tools: &ToolRegistry, llm: &dyn LlmCaller) -> Event {
        let history_json = serde_json::to_string_pretty(&memory.history)
            .unwrap_or_else(|_| "[]".to_string());

        // Build a compression prompt
        let compression_task = format!(
            "Summarize these tool call results into a concise paragraph preserving all key facts.\n\
             Task: {}\nHistory:\n{}",
            memory.task, history_json
        );

        // Temporarily swap in the compression task for the LLM call
        let saved_task = std::mem::replace(&mut memory.task, compression_task);
        let saved_history = std::mem::take(&mut memory.history);

        let summary = match llm.call(memory, &ToolRegistry::new(), "claude-haiku-4-5-20251001") {
            Ok(LlmResponse::FinalAnswer { content }) => content,
            _ => {
                // Compression failed — restore and continue
                memory.task = saved_task;
                memory.history = saved_history;
                memory.log("Reflecting", "COMPRESS_FAILED", "LLM compression failed, keeping history");
                return Event::ReflectDone;
            }
        };

        memory.task = saved_task;
        memory.history = vec![HistoryEntry {
            step:        memory.step,
            tool:        ToolCall { name: "[SUMMARY]".to_string(), args: HashMap::new() },
            observation: summary,
            success:     true,
        }];
        memory.retry_count = 0;
        memory.log("Reflecting", "COMPRESS_DONE", "LLM-based compression complete");

        Event::ReflectDone
    }
}
```
