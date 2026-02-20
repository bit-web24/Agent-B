# agentsm-rs

> **A production-grade, zero-framework Agentic AI library for Rust built on the Hybrid State Machine pattern.**
>
> Transition Table owns the graph. State Traits own the behavior. You own everything.

---

## Table of Contents

1. [Vision & Design Philosophy](#1-vision--design-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Complete File Specifications](#4-complete-file-specifications)
   - 4.1 [Cargo.toml](#41-cargotoml)
   - 4.2 [src/lib.rs](#42-srclibrs)
   - 4.3 [src/types.rs](#43-srctypesrs)
   - 4.4 [src/memory.rs](#44-srcmemoryrs)
   - 4.5 [src/events.rs](#45-srceventsrs)
   - 4.6 [src/states/mod.rs](#46-srcstatessmodrs)
   - 4.7 [src/states/idle.rs](#47-srcstatessidlers)
   - 4.8 [src/states/planning.rs](#48-srcstatesplanningrs)
   - 4.9 [src/states/acting.rs](#49-srcstatesactingrs)
   - 4.10 [src/states/observing.rs](#410-srcstatesobservingrs)
   - 4.11 [src/states/reflecting.rs](#411-srcstatesreflectingrs)
   - 4.12 [src/states/done.rs](#412-srcstatessdoners)
   - 4.13 [src/states/error.rs](#413-srcstateserrorrs)
   - 4.14 [src/transitions.rs](#414-srctransitionsrs)
   - 4.15 [src/tools.rs](#415-srctoolsrs)
   - 4.16 [src/llm/mod.rs](#416-srcllmmodrs)
   - 4.17 [src/llm/openai.rs](#417-srcllmopenairs)
   - 4.18 [src/llm/anthropic.rs](#418-srcllmanthropicrs)
   - 4.19 [src/llm/mock.rs](#419-srcllmmockrs)
   - 4.20 [src/engine.rs](#420-srcenginrs)
   - 4.21 [src/trace.rs](#421-srctracers)
   - 4.22 [src/error.rs](#422-srcerrorrs)
   - 4.23 [src/builder.rs](#423-srcbuilderrs)
   - 4.24 [examples/basic_agent.rs](#424-examplesbasic_agentrs)
   - 4.25 [examples/multi_tool_agent.rs](#425-examplesmulti_tool_agentrs)
   - 4.26 [examples/anthropic_agent.rs](#426-examplesanthropic_agentrs)
   - 4.27 [tests/integration_tests.rs](#427-testsintegration_testsrs)
5. [Data Flow & Execution Model](#5-data-flow--execution-model)
6. [Type Contracts](#6-type-contracts)
7. [Error Handling Strategy](#7-error-handling-strategy)
8. [Testing Strategy](#8-testing-strategy)
9. [Extension Points](#9-extension-points)
10. [Invariants & Rules the Agent Must Never Violate](#10-invariants--rules-the-agent-must-never-violate)

---

## 1. Vision & Design Philosophy

### What This Library Is

`agentsm-rs` is a Rust library for building production Agentic AI systems using a **Hybrid State Machine pattern**. It provides:

- A typed, compile-time-verified state machine engine for agent control flow
- A clean abstraction over LLM providers (OpenAI, Anthropic, any OpenAI-compatible)
- A typed tool registry with schema generation
- Full event-sourcing trace for every agent run
- Zero magic — every behavior is explicit code you own

### What This Library Is NOT

- Not a framework. It does not own your agent's behavior.
- Not opinionated about prompts. Prompt construction belongs to the caller.
- Not a wrapper that hides LLM APIs behind abstraction soup.
- Not async-only. Sync wrappers must be available for all public APIs.

### The Three Laws of This Library

**Law 1 — The Transition Table is the Single Source of Truth.**
Every legal state transition lives in one `HashMap`. If a transition is not in the table, it cannot happen. No exceptions.

**Law 2 — Failure is Data, Not Exceptions.**
Tool failures, LLM errors, and invalid responses must be fed back into the agent's context as observations. The agent decides what to do next. Only truly unrecoverable errors (no handler registered, corrupted memory) should terminate the loop via `Result::Err`.

**Law 3 — The Engine is Dumb.**
The `AgentEngine` struct knows nothing about what states do or why. It only knows: get current state's handler → call handle() → get event → look up next state in table → repeat. No business logic ever lives in the engine.

---

## 2. Architecture Overview

### Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PUBLIC API                           │
│              AgentBuilder  (builder.rs)                 │
└────────────────────────┬────────────────────────────────┘
                         │ builds
┌────────────────────────▼────────────────────────────────┐
│                   AgentEngine  (engine.rs)              │
│   - owns AgentMemory                                    │
│   - owns ToolRegistry                                   │
│   - owns Box<dyn LlmCaller>                             │
│   - owns TransitionTable                                │
│   - owns StateHandlerMap                                │
│   - runs the loop                                       │
└──────┬──────────────────────┬──────────────────────────┘
       │                      │
       ▼                      ▼
┌─────────────┐    ┌──────────────────────┐
│ Transition  │    │   State Handlers     │
│   Table     │    │  (states/*.rs)       │
│ (table.rs)  │    │                      │
│             │    │  IdleState           │
│ HashMap<    │    │  PlanningState       │
│  (State,    │    │  ActingState         │
│   Event),   │    │  ObservingState      │
│  State>     │    │  ReflectingState     │
│             │    │  DoneState           │
└─────────────┘    │  ErrorState          │
                   └──────────┬───────────┘
                              │ uses
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
      ┌──────────────┐ ┌───────────┐ ┌──────────────────┐
      │ AgentMemory  │ │   Tools   │ │   LlmCaller      │
      │ (memory.rs)  │ │(tools.rs) │ │  (llm/*.rs)      │
      │              │ │           │ │                  │
      │ - task       │ │ - registry│ │ OpenAiCaller     │
      │ - history    │ │ - schema  │ │ AnthropicCaller  │
      │ - trace      │ │ - execute │ │ MockLlmCaller    │
      └──────────────┘ └───────────┘ └──────────────────┘
```

### State Machine Diagram

```
                    ┌──────┐
                    │ IDLE │
                    └──┬───┘
                       │ Start
                       ▼
              ┌──────────────────┐
         ┌───▶│    PLANNING      │◀──────────────────────────┐
         │    └────────┬─────────┘                           │
         │             │                                     │
         │    ┌────────┴──────────────────────┐             │
         │    │                               │             │
         │    ▼ LlmToolCall                   ▼ LlmFinalAns │
         │ ┌────────┐                      ┌──────┐         │
         │ │ ACTING │                      │ DONE │         │
         │ └───┬────┘                      └──────┘         │
         │     │ ToolSuccess/ToolFailure                     │
         │     ▼                                            │
         │ ┌───────────┐  NeedsReflection  ┌─────────────┐ │
         │ │ OBSERVING │─────────────────▶│  REFLECTING  │ │
         │ └─────┬─────┘                  └──────┬───────┘ │
         │       │ Continue                      │ ReflectDone
         └───────┘                               └──────────┘
         
         MaxSteps / FatalError from any state ──▶ ERROR (terminal)
```

---

## 3. Project Structure

The agent building this library must create **exactly** this file tree:

```
agentsm-rs/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── types.rs
│   ├── memory.rs
│   ├── events.rs
│   ├── transitions.rs
│   ├── tools.rs
│   ├── engine.rs
│   ├── trace.rs
│   ├── error.rs
│   ├── builder.rs
│   └── states/
│       ├── mod.rs
│       ├── idle.rs
│       ├── planning.rs
│       ├── acting.rs
│       ├── observing.rs
│       ├── reflecting.rs
│       ├── done.rs
│       └── error.rs
│   └── llm/
│       ├── mod.rs
│       ├── openai.rs
│       ├── anthropic.rs
│       └── mock.rs
├── examples/
│   ├── basic_agent.rs
│   ├── multi_tool_agent.rs
│   └── anthropic_agent.rs
└── tests/
    └── integration_tests.rs
```

---

## 4. Complete File Specifications

### 4.1 `Cargo.toml`

```toml
[package]
name        = "agentsm-rs"
version     = "0.1.0"
edition     = "2021"
description = "Production-grade Agentic AI library using Hybrid State Machine pattern"
license     = "MIT"
repository  = "https://github.com/yourusername/agentsm-rs"
keywords    = ["ai", "agent", "state-machine", "llm", "anthropic"]
categories  = ["science", "asynchronous"]

[lib]
name = "agentsm"
path = "src/lib.rs"

[[example]]
name = "basic_agent"
path = "examples/basic_agent.rs"

[[example]]
name = "multi_tool_agent"
path = "examples/multi_tool_agent.rs"

[[example]]
name = "anthropic_agent"
path = "examples/anthropic_agent.rs"

[dependencies]
# Async runtime
tokio        = { version = "1",    features = ["full"] }

# Serialization — used for tool schemas and LLM API payloads
serde        = { version = "1",    features = ["derive"] }
serde_json   = "1"

# LLM provider — OpenAI + all OpenAI-compatible APIs
async-openai = "0.23"

# HTTP client — Anthropic API (no official Rust SDK yet)
reqwest      = { version = "0.12", features = ["json"] }

# Error handling
anyhow       = "1"
thiserror    = "1"

# Structured logging
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Time
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tokio   = { version = "1",    features = ["full", "test-util"] }
mockall = "0.12"

[features]
default  = ["openai", "anthropic"]
openai   = []
anthropic = []
```

---

### 4.2 `src/lib.rs`

**Purpose:** Public API surface. Re-exports everything the user needs to build an agent without reaching into submodules.

**Must export:**
```rust
pub mod types;
pub mod memory;
pub mod events;
pub mod transitions;
pub mod tools;
pub mod engine;
pub mod trace;
pub mod error;
pub mod builder;
pub mod states;
pub mod llm;

// Convenience re-exports at crate root
pub use builder::AgentBuilder;
pub use engine::AgentEngine;
pub use memory::AgentMemory;
pub use types::{State, LlmResponse, ToolCall, HistoryEntry};
pub use events::Event;
pub use tools::{ToolRegistry, ToolFn};
pub use llm::{LlmCaller, LlmCallerExt};
pub use trace::{TraceEntry, Trace};
pub use error::AgentError;
```

---

### 4.3 `src/types.rs`

**Purpose:** All shared data types. No logic. Pure data.

**Must define:**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// All possible agent states.
/// Adding a variant here will cause compile errors in every
/// non-exhaustive match — intentional, by design.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum State {
    Idle,
    Planning,
    Acting,
    Observing,
    Reflecting,
    Done,
    Error,
}

impl State {
    /// Returns true if the state is terminal (loop must exit)
    pub fn is_terminal(&self) -> bool {
        matches!(self, State::Done | State::Error)
    }

    /// Returns the static string name — used for handler map lookup
    pub fn as_str(&self) -> &'static str {
        match self {
            State::Idle       => "Idle",
            State::Planning   => "Planning",
            State::Acting     => "Acting",
            State::Observing  => "Observing",
            State::Reflecting => "Reflecting",
            State::Done       => "Done",
            State::Error      => "Error",
        }
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A tool invocation requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, serde_json::Value>,
}

/// A completed tool invocation stored in history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub step:        usize,
    pub tool:        ToolCall,
    pub observation: String,
    pub success:     bool,
}

/// What the LLM can return. Always one of these two variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmResponse {
    /// LLM wants to invoke a tool
    ToolCall {
        tool:       ToolCall,
        confidence: f64,      // 0.0 - 1.0, estimated from response metadata
    },
    /// LLM produced a final answer — task is complete
    FinalAnswer {
        content: String,
    },
}

/// Configuration for the agent's planning behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Hard cap on number of planning/acting cycles
    pub max_steps: usize,

    /// How many retries before aborting on low confidence
    pub max_retries: usize,

    /// Confidence threshold below which reflection is triggered
    pub confidence_threshold: f64,

    /// Compress history every N steps (0 = never)
    pub reflect_every_n_steps: usize,

    /// Minimum answer length in characters
    pub min_answer_length: usize,
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

### 4.4 `src/memory.rs`

**Purpose:** The agent's working memory. Passed by mutable reference to every state handler. This is the agent's entire world-state.

**Must define:**

```rust
use crate::types::{ToolCall, HistoryEntry, AgentConfig};
use crate::trace::{TraceEntry, Trace};
use chrono::Utc;
use std::collections::HashSet;

#[derive(Debug)]
pub struct AgentMemory {
    // ── Task definition ──────────────────────────────────
    /// The original task description
    pub task:               String,
    /// Classifies task for model selection ("research", "calculation", "default")
    pub task_type:          String,
    /// The system prompt to prepend to every LLM call
    pub system_prompt:      String,

    // ── Execution state ──────────────────────────────────
    /// Current step number (incremented at start of each Planning cycle)
    pub step:               usize,
    /// Number of low-confidence retries consumed
    pub retry_count:        usize,
    /// Last recorded confidence score from LLM
    pub confidence_score:   f64,

    // ── Tool call lifecycle ──────────────────────────────
    /// Set by PlanningState when LLM requests a tool, consumed by ActingState
    pub current_tool_call:  Option<ToolCall>,
    /// Set by ActingState after tool execution, consumed by ObservingState
    pub last_observation:   Option<String>,

    // ── History and results ──────────────────────────────
    /// Ordered list of completed tool calls and their observations
    pub history:            Vec<HistoryEntry>,
    /// Set when LLM produces a final answer
    pub final_answer:       Option<String>,
    /// Set when agent encounters an unrecoverable error
    pub error:              Option<String>,

    // ── Configuration ────────────────────────────────────
    pub config:             AgentConfig,
    /// Tools the agent is not permitted to call
    pub blacklisted_tools:  HashSet<String>,

    // ── Observability ────────────────────────────────────
    /// Full event-sourcing log — every state transition recorded here
    pub trace:              Trace,
}

impl AgentMemory {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task:              task.into(),
            task_type:         "default".to_string(),
            system_prompt:     String::new(),
            step:              0,
            retry_count:       0,
            confidence_score:  1.0,
            current_tool_call: None,
            last_observation:  None,
            history:           Vec::new(),
            final_answer:      None,
            error:             None,
            config:            AgentConfig::default(),
            blacklisted_tools: HashSet::new(),
            trace:             Trace::new(),
        }
    }

    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = task_type.into();
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn blacklist_tool(&mut self, tool_name: impl Into<String>) {
        self.blacklisted_tools.insert(tool_name.into());
    }

    /// Records an event into the trace log. Called by all state handlers.
    pub fn log(&mut self, state: &str, event: &str, data: &str) {
        tracing::debug!(state, event, data, step = self.step, "agent trace");
        self.trace.record(TraceEntry {
            step:      self.step,
            state:     state.to_string(),
            event:     event.to_string(),
            data:      data.to_string(),
            timestamp: Utc::now(),
        });
    }

    /// Builds the messages array to send to the LLM.
    /// Implementors: include system prompt, compressed history summary (if any),
    /// recent history entries, and the current task as the last user message.
    pub fn build_messages(&self) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // System message
        if !self.system_prompt.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": self.system_prompt
            }));
        }

        // History as assistant/tool messages
        for entry in &self.history {
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": format!("Called tool '{}' with args {:?}",
                    entry.tool.name, entry.tool.args)
            }));
            messages.push(serde_json::json!({
                "role": "user",
                "content": format!("Tool result: {}", entry.observation)
            }));
        }

        // Current task
        messages.push(serde_json::json!({
            "role": "user",
            "content": &self.task
        }));

        messages
    }
}
```

---

### 4.5 `src/events.rs`

**Purpose:** All events that drive state transitions. Exhaustive matching enforced by compiler.

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Event {
    // ── Lifecycle ───────────────────────────────────────
    /// Emitted by IdleState — starts the loop
    Start,

    // ── Planning outcomes ───────────────────────────────
    /// LLM returned a tool call with sufficient confidence
    LlmToolCall,
    /// LLM returned a final answer of acceptable quality
    LlmFinalAnswer,
    /// Step counter hit max_steps
    MaxSteps,
    /// LLM confidence below threshold, retry budget remaining
    LowConfidence,
    /// LLM final answer too short, needs retry
    AnswerTooShort,
    /// LLM requested a blacklisted tool
    ToolBlacklisted,
    /// LLM API call failed unrecoverably
    FatalError,

    // ── Acting outcomes ─────────────────────────────────
    /// Tool executed and returned a result
    ToolSuccess,
    /// Tool execution raised an error (error becomes observation data)
    ToolFailure,

    // ── Observing outcomes ──────────────────────────────
    /// Normal flow: proceed back to planning
    Continue,
    /// Step count triggers reflection/compression
    NeedsReflection,

    // ── Reflecting outcomes ─────────────────────────────
    /// Compression complete, proceed to planning
    ReflectDone,
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
```

---

### 4.6 `src/states/mod.rs`

**Purpose:** Declares the `AgentState` trait and re-exports all concrete state implementations.

```rust
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;
use crate::error::AgentError;

mod idle;
mod planning;
mod acting;
mod observing;
mod reflecting;
mod done;
mod error;

pub use idle::IdleState;
pub use planning::PlanningState;
pub use acting::ActingState;
pub use observing::ObservingState;
pub use reflecting::ReflectingState;
pub use done::DoneState;
pub use error::ErrorState;

/// The contract every state must fulfill.
///
/// # Implementing a State
///
/// 1. `handle()` performs the state's work using only `memory`, `tools`, and `llm`.
/// 2. `handle()` MUST return an Event — never panic, never return nothing.
/// 3. If work succeeds, return the success Event.
/// 4. If work fails non-fatally (tool error, bad LLM output), set
///    `memory.last_observation` or `memory.error` and return the
///    appropriate failure Event. Do NOT return Err — failure is data.
/// 5. Only return a Result::Err for truly unrecoverable situations
///    (e.g., no handler registered, memory corrupted).
/// 6. Always call `memory.log()` at least once per handle() call.
///
pub trait AgentState: Send + Sync {
    /// Returns the unique string name of this state.
    /// Must match the key used in the engine's handler map.
    fn name(&self) -> &'static str;

    /// Execute this state's logic. Returns the Event that drives
    /// the next transition lookup in the transition table.
    fn handle(
        &self,
        memory: &mut AgentMemory,
        tools:  &ToolRegistry,
        llm:    &dyn LlmCaller,
    ) -> Event;
}
```

---

### 4.7 `src/states/idle.rs`

**Purpose:** Entry point. Only logs the start event and emits `Event::Start`.

**Rules:**
- No LLM calls
- No tool calls
- Only sets up trace log entry
- Always transitions to Planning via `Event::Start`

```rust
use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;

pub struct IdleState;

impl AgentState for IdleState {
    fn name(&self) -> &'static str { "Idle" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        _tools: &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        memory.log("Idle", "AGENT_STARTED", &format!(
            "task='{}' task_type='{}' max_steps={}",
            memory.task, memory.task_type, memory.config.max_steps
        ));
        Event::Start
    }
}
```

---

### 4.8 `src/states/planning.rs`

**Purpose:** The most complex state. Calls the LLM, validates the response, handles all planning-level decisions.

**Rules:**
- Increment `memory.step` at the BEGINNING of handle()
- Check `memory.step >= memory.config.max_steps` BEFORE calling LLM
- Select model based on `memory.task_type`
- On LLM error: set `memory.error`, return `Event::FatalError`
- On low confidence (below threshold AND retry budget remaining): increment `memory.retry_count`, return `Event::LowConfidence`
- On tool call to blacklisted tool: log it, return `Event::ToolBlacklisted` (do NOT set memory.error)
- On final answer shorter than `config.min_answer_length`: return `Event::AnswerTooShort`
- On valid tool call: set `memory.current_tool_call`, return `Event::LlmToolCall`
- On valid final answer: set `memory.final_answer`, return `Event::LlmFinalAnswer`

**Model selection logic:**
```
task_type == "research"    → "claude-opus-4-6"      or "gpt-4o"
task_type == "calculation" → "claude-haiku-4-5-20251001" or "gpt-4o-mini"
task_type == "default"     → "claude-sonnet-4-6"    or "gpt-4o"
```

The LLM caller receives the model string — it is responsible for routing to the right provider. The planning state only decides which model tier to request.

**Full implementation signature:**
```rust
pub struct PlanningState;

impl AgentState for PlanningState {
    fn name(&self) -> &'static str { "Planning" }
    fn handle(&self, memory: &mut AgentMemory, tools: &ToolRegistry, llm: &dyn LlmCaller) -> Event {
        // 1. Guard: max steps
        // 2. Increment step
        // 3. Select model
        // 4. Call llm.call(memory, tools, model)
        // 5. Handle Result::Err → FatalError
        // 6. Match LlmResponse:
        //    - ToolCall: check confidence, check blacklist, set current_tool_call
        //    - FinalAnswer: check length, set final_answer
    }
}

impl PlanningState {
    fn select_model(&self, task_type: &str) -> &'static str { ... }
    fn handle_tool_call(&self, memory: &mut AgentMemory, tool: ToolCall, confidence: f64) -> Event { ... }
    fn handle_final_answer(&self, memory: &mut AgentMemory, content: String) -> Event { ... }
}
```

---

### 4.9 `src/states/acting.rs`

**Purpose:** Executes the tool call stored in `memory.current_tool_call`. Records result as `memory.last_observation`.

**Rules:**
- If `memory.current_tool_call` is `None`: set `memory.error`, return `Event::FatalError`
- If tool name not in registry: set `memory.last_observation` to error string, return `Event::ToolFailure` (NOT FatalError — this is recoverable data)
- On tool execution success: prefix result with `"SUCCESS: "`, set `memory.last_observation`, return `Event::ToolSuccess`
- On tool execution error: prefix with `"ERROR: {type}: {msg}"`, set `memory.last_observation`, return `Event::ToolFailure`
- **CRITICAL:** Tool failure NEVER crashes the loop. The error string becomes an observation the LLM can reason about.
- Log tool name and args before execution, log outcome after

---

### 4.10 `src/states/observing.rs`

**Purpose:** Commits the completed tool result into `memory.history`. Decides whether to reflect.

**Rules:**
- Always commit `(current_tool_call, last_observation)` as a new `HistoryEntry` into `memory.history`
- Clear `memory.current_tool_call` and `memory.last_observation` after committing
- If `memory.step % memory.config.reflect_every_n_steps == 0` AND `reflect_every_n_steps > 0`: return `Event::NeedsReflection`
- Otherwise: return `Event::Continue`
- Do NOT call LLM here
- Do NOT execute tools here

---

### 4.11 `src/states/reflecting.rs`

**Purpose:** Compresses growing history to keep the context window lean.

**Rules:**
- Summarize `memory.history` into a single `HistoryEntry` with `tool.name = "[SUMMARY]"`
- In production: call the LLM with the history and request a compression summary
- The compressed summary replaces ALL existing history entries (memory.history = vec![summary_entry])
- Reset `memory.retry_count = 0` after reflection (fresh retry budget for next phase)
- Always return `Event::ReflectDone`
- If LLM compression call fails: log the failure but DO NOT fail the agent — keep existing history and return `Event::ReflectDone` anyway

**Compression prompt template (use this exact prompt):**
```
Summarize the following tool call history into a single concise paragraph
that preserves all key facts, findings, and data needed to continue the task.
Task: {task}
History: {history_json}
```

---

### 4.12 `src/states/done.rs`

**Purpose:** Terminal state. Logs completion. Does nothing else.

**Rules:**
- Log the final answer (truncated to 100 chars in the log)
- Return `Event::Start` (will never be used — engine exits before re-entering)
- Do NOT modify any memory fields

---

### 4.13 `src/states/error.rs`

**Purpose:** Terminal error state. Logs failure reason. Does nothing else.

**Rules:**
- Log `memory.error` (or "Unknown error" if None)
- Return `Event::Start` (will never be used — engine exits before re-entering)
- Do NOT modify any memory fields

---

### 4.14 `src/transitions.rs`

**Purpose:** Builds and owns the transition table. The single source of truth for all legal state transitions.

**Must define:**

```rust
use std::collections::HashMap;
use crate::types::State;
use crate::events::Event;

pub type TransitionTable = HashMap<(State, Event), State>;

/// Builds the complete, immutable transition table.
/// This function defines ALL legal behaviors of the agent.
/// Any (State, Event) pair not in this table is illegal and
/// will cause AgentEngine::run() to return AgentError::InvalidTransition.
pub fn build_transition_table() -> TransitionTable {
    let mut t = HashMap::new();

    // ── IDLE ─────────────────────────────────────────────
    t.insert((State::Idle,       Event::Start),           State::Planning);

    // ── PLANNING ─────────────────────────────────────────
    t.insert((State::Planning,   Event::LlmToolCall),     State::Acting);
    t.insert((State::Planning,   Event::LlmFinalAnswer),  State::Done);
    t.insert((State::Planning,   Event::MaxSteps),        State::Error);
    t.insert((State::Planning,   Event::LowConfidence),   State::Reflecting);
    t.insert((State::Planning,   Event::AnswerTooShort),  State::Planning);
    t.insert((State::Planning,   Event::ToolBlacklisted), State::Planning);
    t.insert((State::Planning,   Event::FatalError),      State::Error);

    // ── ACTING ───────────────────────────────────────────
    t.insert((State::Acting,     Event::ToolSuccess),     State::Observing);
    t.insert((State::Acting,     Event::ToolFailure),     State::Observing);
    t.insert((State::Acting,     Event::FatalError),      State::Error);

    // ── OBSERVING ────────────────────────────────────────
    t.insert((State::Observing,  Event::Continue),        State::Planning);
    t.insert((State::Observing,  Event::NeedsReflection), State::Reflecting);

    // ── REFLECTING ───────────────────────────────────────
    t.insert((State::Reflecting, Event::ReflectDone),     State::Planning);

    // Note: DONE and ERROR are terminal — no outgoing transitions.
    // Engine checks State::is_terminal() and exits before table lookup.

    t
}

/// Validates that a given (state, event) pair is legal.
pub fn is_valid_transition(table: &TransitionTable, state: &State, event: &Event) -> bool {
    table.contains_key(&(state.clone(), event.clone()))
}
```

---

### 4.15 `src/tools.rs`

**Purpose:** Tool registration, schema management, and execution.

**Must define:**

```rust
use std::collections::HashMap;
use serde_json::Value;
use crate::error::AgentError;

/// A tool function: takes JSON args, returns string result or error string.
/// Box<dyn Fn> — heap-allocated, Send + Sync for thread safety.
pub type ToolFn = Box<dyn Fn(&HashMap<String, Value>) -> Result<String, String> + Send + Sync>;

/// Tool schema for sending to LLM (OpenAI / Anthropic tool format)
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolSchema {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,   // JSON Schema object
}

/// Registered tool entry
struct ToolEntry {
    schema: ToolSchema,
    func:   ToolFn,
}

pub struct ToolRegistry {
    tools: HashMap<String, ToolEntry>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    /// Register a tool with its schema and implementation.
    ///
    /// # Arguments
    /// * `name`        - Unique tool name (must match schema name)
    /// * `description` - Clear description of what this tool does and when to use it
    /// * `schema`      - JSON Schema for the input parameters
    /// * `func`        - The actual implementation
    pub fn register(
        &mut self,
        name:        impl Into<String>,
        description: impl Into<String>,
        schema:      Value,
        func:        ToolFn,
    ) {
        let name = name.into();
        self.tools.insert(name.clone(), ToolEntry {
            schema: ToolSchema {
                name:         name.clone(),
                description:  description.into(),
                input_schema: schema,
            },
            func,
        });
    }

    /// Execute a named tool with given arguments.
    /// Returns Ok(result_string) or Err(error_string).
    /// Never panics — all errors are captured as Err variants.
    pub fn execute(&self, name: &str, args: &HashMap<String, Value>) -> Result<String, String> {
        match self.tools.get(name) {
            Some(entry) => (entry.func)(args),
            None        => Err(format!("Tool '{}' not found in registry", name)),
        }
    }

    /// Returns true if a tool with this name is registered.
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns all tool schemas — used to build the tools array for LLM calls.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|e| e.schema.clone()).collect()
    }

    /// Returns the count of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}
```

---

### 4.16 `src/llm/mod.rs`

**Purpose:** The `LlmCaller` trait and re-exports. This is the boundary between the state machine and the network.

```rust
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::LlmResponse;
use async_trait::async_trait;

mod openai;
mod anthropic;
mod mock;

pub use openai::OpenAiCaller;
pub use anthropic::AnthropicCaller;
pub use mock::MockLlmCaller;

/// The single interface between the state machine and any LLM provider.
///
/// # Contract
/// - Must be Send + Sync (used behind Box<dyn LlmCaller>)
/// - Returns Ok(LlmResponse) on any valid LLM interaction
/// - Returns Err(String) ONLY for unrecoverable failures:
///   - Network failure after retries exhausted
///   - Authentication failure
///   - Response unparseable as LlmResponse
/// - MUST build the tool schemas from `tools.schemas()` and include
///   them in every API call
/// - MUST build messages from `memory.build_messages()`
///
pub trait LlmCaller: Send + Sync {
    fn call(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String>;
}

/// Async version of LlmCaller for async runtimes.
#[async_trait]
pub trait AsyncLlmCaller: Send + Sync {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String>;
}

/// Extension trait: wraps an AsyncLlmCaller into a sync LlmCaller
/// using tokio::runtime::Handle::block_on.
pub struct SyncWrapper<T: AsyncLlmCaller>(pub T);

impl<T: AsyncLlmCaller> LlmCaller for SyncWrapper<T> {
    fn call(&self, memory: &AgentMemory, tools: &ToolRegistry, model: &str) -> Result<LlmResponse, String> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(self.0.call_async(memory, tools, model))
    }
}

pub use SyncWrapper as LlmCallerExt;
```

---

### 4.17 `src/llm/openai.rs`

**Purpose:** Implements `AsyncLlmCaller` for OpenAI and all OpenAI-compatible APIs.

**Must implement:**

```rust
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage,
        ChatCompletionTool,
        ChatCompletionToolType,
        CreateChatCompletionRequestArgs,
        FunctionObject,
        ChatCompletionMessageToolCall,
    },
    Client,
};
use async_trait::async_trait;
use crate::llm::AsyncLlmCaller;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::{LlmResponse, ToolCall};
use std::collections::HashMap;

pub struct OpenAiCaller {
    client: Client<OpenAIConfig>,
}

impl OpenAiCaller {
    /// Standard OpenAI client using OPENAI_API_KEY env var
    pub fn new() -> Self {
        Self { client: Client::new() }
    }

    /// Custom base URL — for Groq, Together, Ollama, Fireworks, etc.
    /// api_base example: "https://api.groq.com/openai/v1"
    pub fn with_base_url(api_base: impl Into<String>, api_key: impl Into<String>) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(api_base)
            .with_api_key(api_key);
        Self { client: Client::with_config(config) }
    }

    /// Convert our ToolSchema into async-openai's ChatCompletionTool type
    fn build_tools(tools: &ToolRegistry) -> Vec<ChatCompletionTool> {
        tools.schemas().into_iter().map(|schema| {
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name:        schema.name,
                    description: Some(schema.description),
                    parameters:  Some(schema.input_schema),
                    strict:      Some(true),
                },
            }
        }).collect()
    }

    /// Parse the first tool call from an OpenAI response into our ToolCall type
    fn parse_tool_call(tc: &ChatCompletionMessageToolCall) -> Result<ToolCall, String> {
        let args: HashMap<String, serde_json::Value> =
            serde_json::from_str(&tc.function.arguments)
                .map_err(|e| format!("Failed to parse tool args: {}", e))?;
        Ok(ToolCall {
            name: tc.function.name.clone(),
            args,
        })
    }
}

#[async_trait]
impl AsyncLlmCaller for OpenAiCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        let messages_json = memory.build_messages();

        // Convert serde_json::Value messages to async-openai types
        // Use serde round-trip: serialize to string, deserialize as typed
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::Value::Array(messages_json))
                .map_err(|e| format!("Failed to build messages: {}", e))?;

        let oai_tools = Self::build_tools(tools);

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(model).messages(messages);

        if !oai_tools.is_empty() {
            request_builder.tools(oai_tools);
        }

        let request = request_builder.build()
            .map_err(|e| format!("Failed to build request: {}", e))?;

        let response = self.client.chat()
            .create(request)
            .await
            .map_err(|e| format!("OpenAI API error: {}", e))?;

        let choice = response.choices.into_iter().next()
            .ok_or("Empty response from OpenAI")?;

        let message = choice.message;

        // Tool call takes priority over text content
        if let Some(tool_calls) = message.tool_calls {
            if let Some(tc) = tool_calls.into_iter().next() {
                let tool = Self::parse_tool_call(&tc)?;
                return Ok(LlmResponse::ToolCall { tool, confidence: 1.0 });
            }
        }

        let content = message.content
            .ok_or("No content in OpenAI response")?;

        Ok(LlmResponse::FinalAnswer { content })
    }
}
```

---

### 4.18 `src/llm/anthropic.rs`

**Purpose:** Implements `AsyncLlmCaller` for the Anthropic API using raw `reqwest`. No community crate dependency.

**Must implement these types and the full caller:**

```rust
// ── Anthropic request types ──────────────────────────────

#[derive(serde::Serialize)]
struct AnthropicRequest {
    model:      String,
    max_tokens: u32,
    system:     Option<String>,
    tools:      Vec<AnthropicToolDef>,
    messages:   Vec<AnthropicMessage>,
}

#[derive(serde::Serialize)]
struct AnthropicToolDef {
    name:         String,
    description:  String,
    input_schema: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct AnthropicMessage {
    role:    String,
    content: serde_json::Value,   // string or array of content blocks
}

// ── Anthropic response types ─────────────────────────────

#[derive(serde::Deserialize, Debug)]
struct AnthropicResponse {
    content:     Vec<AnthropicContentBlock>,
    stop_reason: String,
    usage:       AnthropicUsage,
}

#[derive(serde::Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens:  u32,
    output_tokens: u32,
}

#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id:    String,
        name:  String,
        input: serde_json::Value,
    },
}

// ── Caller ───────────────────────────────────────────────

pub struct AnthropicCaller {
    client:  reqwest::Client,
    api_key: String,
    api_base: String,
}

impl AnthropicCaller {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client:   reqwest::Client::new(),
            api_key:  api_key.into(),
            api_base: "https://api.anthropic.com".to_string(),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;
        Ok(Self::new(key))
    }

    fn build_tool_defs(tools: &ToolRegistry) -> Vec<AnthropicToolDef> {
        tools.schemas().into_iter().map(|s| AnthropicToolDef {
            name:         s.name,
            description:  s.description,
            input_schema: s.input_schema,
        }).collect()
    }

    fn build_messages(memory: &AgentMemory) -> Vec<AnthropicMessage> {
        // Convert memory.build_messages() (serde_json::Value array)
        // into Vec<AnthropicMessage>
        // Filter out "system" role (sent separately in AnthropicRequest.system)
        memory.build_messages()
            .into_iter()
            .filter(|m| m["role"] != "system")
            .map(|m| AnthropicMessage {
                role:    m["role"].as_str().unwrap_or("user").to_string(),
                content: m["content"].clone(),
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl AsyncLlmCaller for AnthropicCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        let system = if memory.system_prompt.is_empty() {
            None
        } else {
            Some(memory.system_prompt.clone())
        };

        let body = AnthropicRequest {
            model:      model.to_string(),
            max_tokens: 4096,
            system,
            tools:      Self::build_tool_defs(tools),
            messages:   Self::build_messages(memory),
        };

        let response = self.client
            .post(format!("{}/v1/messages", self.api_base))
            .header("x-api-key",         &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type",      "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Network error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body   = response.text().await.unwrap_or_default();
            return Err(format!("Anthropic API error {}: {}", status, body));
        }

        let parsed: AnthropicResponse = response.json()
            .await
            .map_err(|e| format!("Failed to parse Anthropic response: {}", e))?;

        // Tool use takes priority
        for block in parsed.content {
            match block {
                AnthropicContentBlock::ToolUse { name, input, .. } => {
                    let args = serde_json::from_value(input)
                        .map_err(|e| format!("Invalid tool args: {}", e))?;
                    return Ok(LlmResponse::ToolCall {
                        tool: ToolCall { name, args },
                        confidence: 1.0,
                    });
                }
                AnthropicContentBlock::Text { text } => {
                    return Ok(LlmResponse::FinalAnswer { content: text });
                }
            }
        }

        Err("Anthropic returned empty content".to_string())
    }
}
```

---

### 4.19 `src/llm/mock.rs`

**Purpose:** A scriptable mock for testing. Returns pre-programmed responses in sequence.

```rust
use std::sync::Mutex;
use crate::llm::LlmCaller;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::{LlmResponse, ToolCall};

pub struct MockLlmCaller {
    responses: Mutex<Vec<LlmResponse>>,
    call_log:  Mutex<Vec<(String, String)>>,  // (model, memory.task)
}

impl MockLlmCaller {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            call_log:  Mutex::new(Vec::new()),
        }
    }

    /// Returns the number of times call() was invoked
    pub fn call_count(&self) -> usize {
        self.call_log.lock().unwrap().len()
    }

    /// Returns the model string passed to the Nth call (0-indexed)
    pub fn model_for_call(&self, n: usize) -> Option<String> {
        self.call_log.lock().unwrap()
            .get(n)
            .map(|(model, _)| model.clone())
    }
}

impl LlmCaller for MockLlmCaller {
    fn call(
        &self,
        memory: &AgentMemory,
        _tools: &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        self.call_log.lock().unwrap()
            .push((model.to_string(), memory.task.clone()));

        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Err("MockLlmCaller: no more programmed responses".to_string());
        }
        Ok(responses.remove(0))
    }
}
```

---

### 4.20 `src/engine.rs`

**Purpose:** The agent loop. Dumb by design. Knows nothing about what states do. Only wires table + handlers.

**Rules the engine MUST follow:**
1. Check `state.is_terminal()` BEFORE calling the handler, exit immediately if true
2. Call `handler.handle(memory, tools, llm)` to get the event
3. Look up `(current_state, event)` in the transition table
4. If not found: return `Err(AgentError::InvalidTransition)`
5. Set `self.current_state = next_state`
6. Repeat from step 1
7. On exit (terminal state): check if Done or Error and return appropriately
8. Hard cap at `memory.config.max_steps * 2` total loop iterations as safety valve

```rust
use std::collections::HashMap;
use crate::states::AgentState;
use crate::types::State;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;
use crate::transitions::TransitionTable;
use crate::trace::Trace;
use crate::error::AgentError;

pub struct AgentEngine {
    pub memory:     AgentMemory,
    pub tools:      ToolRegistry,
    pub llm:        Box<dyn LlmCaller>,
    state:          State,
    transitions:    TransitionTable,
    handlers:       HashMap<&'static str, Box<dyn AgentState>>,
}

impl AgentEngine {
    /// Creates a new engine. Prefer using AgentBuilder for ergonomic construction.
    pub fn new(
        memory:      AgentMemory,
        tools:       ToolRegistry,
        llm:         Box<dyn LlmCaller>,
        transitions: TransitionTable,
        handlers:    HashMap<&'static str, Box<dyn AgentState>>,
    ) -> Self {
        Self {
            memory,
            tools,
            llm,
            state: State::Idle,
            transitions,
            handlers,
        }
    }

    /// Run the agent to completion.
    /// Returns Ok(final_answer) or Err(AgentError).
    pub fn run(&mut self) -> Result<String, AgentError> {
        let safety_cap = self.memory.config.max_steps * 3;
        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > safety_cap {
                return Err(AgentError::SafetyCapExceeded(iterations));
            }

            tracing::info!(state = %self.state, iteration = iterations, "agent loop tick");

            // Exit condition: terminal state
            if self.state.is_terminal() {
                return match self.state {
                    State::Done  => Ok(self.memory.final_answer.clone()
                        .unwrap_or_else(|| "[No answer produced]".to_string())),
                    State::Error => Err(AgentError::AgentFailed(
                        self.memory.error.clone()
                            .unwrap_or_else(|| "Unknown error".to_string())
                    )),
                    _ => unreachable!(),
                };
            }

            // Get handler for current state
            let state_name = self.state.as_str();
            let handler = self.handlers.get(state_name)
                .ok_or_else(|| AgentError::NoHandlerForState(state_name.to_string()))?;

            // Execute state — get event
            let event = handler.handle(&mut self.memory, &self.tools, self.llm.as_ref());

            tracing::debug!(state = %self.state, event = %event, "state produced event");

            // Look up transition
            let key = (self.state.clone(), event.clone());
            let next_state = self.transitions.get(&key)
                .cloned()
                .ok_or_else(|| AgentError::InvalidTransition {
                    from:  self.state.clone(),
                    event: event.clone(),
                })?;

            tracing::info!(from = %self.state, event = %event, to = %next_state, "transition");
            println!("  ══ {} --{}--> {} ══", self.state, event, next_state);

            self.state = next_state;
        }
    }

    /// Returns a reference to the full execution trace.
    pub fn trace(&self) -> &Trace {
        &self.memory.trace
    }

    /// Returns the current state (useful for inspection after run).
    pub fn current_state(&self) -> &State {
        &self.state
    }
}
```

---

### 4.21 `src/trace.rs`

**Purpose:** Event-sourcing log. Immutable record of every agent action.

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub step:      usize,
    pub state:     String,
    pub event:     String,
    pub data:      String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    entries: Vec<TraceEntry>,
}

impl Trace {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn record(&mut self, entry: TraceEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[TraceEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all entries for a given state name
    pub fn for_state(&self, state: &str) -> Vec<&TraceEntry> {
        self.entries.iter().filter(|e| e.state == state).collect()
    }

    /// Serializes the trace to a pretty-printed JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.entries)
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// Prints a human-readable trace table to stdout
    pub fn print(&self) {
        println!("\n{:<6} {:<14} {:<28} {}", "step", "state", "event", "data");
        println!("{}", "─".repeat(80));
        for e in &self.entries {
            println!("{:<6} {:<14} {:<28} {}", e.step, e.state, e.event, &e.data.chars().take(30).collect::<String>());
        }
    }
}
```

---

### 4.22 `src/error.rs`

**Purpose:** All error types the library can produce.

```rust
use thiserror::Error;
use crate::types::State;
use crate::events::Event;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Agent failed: {0}")]
    AgentFailed(String),

    #[error("Invalid transition: {from} + {event} not in transition table")]
    InvalidTransition { from: State, event: Event },

    #[error("No handler registered for state: {0}")]
    NoHandlerForState(String),

    #[error("Safety cap exceeded after {0} iterations")]
    SafetyCapExceeded(usize),

    #[error("LLM caller error: {0}")]
    LlmError(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Build error: {0}")]
    BuildError(String),
}
```

---

### 4.23 `src/builder.rs`

**Purpose:** Ergonomic fluent API for constructing an `AgentEngine`. This is the primary user-facing interface.

```rust
use std::collections::HashMap;
use crate::engine::AgentEngine;
use crate::error::AgentError;
use crate::memory::AgentMemory;
use crate::tools::{ToolRegistry, ToolFn};
use crate::llm::LlmCaller;
use crate::states::{
    AgentState, IdleState, PlanningState, ActingState,
    ObservingState, ReflectingState, DoneState, ErrorState,
};
use crate::transitions::build_transition_table;
use crate::types::AgentConfig;

pub struct AgentBuilder {
    memory:  AgentMemory,
    tools:   ToolRegistry,
    llm:     Option<Box<dyn LlmCaller>>,
    config:  Option<AgentConfig>,
}

impl AgentBuilder {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            memory: AgentMemory::new(task),
            tools:  ToolRegistry::new(),
            llm:    None,
            config: None,
        }
    }

    pub fn task_type(mut self, t: impl Into<String>) -> Self {
        self.memory.task_type = t.into(); self
    }

    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.memory.system_prompt = p.into(); self
    }

    pub fn llm(mut self, llm: Box<dyn LlmCaller>) -> Self {
        self.llm = Some(llm); self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config); self
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.memory.config.max_steps = n; self
    }

    pub fn tool(
        mut self,
        name:        impl Into<String>,
        description: impl Into<String>,
        schema:      serde_json::Value,
        func:        ToolFn,
    ) -> Self {
        self.tools.register(name, description, schema, func);
        self
    }

    pub fn blacklist_tool(mut self, name: impl Into<String>) -> Self {
        self.memory.blacklist_tool(name); self
    }

    /// Builds the AgentEngine with all default state handlers.
    pub fn build(mut self) -> Result<AgentEngine, AgentError> {
        let llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required".to_string()))?;

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        // Register default state handlers
        let mut handlers: HashMap<&'static str, Box<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle",       Box::new(IdleState));
        handlers.insert("Planning",   Box::new(PlanningState));
        handlers.insert("Acting",     Box::new(ActingState));
        handlers.insert("Observing",  Box::new(ObservingState));
        handlers.insert("Reflecting", Box::new(ReflectingState));
        handlers.insert("Done",       Box::new(DoneState));
        handlers.insert("Error",      Box::new(ErrorState));

        Ok(AgentEngine::new(
            self.memory,
            self.tools,
            llm,
            build_transition_table(),
            handlers,
        ))
    }

    /// Builds with custom state handlers — for advanced users extending the library.
    pub fn build_with_handlers(
        mut self,
        extra_handlers: HashMap<&'static str, Box<dyn AgentState>>,
    ) -> Result<AgentEngine, AgentError> {
        let mut engine = self.build()?;
        // Caller can replace individual handlers via this method
        // (AgentEngine must expose a register_handler method for this)
        Ok(engine)
    }
}
```

---

### 4.24 `examples/basic_agent.rs`

Demonstrates the complete minimal working agent with OpenAI and a single search tool.

```rust
use agentsm::{AgentBuilder, AgentConfig};
use agentsm::llm::{OpenAiCaller, LlmCallerExt};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));

    let mut engine = AgentBuilder::new("What is the capital of France and what is its population?")
        .task_type("research")
        .system_prompt("You are a helpful research assistant. Use the search tool to find information.")
        .llm(llm)
        .max_steps(10)
        .tool(
            "search",
            "Search the web for current information. Use for any factual queries.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" }
                },
                "required": ["query"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let query = args["query"].as_str().unwrap_or("");
                // In production: call a real search API here
                Ok(format!("Search results for '{}': Paris is the capital of France with a population of approximately 2.1 million in the city proper.", query))
            }),
        )
        .build()?;

    match engine.run() {
        Ok(answer) => {
            println!("\n=== FINAL ANSWER ===\n{}", answer);
            println!("\n=== TRACE ===");
            engine.trace().print();
        }
        Err(e) => eprintln!("Agent failed: {}", e),
    }

    Ok(())
}
```

---

### 4.25 `examples/multi_tool_agent.rs`

Demonstrates multiple tools, task_type routing, and blacklisted tools.

```rust
// Full working example with: search, calculator, weather tools
// Shows: tool blacklisting, task_type="calculation", custom AgentConfig
// Must compile and run successfully
```

---

### 4.26 `examples/anthropic_agent.rs`

Demonstrates the Anthropic provider with Claude models.

```rust
use agentsm::{AgentBuilder};
use agentsm::llm::{AnthropicCaller, LlmCallerExt};

// Shows: AnthropicCaller::from_env(), claude model strings,
// system_prompt, tool registration pattern for Claude
// Must compile and run successfully
```

---

### 4.27 `tests/integration_tests.rs`

All tests must pass with `cargo test`.

```rust
// Required test cases:

// test_idle_to_planning_transition
// Verify IdleState produces Event::Start and engine transitions to Planning

// test_planning_max_steps_guard
// Set step = max_steps, verify Planning returns MaxSteps, engine goes to Error

// test_planning_tool_blacklist
// Register blacklisted tool, mock LLM returns it, verify ToolBlacklisted event

// test_acting_unknown_tool_is_failure_not_crash
// Call acting with tool not in registry, verify ToolFailure (not panic/FatalError)

// test_observing_triggers_reflection_at_step_5
// Set step=5, verify NeedsReflection event

// test_observing_commits_to_history
// Run through acting+observing, verify memory.history has one entry

// test_full_run_with_mock_llm
// Program MockLlmCaller with: ToolCall(search), ToolCall(calculator), FinalAnswer
// Verify run() returns Ok with non-empty string

// test_full_run_reaches_done_state
// After run(), verify engine.current_state() == State::Done

// test_invalid_transition_returns_error
// Manually set engine to Done state, emit an event not in table
// Verify AgentError::InvalidTransition

// test_trace_records_all_steps
// After full run, verify trace.len() > 0 and entries have correct state names

// test_tool_registry_execute_unknown_returns_err
// Verify registry.execute("nonexistent", &{}) returns Err, not panic

// test_mock_llm_call_count
// Run agent, verify MockLlmCaller.call_count() matches expected number of LLM calls

// test_agent_config_respected
// Set max_steps=2, verify agent fails with MaxSteps after 2 steps

// test_builder_requires_llm
// Call AgentBuilder::new(...).build() without setting llm
// Verify BuildError is returned
```

---

## 5. Data Flow & Execution Model

```
AgentBuilder::new(task)
    .llm(provider)
    .tool("name", desc, schema, fn)
    .build()
        │
        ▼
AgentEngine::run()
        │
        ▼
    LOOP:
        │
        ├── state.is_terminal()? ──YES──▶ return Ok/Err
        │
        ├── handler = handlers[current_state.as_str()]
        │
        ├── event = handler.handle(&mut memory, &tools, &llm)
        │       │
        │       ├── [PlanningState]
        │       │     ├── memory.step++
        │       │     ├── model = select_model(memory.task_type)
        │       │     ├── response = llm.call(memory, tools, model)
        │       │     │     └── builds messages from memory.build_messages()
        │       │     │     └── builds tool schemas from tools.schemas()
        │       │     │     └── calls LLM API
        │       │     │     └── parses response into LlmResponse
        │       │     └── returns Event based on response
        │       │
        │       ├── [ActingState]
        │       │     ├── tool_call = memory.current_tool_call
        │       │     ├── result = tools.execute(tool_call.name, tool_call.args)
        │       │     ├── memory.last_observation = result (SUCCESS: or ERROR:)
        │       │     └── returns ToolSuccess or ToolFailure
        │       │
        │       └── [ObservingState]
        │             ├── memory.history.push(HistoryEntry)
        │             ├── memory.current_tool_call = None
        │             ├── memory.last_observation = None
        │             └── returns Continue or NeedsReflection
        │
        ├── next = transitions[(current_state, event)]
        │           └── if not found: return Err(InvalidTransition)
        │
        └── current_state = next  ──▶ LOOP
```

---

## 6. Type Contracts

These invariants are enforced by the type system and must never be violated by any implementation:

| Invariant | Enforcement |
|---|---|
| Every `Event` variant is handled in the transition table | `HashMap` lookup returns `Option` — missing = `InvalidTransition` error |
| `current_tool_call` is always `Some` when `ActingState::handle()` runs | `ActingState` checks `None` → `FatalError` event |
| `final_answer` is always `Some` when engine returns `Ok` | Engine returns `final_answer.unwrap_or("[No answer]")` |
| `error` is always `Some` when engine returns `AgentError::AgentFailed` | `ErrorState` sets it; `AgentFailed` reads it |
| Tool failures never panic | All `tools.execute()` calls are wrapped in `Result` |
| LLM failures never panic | All `llm.call()` calls are wrapped in `Result` |
| History entries are append-only | `Vec::push()` only — never remove individual entries |
| Trace entries are append-only | `Trace::record()` only — never mutate existing entries |

---

## 7. Error Handling Strategy

```
Category              Action                            Event/Error
─────────────────────────────────────────────────────────────────────
LLM call failed       Set memory.error, return event    FatalError → Error state
Tool not in registry  Set last_observation, continue    ToolFailure → Observing
Tool threw exception  Set last_observation, continue    ToolFailure → Observing
Blacklisted tool      Log, continue                     ToolBlacklisted → Planning (retry)
Low confidence        Increment retry_count, continue   LowConfidence → Reflecting
Answer too short      Log, continue                     AnswerTooShort → Planning (retry)
Max steps exceeded    Set memory.error, return event    MaxSteps → Error state
Invalid transition    Return Err immediately            AgentError::InvalidTransition
No handler for state  Return Err immediately            AgentError::NoHandlerForState
Safety cap exceeded   Return Err immediately            AgentError::SafetyCapExceeded
```

**The golden rule:** If recovery is possible (the LLM can reason about it), the error becomes an observation. If recovery is impossible (structural bug in the library), it becomes an `AgentError`.

---

## 8. Testing Strategy

**Unit tests** (in each `src/states/*.rs` file):
- Each state tested in complete isolation
- Use `MockLlmCaller` and empty/minimal `ToolRegistry`
- Test every branch: success path, every failure path, every edge case

**Integration tests** (in `tests/integration_tests.rs`):
- Full engine run with `MockLlmCaller`
- Verify state sequence via trace inspection
- Test the builder API end-to-end

**Test helpers to implement in `src/` or `tests/`:**
```rust
// Convenience constructors for testing
fn test_memory() -> AgentMemory { AgentMemory::new("test task") }
fn test_tools() -> ToolRegistry { ToolRegistry::new() }
fn make_tool_call_response(name: &str) -> LlmResponse { ... }
fn make_final_answer(content: &str) -> LlmResponse { ... }
fn make_mock_llm(responses: Vec<LlmResponse>) -> MockLlmCaller { ... }
```

---

## 9. Extension Points

The library is designed to be extended without modifying core files:

**Custom state:** Implement `AgentState` trait, register in engine's handler map.

**Custom LLM provider:** Implement `AsyncLlmCaller`, wrap with `LlmCallerExt`.

**Custom transition table:** Call `build_transition_table()`, add your rows, pass to `AgentEngine::new()`.

**Custom memory fields:** The `AgentMemory` struct is public — embed it in a wrapper struct and pass it through your custom state implementations.

**Multi-agent:** Each `AgentEngine` is a self-contained value. Spawn multiple engines in separate tokio tasks, coordinate via channels (`tokio::sync::mpsc`).

---

## 10. Invariants & Rules the Agent Must Never Violate

These rules are absolute. Any generated code that violates them is incorrect.

1. **The engine never contains business logic.** If you find yourself writing an `if` statement in `AgentEngine::run()` that is not about loop control, it belongs in a state handler.

2. **State handlers never call each other.** `PlanningState::handle()` never calls `ActingState::handle()`. States communicate only through events and shared memory.

3. **State handlers never modify `memory.step` except PlanningState.** Only `PlanningState::handle()` increments `memory.step`.

4. **`memory.current_tool_call` is only written by PlanningState and cleared by ObservingState.** No other state touches it.

5. **`memory.last_observation` is only written by ActingState and cleared by ObservingState.** No other state touches it.

6. **Tool execution never throws.** Every `tools.execute()` call is inside a match or `?` that produces `Result`. Panics in tool functions must be caught via `std::panic::catch_unwind` if tools are user-supplied.

7. **The transition table is built once and never mutated after construction.** It is passed immutably to the engine.

8. **Trace entries are never deleted or modified.** The trace is an append-only event log.

9. **`DoneState` and `ErrorState` never modify memory.** They are read-only observers.

10. **All public API types implement `Debug`.** No exception. Observability requires it.

---

## Quick Start After Build

```bash
# Build
cargo build

# Run tests
cargo test

# Run basic example (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... cargo run --example basic_agent

# Run with Anthropic (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_agent

# Run with full logging
RUST_LOG=debug cargo run --example basic_agent
```

---

*This README is a complete specification. An agent following it has everything needed to build `agentsm-rs` as a fully working, production-ready Rust library with zero ambiguity.*
