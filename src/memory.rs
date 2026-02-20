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
