use crate::types::{ToolCall, HistoryEntry, AgentConfig, ToolResult};
use crate::trace::{TraceEntry, Trace};
use crate::human::{HumanApprovalRequest, ApprovalPolicy, HumanDecision};
use chrono::Utc;
use std::collections::HashSet;
use std::sync::Arc;

pub struct ApprovalCallback(pub Arc<dyn Fn(HumanApprovalRequest) -> HumanDecision + Send + Sync>);

impl std::fmt::Debug for ApprovalCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<callback>")
    }
}

impl Clone for ApprovalCallback {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

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

    /// Multiple tool calls queued for parallel execution.
    pub pending_tool_calls: Vec<ToolCall>,
    /// Results from parallel tool execution.
    pub parallel_results:   Vec<ToolResult>,

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

    // ── Human-in-the-Loop ────────────────────────────────
    /// Set when a tool call requires human approval
    pub pending_approval:   Option<HumanApprovalRequest>,
    /// Policy defining which tools require approval
    pub approval_policy:    ApprovalPolicy,
    /// Callback invoked when approval is needed
    pub approval_callback:  Option<ApprovalCallback>,

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
            pending_tool_calls: Vec::new(),
            parallel_results:   Vec::new(),
            history:           Vec::new(),
            final_answer:      None,
            error:             None,
            config:            AgentConfig::default(),
            blacklisted_tools: HashSet::new(),
            pending_approval:   None,
            approval_policy:    ApprovalPolicy::default(),
            approval_callback:  None,
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
    /// Groups parallel tool calls into single assistant messages to comply with LLM protocols.
    pub fn build_messages(&self) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // System message
        if !self.system_prompt.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": self.system_prompt
            }));
        }

        // Current task (the initial user request)
        messages.push(serde_json::json!({
            "role": "user",
            "content": &self.task
        }));

        // History grouped by step
        let mut steps: Vec<Vec<&HistoryEntry>> = Vec::new();
        for entry in &self.history {
            if let Some(last_step) = steps.last_mut() {
                if last_step[0].step == entry.step {
                    last_step.push(entry);
                    continue;
                }
            }
            steps.push(vec![entry]);
        }

        for step_entries in steps {
            let mut oai_tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            for entry in step_entries {
                let tool_id = entry.tool.id.clone().unwrap_or_else(|| "legacy".to_string());
                
                oai_tool_calls.push(serde_json::json!({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": entry.tool.name,
                        "arguments": serde_json::to_string(&entry.tool.args).unwrap_or_default()
                    }
                }));

                tool_results.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": entry.tool.name,
                    "content": entry.observation
                }));
            }

            // 1. Assistant message with ALL tool calls from this step
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": oai_tool_calls
            }));

            // 2. Individual tool messages for each result
            messages.extend(tool_results);
        }

        messages
    }
}
