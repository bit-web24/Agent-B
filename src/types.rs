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

    /// Model selection map: task_type → model name string.
    ///
    /// The key `"default"` is used as the fallback when the agent's
    /// `task_type` has no explicit entry.
    ///
    /// Example:
    /// ```no_run
    /// # use std::collections::HashMap;
    /// # use agentsm::AgentConfig;
    /// let _config = AgentConfig {
    ///     models: [
    ///         ("default".to_string(),     "gpt-4o".to_string()),
    ///         ("research".to_string(),    "gpt-4o".to_string()),
    ///         ("calculation".to_string(), "gpt-4o-mini".to_string()),
    ///     ].into(),
    ///     ..Default::default()
    /// };
    /// ```
    /// Leave empty to fall back on the LLM caller's own default.
    pub models: HashMap<String, String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps:             15,
            max_retries:           3,
            confidence_threshold:  0.4,
            reflect_every_n_steps: 5,
            min_answer_length:     20,
            models:                HashMap::new(), // no hardcoded defaults
        }
    }
}

