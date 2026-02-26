use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A named state in the agent's state machine.
///
/// States are identified by their string name. The library ships with
/// seven well-known constants (`State::IDLE`, `State::PLANNING`, …)
/// but users can define any number of custom states.
///
/// # Defining a Custom State
///
/// ```
/// use agentsm::State;
/// let researching = State::new("Researching");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct State(pub String);

impl State {
    /// Create a new state with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the string name of this state.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns true if this is one of the default terminal states
    /// (`"Done"` or `"Error"`).
    pub fn is_terminal(&self) -> bool {
        self.0 == "Done" || self.0 == "Error"
    }

    // ── Well-known built-in state constructors ──────────────────────────
    pub fn idle()       -> Self { Self::new("Idle") }
    pub fn planning()   -> Self { Self::new("Planning") }
    pub fn acting()     -> Self { Self::new("Acting") }
    pub fn observing()  -> Self { Self::new("Observing") }
    pub fn reflecting() -> Self { Self::new("Reflecting") }
    pub fn done()       -> Self { Self::new("Done") }
    pub fn error()      -> Self { Self::new("Error") }
    pub fn parallel_acting() -> Self { Self::new("ParallelActing") }
    pub fn waiting_for_human() -> Self { Self::new("WaitingForHuman") }
}

/// Result of a single tool execution in a parallel batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name:  String,
    pub tool_args:  HashMap<String, serde_json::Value>,
    pub id:         Option<String>,
    pub output:     String,      // "SUCCESS: ..." or "ERROR: ..."
    pub success:    bool,
    pub latency_ms: u64,
}

impl ToolResult {
    pub fn success(tool_name: String, tool_args: HashMap<String, serde_json::Value>,
                   id: Option<String>, output: String, latency_ms: u64) -> Self {
        Self { tool_name, tool_args, id, output: format!("SUCCESS: {}", output),
               success: true, latency_ms }
    }

    pub fn failure(tool_name: String, tool_args: HashMap<String, serde_json::Value>,
                   id: Option<String>, error: String, latency_ms: u64) -> Self {
        Self { tool_name, tool_args, id, output: format!("ERROR: {}", error),
               success: false, latency_ms }
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A tool invocation requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, serde_json::Value>,
    pub id:   Option<String>,
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
        usage:      Option<crate::budget::TokenUsage>,
    },
    /// LLM wants to invoke multiple tools in parallel
    ParallelToolCalls {
        tools:      Vec<ToolCall>,
        confidence: f64,
        usage:      Option<crate::budget::TokenUsage>,
    },
    /// LLM produced a final answer — task is complete
    FinalAnswer {
        content: String,
        usage:   Option<crate::budget::TokenUsage>,
    },
}

/// A chunk of streaming output from an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmStreamChunk {
    /// A piece of text content
    Content(String),
    /// Partial tool call arguments (accumulated)
    ToolCallDelta {
        name: Option<String>,
        args_json: String,
    },
    /// LLM finished streaming and returned a full response
    Done(LlmResponse),
}

/// High-level events emitted by the Agent during streaming execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentOutput {
    /// A new state has started execution
    StateStarted(State),
    /// A token/chunk of text from the LLM
    LlmToken(String),
    /// A chunk of tool call arguments
    ToolCallDelta {
        name: Option<String>,
        args_json: String,
    },
    /// A tool call is being initiated (fully parsed)
    ToolCallStarted {
        name: String,
        args: HashMap<String, serde_json::Value>,
    },
    /// A tool call has completed
    ToolCallFinished {
        name:    String,
        result:  String,
        success: bool,
    },
    /// A generic action or progress message
    Action(String),
    /// The agent has produced a final answer
    FinalAnswer(String),
    /// An error occurred during execution
    Error(String),
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

    /// Whether to allow parallel tool execution
    pub parallel_tools: bool,

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
            min_answer_length:     5,
            parallel_tools:        true,
            models:                HashMap::new(), // no hardcoded defaults
        }
    }
}

