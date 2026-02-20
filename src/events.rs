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
