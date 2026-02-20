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
