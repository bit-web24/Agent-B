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
pub use types::{State, LlmResponse, ToolCall, HistoryEntry, AgentConfig};
pub use events::Event;
pub use tools::{ToolRegistry, ToolFn, Tool};
pub use llm::{LlmCaller, LlmCallerExt, RetryingLlmCaller};
pub use trace::{TraceEntry, Trace};
pub use error::AgentError;
