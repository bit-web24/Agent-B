use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, State};
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::events::Event;
use async_trait::async_trait;

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
#[async_trait]
pub trait AgentState: Send + Sync {
    /// Returns the unique string name of this state.
    /// Must match the key used in the engine's handler map.
    fn name(&self) -> &'static str;

    /// Execute this state's logic asynchronously. Returns the Event that drives
    /// the next transition lookup in the transition table.
    ///
    /// If `output_tx` is provided, the state should emit `AgentOutput` events
    /// (tokens, tool calls, etc.) to it.
    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &ToolRegistry,
        llm:       &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event;
}
