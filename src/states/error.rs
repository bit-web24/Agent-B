use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, State};
use async_trait::async_trait;

pub struct ErrorState;

#[async_trait]
impl AgentState for ErrorState {
    fn name(&self) -> &'static str { "Error" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        _tools:    &ToolRegistry,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::error()));
        }
        // Clone to avoid holding an immutable borrow while mutably borrowing for log()
        let error_msg = memory.error.clone()
            .unwrap_or_else(|| "Unknown error".to_string());
        memory.log("Error", "AGENT_FAILED", &error_msg);

        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::Error(error_msg));
        }
        Event::start()  // Will never be used — engine exits before re-entering
    }
}
