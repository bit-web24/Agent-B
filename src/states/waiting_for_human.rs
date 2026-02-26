use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, State};
use crate::human::HumanDecision;
use async_trait::async_trait;
use std::sync::Arc;

pub struct WaitingForHumanState;

#[async_trait]
impl AgentState for WaitingForHumanState {
    fn name(&self) -> &'static str { "WaitingForHuman" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        _tools:    &std::sync::Arc<ToolRegistry>,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::waiting_for_human()));
            let _ = tx.send(AgentOutput::Action("Waiting for human approval...".to_string()));
        }

        let request = match memory.pending_approval.take() {
            Some(req) => req,
            None => {
                memory.error = Some("WaitingForHumanState called with no pending_approval".to_string());
                memory.log("WaitingForHuman", "FATAL_ERROR", "No pending_approval");
                return Event::fatal_error();
            }
        };

        memory.log("WaitingForHuman", "APPROVAL_REQUEST", &request.tool_name);

        let callback = match memory.approval_callback.as_ref() {
            Some(cb) => Arc::clone(&cb.0),
            None => {
                // If no callback, we might just hang or return error.
                // In a real prod system, this might wait on a channel or external event.
                // For now, let's treat it as a fatal error if no callback is registered.
                memory.error = Some("No approval_callback registered".to_string());
                memory.log("WaitingForHuman", "FATAL_ERROR", "No callback");
                return Event::fatal_error();
            }
        };

        // Invoke callback
        // NOTE: In a more complex system, this might be a long-running wait.
        // For simplicity, we assume the callback handles the interaction.
        let decision = callback(request);

        match decision {
            HumanDecision::Approved => {
                memory.log("WaitingForHuman", "APPROVED", "Human approved action");
                Event::human_approved()
            }
            HumanDecision::Rejected(reason) => {
                memory.log("WaitingForHuman", "REJECTED", &reason);
                memory.last_observation = Some(format!("REJECTED: {}", reason));
                Event::human_rejected()
            }
            HumanDecision::Modified { tool_name, tool_args } => {
                memory.log("WaitingForHuman", "MODIFIED", &tool_name);
                // Update the tool call in memory
                if let Some(ref mut tc) = memory.current_tool_call {
                    tc.name = tool_name;
                    tc.args = tool_args;
                }
                Event::human_modified()
            }
        }
    }
}
