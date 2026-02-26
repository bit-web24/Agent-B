use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, State};
use async_trait::async_trait;

pub struct DoneState;

#[async_trait]
impl AgentState for DoneState {
    fn name(&self) -> &'static str { "Done" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        _tools:    &std::sync::Arc<ToolRegistry>,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::done()));
        }

        let answer = memory.final_answer.clone().unwrap_or_else(|| "[No answer]".to_string());
        let truncated: String = answer.chars().take(100).collect();
        memory.log("Done", "TASK_COMPLETE", &truncated);

        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::FinalAnswer(answer));
        }
        Event::start()  // Will never be used — engine exits before re-entering
    }
}
