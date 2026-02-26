use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, State};
use async_trait::async_trait;

pub struct IdleState;

#[async_trait]
impl AgentState for IdleState {
    fn name(&self) -> &'static str { "Idle" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        _tools:    &std::sync::Arc<ToolRegistry>,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::idle()));
        }

        memory.log("Idle", "AGENT_STARTED", &format!(
            "task='{}' task_type='{}' max_steps={}",
            memory.task, memory.task_type, memory.config.max_steps
        ));
        Event::start()
    }
}
