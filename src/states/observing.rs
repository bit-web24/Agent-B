use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, HistoryEntry, State};
use async_trait::async_trait;

pub struct ObservingState;

#[async_trait]
impl AgentState for ObservingState {
    fn name(&self) -> &'static str { "Observing" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        _tools:    &std::sync::Arc<ToolRegistry>,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::observing()));
        }
        // Commit tool call and observation to history (single call legacy)
        let tool_call = memory.current_tool_call.take();
        let observation = memory.last_observation.take();

        match (tool_call, observation) {
            (Some(tool), Some(obs)) => {
                let success = obs.starts_with("SUCCESS:");
                let entry = HistoryEntry {
                    step: memory.step,
                    tool,
                    observation: obs.clone(),
                    success,
                };
                memory.history.push(entry);
                memory.log("Observing", "HISTORY_COMMIT", &format!(
                    "step={} success={} len={}", memory.step, success, memory.history.len()
                ));
            }
            _ => {}
        }

        // Commit parallel results if any
        let parallel = memory.parallel_results.drain(..).collect::<Vec<_>>();
        for res in parallel {
            memory.log("Observing", "HISTORY_COMMIT_PARALLEL", &format!(
                "step={} tool={} success={}", memory.step, res.tool_name, res.success
            ));
            let entry = HistoryEntry {
                step: memory.step,
                tool: crate::types::ToolCall { 
                    name: res.tool_name, 
                    args: res.tool_args,
                    id:   res.id,
                },
                observation: res.output,
                success: res.success,
            };
            memory.history.push(entry);
        }

        // Check if reflection is needed
        let reflect_interval = memory.config.reflect_every_n_steps;
        if reflect_interval > 0 && memory.step % reflect_interval == 0 {
            memory.log("Observing", "NEEDS_REFLECTION", &format!(
                "step={} interval={}", memory.step, reflect_interval
            ));
            Event::needs_reflection()
        } else {
            Event::r#continue()
        }
    }
}
