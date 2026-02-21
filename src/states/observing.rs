use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;
use crate::types::HistoryEntry;

pub struct ObservingState;

impl AgentState for ObservingState {
    fn name(&self) -> &'static str { "Observing" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        _tools: &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        // Commit tool call and observation to history
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
            _ => {
                memory.log("Observing", "WARNING", "No tool_call or observation to commit");
            }
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
