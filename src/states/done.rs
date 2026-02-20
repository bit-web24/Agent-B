use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;

pub struct DoneState;

impl AgentState for DoneState {
    fn name(&self) -> &'static str { "Done" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        _tools: &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        let answer = memory.final_answer.as_deref().unwrap_or("[No answer]");
        let truncated: String = answer.chars().take(100).collect();
        memory.log("Done", "TASK_COMPLETE", &truncated);
        Event::Start  // Will never be used — engine exits before re-entering
    }
}
