use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;

pub struct ErrorState;

impl AgentState for ErrorState {
    fn name(&self) -> &'static str { "Error" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        _tools: &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        // Clone to avoid holding an immutable borrow while mutably borrowing for log()
        let error_msg = memory.error.clone()
            .unwrap_or_else(|| "Unknown error".to_string());
        memory.log("Error", "AGENT_FAILED", &error_msg);
        Event::Start  // Will never be used — engine exits before re-entering
    }
}
