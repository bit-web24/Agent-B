use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;

pub struct IdleState;

impl AgentState for IdleState {
    fn name(&self) -> &'static str { "Idle" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        _tools: &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        memory.log("Idle", "AGENT_STARTED", &format!(
            "task='{}' task_type='{}' max_steps={}",
            memory.task, memory.task_type, memory.config.max_steps
        ));
        Event::Start
    }
}
