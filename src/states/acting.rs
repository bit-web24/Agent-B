use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;

pub struct ActingState;

impl AgentState for ActingState {
    fn name(&self) -> &'static str { "Acting" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        tools:  &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        // Extract tool call from memory
        let tool_call = match memory.current_tool_call.as_ref() {
            Some(tc) => tc.clone(),
            None => {
                memory.error = Some("ActingState called with no current_tool_call".to_string());
                memory.log("Acting", "FATAL_ERROR", "No current_tool_call");
                return Event::fatal_error();
            }
        };

        memory.log("Acting", "TOOL_EXECUTE", &format!(
            "tool='{}' args={:?}", tool_call.name, tool_call.args
        ));

        // Execute tool
        match tools.execute(&tool_call.name, &tool_call.args) {
            Ok(result) => {
                let observation = format!("SUCCESS: {}", result);
                memory.last_observation = Some(observation.clone());
                memory.log("Acting", "TOOL_SUCCESS", &result.chars().take(100).collect::<String>());
                Event::tool_success()
            }
            Err(err) => {
                let observation = format!("ERROR: {}", err);
                memory.last_observation = Some(observation.clone());
                memory.log("Acting", "TOOL_FAILURE", &err);
                Event::tool_failure()
            }
        }
    }
}
