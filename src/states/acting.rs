use crate::events::Event;
use crate::llm::AsyncLlmCaller;
use crate::memory::AgentMemory;
use crate::states::AgentState;
use crate::tools::ToolRegistry;
use crate::types::{AgentOutput, State};
use async_trait::async_trait;

pub struct ActingState;

#[async_trait]
impl AgentState for ActingState {
    fn name(&self) -> &'static str {
        "Acting"
    }

    async fn handle(
        &self,
        memory: &mut AgentMemory,
        tools: &std::sync::Arc<ToolRegistry>,
        _llm: &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::acting()));
        }

        // Extract tool call from memory
        let tool_call = match memory.current_tool_call.as_ref() {
            Some(tc) => tc.clone(),
            None => {
                memory.error = Some("ActingState called with no current_tool_call".to_string());
                memory.log("Acting", "FATAL_ERROR", "No current_tool_call");
                return Event::fatal_error();
            }
        };

        memory.log(
            "Acting",
            "TOOL_EXECUTE",
            &format!("tool='{}' args={:?}", tool_call.name, tool_call.args),
        );

        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::ToolCallStarted {
                name: tool_call.name.clone(),
                args: tool_call.args.clone(),
            });
        }

        // Hook: on_tool_start
        memory
            .hooks
            .on_tool_start(&tool_call.name, &tool_call.args, memory);

        // Execute tool
        match tools.execute(&tool_call.name, &tool_call.args) {
            Ok(result) => {
                let observation = format!("SUCCESS: {}", result);
                memory.last_observation = Some(observation.clone());
                memory.log(
                    "Acting",
                    "TOOL_SUCCESS",
                    &result.chars().take(100).collect::<String>(),
                );

                if let Some(tx) = output_tx {
                    let _ = tx.send(AgentOutput::ToolCallFinished {
                        name: tool_call.name.clone(),
                        result: result.clone(),
                        success: true,
                    });
                }

                // Hook: on_tool_end
                memory
                    .hooks
                    .on_tool_end(&tool_call.name, &result, true, memory);

                Event::tool_success()
            }
            Err(err) => {
                let observation = format!("ERROR: {}", err);
                memory.last_observation = Some(observation.clone());
                memory.log("Acting", "TOOL_FAILURE", &err);

                if let Some(tx) = output_tx {
                    let _ = tx.send(AgentOutput::ToolCallFinished {
                        name: tool_call.name.clone(),
                        result: err.clone(),
                        success: false,
                    });
                }

                // Hook: on_tool_end
                memory
                    .hooks
                    .on_tool_end(&tool_call.name, &err, false, memory);

                Event::tool_failure()
            }
        }
    }
}
