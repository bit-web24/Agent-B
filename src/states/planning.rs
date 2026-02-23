use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{LlmResponse, ToolCall, AgentOutput, LlmStreamChunk, State};
use async_trait::async_trait;
use futures::StreamExt;

pub struct PlanningState;

impl PlanningState {
    /// Resolve the model to use for this call.
    ///
    /// Priority:
    ///   1. `memory.config.models[task_type]`  — exact task-type match
    ///   2. `memory.config.models["default"]`  — generic fallback
    ///   3. `""`                               — let the LlmCaller use its own default
    fn resolve_model<'a>(&self, memory: &'a AgentMemory) -> &'a str {
        let models = &memory.config.models;
        models
            .get(&memory.task_type)
            .or_else(|| models.get("default"))
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    fn handle_tool_call(&self, memory: &mut AgentMemory, tool: ToolCall, confidence: f64) -> Event {
        // Check blacklist
        if memory.blacklisted_tools.contains(&tool.name) {
            memory.log("Planning", "TOOL_BLACKLISTED", &format!(
                "Requested blacklisted tool: {}", tool.name
            ));
            return Event::tool_blacklisted();
        }

        // Check confidence
        if confidence < memory.config.confidence_threshold
            && memory.retry_count < memory.config.max_retries
        {
            memory.retry_count += 1;
            memory.confidence_score = confidence;
            memory.log("Planning", "LOW_CONFIDENCE", &format!(
                "confidence={:.2} threshold={:.2} retry={}/{}",
                confidence,
                memory.config.confidence_threshold,
                memory.retry_count,
                memory.config.max_retries
            ));
            return Event::low_confidence();
        }

        // Accept tool call
        memory.current_tool_call = Some(tool.clone());
        memory.confidence_score = confidence;
        memory.log("Planning", "LLM_TOOL_CALL", &format!(
            "tool='{}' confidence={:.2}", tool.name, confidence
        ));
        Event::llm_tool_call()
    }

    fn handle_final_answer(&self, memory: &mut AgentMemory, content: String) -> Event {
        // Check minimum length
        if content.len() < memory.config.min_answer_length {
            memory.log("Planning", "ANSWER_TOO_SHORT", &format!(
                "len={} min={}", content.len(), memory.config.min_answer_length
            ));
            return Event::answer_too_short();
        }

        // Accept answer
        memory.final_answer = Some(content.clone());
        memory.log("Planning", "LLM_FINAL_ANSWER", &content.chars().take(100).collect::<String>());
        Event::llm_final_answer()
    }
}

#[async_trait]
impl AgentState for PlanningState {
    fn name(&self) -> &'static str { "Planning" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &ToolRegistry,
        llm:       &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::planning()));
        }

        // 1. Guard: max steps
        if memory.step >= memory.config.max_steps {
            memory.error = Some(format!("Max steps {} exceeded", memory.config.max_steps));
            memory.log("Planning", "MAX_STEPS", &format!("step={}", memory.step));
            return Event::max_steps();
        }

        // 2. Increment step
        memory.step += 1;
        memory.log("Planning", "STEP_START", &format!("step={}/{}", memory.step, memory.config.max_steps));

        // 3. Resolve model
        let model = self.resolve_model(memory).to_string();

        // 4. Call LLM with streaming
        let mut stream = llm.call_stream_async(memory, tools, &model);
        let mut final_response = None;
        let mut stream_error = None;

        while let Some(res) = stream.next().await {
            match res {
                Ok(LlmStreamChunk::Content(token)) => {
                    if let Some(tx) = output_tx {
                        let _ = tx.send(AgentOutput::LlmToken(token));
                    }
                }
                Ok(LlmStreamChunk::ToolCallDelta { name, args_json }) => {
                    if let Some(tx) = output_tx {
                        let _ = tx.send(AgentOutput::ToolCallDelta { name, args_json });
                    }
                }
                Ok(LlmStreamChunk::Done(resp)) => {
                    final_response = Some(resp);
                }
                Err(err) => {
                    stream_error = Some(err);
                    break;
                }
            }
        }

        // Drop the stream before borrowing memory mutably again
        drop(stream);

        if let Some(err) = stream_error {
            memory.error = Some(format!("LLM streaming error: {}", err));
            memory.log("Planning", "LLM_ERROR", &err);
            return Event::fatal_error();
        }

        match final_response {
            Some(LlmResponse::ToolCall { tool, confidence }) => {
                self.handle_tool_call(memory, tool, confidence)
            }
            Some(LlmResponse::FinalAnswer { content }) => {
                self.handle_final_answer(memory, content)
            }
            None => {
                memory.error = Some("LLM stream ended without Done chunk".to_string());
                memory.log("Planning", "LLM_ERROR", "End of stream without response");
                Event::fatal_error()
            }
        }
    }
}
