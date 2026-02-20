use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;
use crate::types::{LlmResponse, ToolCall};

pub struct PlanningState;

impl PlanningState {
    fn select_model(&self, task_type: &str) -> &'static str {
        match task_type {
            "research"    => "claude-opus-4-6",
            "calculation" => "claude-haiku-4-5-20251001",
            _             => "claude-sonnet-4-6",
        }
    }

    fn handle_tool_call(&self, memory: &mut AgentMemory, tool: ToolCall, confidence: f64) -> Event {
        // Check blacklist
        if memory.blacklisted_tools.contains(&tool.name) {
            memory.log("Planning", "TOOL_BLACKLISTED", &format!(
                "Requested blacklisted tool: {}", tool.name
            ));
            return Event::ToolBlacklisted;
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
            return Event::LowConfidence;
        }

        // Accept tool call
        memory.current_tool_call = Some(tool.clone());
        memory.confidence_score = confidence;
        memory.log("Planning", "LLM_TOOL_CALL", &format!(
            "tool='{}' confidence={:.2}", tool.name, confidence
        ));
        Event::LlmToolCall
    }

    fn handle_final_answer(&self, memory: &mut AgentMemory, content: String) -> Event {
        // Check minimum length
        if content.len() < memory.config.min_answer_length {
            memory.log("Planning", "ANSWER_TOO_SHORT", &format!(
                "len={} min={}", content.len(), memory.config.min_answer_length
            ));
            return Event::AnswerTooShort;
        }

        // Accept answer
        memory.final_answer = Some(content.clone());
        memory.log("Planning", "LLM_FINAL_ANSWER", &content.chars().take(100).collect::<String>());
        Event::LlmFinalAnswer
    }
}

impl AgentState for PlanningState {
    fn name(&self) -> &'static str { "Planning" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        tools:  &ToolRegistry,
        llm:    &dyn LlmCaller,
    ) -> Event {
        // 1. Guard: max steps
        if memory.step >= memory.config.max_steps {
            memory.error = Some(format!("Max steps {} exceeded", memory.config.max_steps));
            memory.log("Planning", "MAX_STEPS", &format!("step={}", memory.step));
            return Event::MaxSteps;
        }

        // 2. Increment step
        memory.step += 1;
        memory.log("Planning", "STEP_START", &format!("step={}/{}", memory.step, memory.config.max_steps));

        // 3. Select model
        let model = self.select_model(&memory.task_type);

        // 4. Call LLM
        match llm.call(memory, tools, model) {
            Err(err) => {
                memory.error = Some(format!("LLM error: {}", err));
                memory.log("Planning", "LLM_ERROR", &err);
                Event::FatalError
            }
            Ok(LlmResponse::ToolCall { tool, confidence }) => {
                self.handle_tool_call(memory, tool, confidence)
            }
            Ok(LlmResponse::FinalAnswer { content }) => {
                self.handle_final_answer(memory, content)
            }
        }
    }
}
