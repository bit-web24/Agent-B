use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{LlmResponse, ToolCall, AgentOutput, State, LlmStreamChunk};
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

        // Check human approval
        if memory.approval_policy.needs_approval(&tool.name, &tool.args) {
            memory.pending_approval = Some(crate::human::HumanApprovalRequest {
                tool_name: tool.name.clone(),
                tool_args: tool.args.clone(),
                risk_level: crate::human::RiskLevel::High, // Default to High for now if policy says yes
                reason: "Policy-mandated approval".to_string(),
            });
            memory.current_tool_call = Some(tool);
            memory.confidence_score = confidence;
            memory.log("Planning", "APPROVAL_REQUIRED", "Action needs human approval");
            return Event::human_approval_required();
        }

        // Accept tool call
        memory.current_tool_call = Some(tool.clone());
        memory.pending_tool_calls.clear(); // Clear parallel queue if single call
        memory.confidence_score = confidence;
        memory.log("Planning", "LLM_TOOL_CALL", &format!(
            "tool='{}' confidence={:.2}", tool.name, confidence
        ));
        Event::llm_tool_call()
    }

    fn handle_parallel_tool_calls(&self, memory: &mut AgentMemory, tools: Vec<ToolCall>, confidence: f64) -> Event {
        memory.current_tool_call = None;
        memory.pending_tool_calls = tools.clone();
        memory.parallel_results.clear();
        memory.confidence_score = confidence;
        memory.log("Planning", "LLM_PARALLEL_TOOLS", &format!(
            "count={} confidence={:.2}", tools.len(), confidence
        ));
        Event::llm_parallel_tool_calls()
    }

    fn handle_final_answer(
        &self,
        memory: &mut AgentMemory,
        content: String,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
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

        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::FinalAnswer(content));
        }

        Event::llm_final_answer()
    }
}

#[async_trait]
impl AgentState for PlanningState {
    fn name(&self) -> &'static str { "Planning" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &std::sync::Arc<ToolRegistry>,
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

        // 2. Guard: Token Budget
        if let Some(budget) = memory.budget {
            if budget.is_exceeded(memory.total_usage) {
                memory.error = Some("Token budget exceeded".to_string());
                memory.log("Planning", "BUDGET_EXCEEDED", &format!("{:?}", memory.total_usage));
                return Event::fatal_error();
            }
        }

        // 2. Increment step
        memory.step += 1;
        memory.log("Planning", "STEP_START", &format!("step={}/{}", memory.step, memory.config.max_steps));

        // 3. Resolve model
        let model = self.resolve_model(memory).to_string();

        // 4. Call LLM (streaming)
        let (final_resp, stream_err) = {
            let mut stream = llm.call_stream_async(memory, tools, &model, output_tx);
            let mut final_resp = None;
            let mut stream_err = None;

            while let Some(chunk_res) = stream.next().await {
                match chunk_res {
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
                        final_resp = Some(resp);
                    }
                    Err(err) => {
                        stream_err = Some(err);
                        break;
                    }
                }
            }

            (final_resp, stream_err)
        };

        let resp = if let Some(err) = stream_err {
            memory.log("Planning", "LLM_STREAM_ERROR", &err);
            match llm.call_async(memory, tools, &model, output_tx).await {
                Ok(resp) => {
                    memory.log("Planning", "LLM_FALLBACK_SYNC", "Recovered via non-stream call");
                    resp
                }
                Err(sync_err) => {
                    memory.error = Some(format!(
                        "LLM stream error: {} | fallback call_async error: {}",
                        err, sync_err
                    ));
                    memory.log("Planning", "LLM_ERROR", &sync_err);
                    return Event::fatal_error();
                }
            }
        } else {
            match final_resp {
                Some(r) => r,
                None => {
                    let stream_end_err = "LLM stream ended without Done chunk".to_string();
                    memory.log("Planning", "STREAM_ERROR", &stream_end_err);
                    match llm.call_async(memory, tools, &model, output_tx).await {
                        Ok(resp) => {
                            memory.log("Planning", "LLM_FALLBACK_SYNC", "Recovered from incomplete stream");
                            resp
                        }
                        Err(sync_err) => {
                            memory.error = Some(format!(
                                "{} | fallback call_async error: {}",
                                stream_end_err, sync_err
                            ));
                            memory.log("Planning", "LLM_ERROR", &sync_err);
                            return Event::fatal_error();
                        }
                    }
                }
            }
        };
        let (LlmResponse::ToolCall { usage, .. } |
               LlmResponse::ParallelToolCalls { usage, .. } |
               LlmResponse::FinalAnswer { usage, .. }) = &resp;

        if let Some(u) = usage {
            memory.total_usage.add(*u);
        }

        match resp {
            LlmResponse::ToolCall { tool, confidence, .. } => {
                self.handle_tool_call(memory, tool, confidence)
            }
            LlmResponse::ParallelToolCalls { tools, confidence, .. } => {
                self.handle_parallel_tool_calls(memory, tools, confidence)
            }
            LlmResponse::FinalAnswer { content, .. } => {
                self.handle_final_answer(memory, content, output_tx)
            }
        }
    }
}
