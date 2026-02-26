use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, State, ToolResult};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::time::Instant;
use futures::future::join_all;

pub struct ParallelActingState;

#[async_trait]
impl AgentState for ParallelActingState {
    fn name(&self) -> &'static str { "ParallelActing" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        tools:     &Arc<ToolRegistry>,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::parallel_acting()));
        }

        let pending = memory.pending_tool_calls.clone();
        let count = pending.len();
        memory.log("ParallelActing", "PARALLEL_ACTING_START", &format!("count={}", count));

        let mut tasks = Vec::new();
        for tool_call in pending {
            let tools_clone = Arc::clone(tools);
            let tx_clone = output_tx.cloned();
            
            tasks.push(tokio::task::spawn_blocking(move || {
                let start = Instant::now();
                
                if let Some(ref tx) = tx_clone {
                    let _ = tx.send(AgentOutput::ToolCallStarted {
                        name: tool_call.name.clone(),
                        args: tool_call.args.clone(),
                    });
                }

                let result = tools_clone.execute(&tool_call.name, &tool_call.args);
                let latency = start.elapsed().as_millis() as u64;

                let tool_result = match result {
                    Ok(res) => {
                        if let Some(ref tx) = tx_clone {
                            let _ = tx.send(AgentOutput::ToolCallFinished {
                                name: tool_call.name.clone(),
                                result: res.clone(),
                                success: true,
                            });
                        }
                        ToolResult::success(tool_call.name.clone(), tool_call.args.clone(), tool_call.id.clone(), res, latency)
                    }
                    Err(err) => {
                        if let Some(ref tx) = tx_clone {
                            let _ = tx.send(AgentOutput::ToolCallFinished {
                                name: tool_call.name.clone(),
                                result: err.clone(),
                                success: false,
                            });
                        }
                        ToolResult::failure(tool_call.name.clone(), tool_call.args.clone(), tool_call.id.clone(), err, latency)
                    }
                };
                tool_result
            }));
        }

        let results = join_all(tasks).await;
        let mut tool_results = Vec::new();
        let mut success_count = 0;

        for res in results {
            if let Ok(tool_res) = res {
                if tool_res.success {
                    success_count += 1;
                }
                tool_results.push(tool_res);
            }
        }

        memory.parallel_results = tool_results;
        memory.pending_tool_calls.clear();
        memory.log("ParallelActing", "PARALLEL_ACTING_DONE", &format!("success={}/{}", success_count, count));

        if success_count > 0 || count == 0 {
            Event::tool_success()
        } else {
            Event::tool_failure()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCall;
    use crate::tools::Tool;
    use std::collections::HashMap;

    struct MockLlm;
    #[async_trait]
    impl AsyncLlmCaller for MockLlm {
        async fn call_async(&self, _: &AgentMemory, _: &ToolRegistry, _: &str) -> Result<crate::types::LlmResponse, String> {
            Err("Not used".to_string())
        }
        fn call_stream_async<'a>(&'a self, _: &'a AgentMemory, _: &'a ToolRegistry, _: &'a str) -> futures::stream::BoxStream<'a, Result<crate::types::LlmStreamChunk, String>> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn test_parallel_acting_success() {
        let mut memory = AgentMemory::new("test");
        let mut registry = ToolRegistry::new();
        
        registry.register_tool(Tool::new("t1", "t1").call(|_| Ok("r1".to_string())));
        registry.register_tool(Tool::new("t2", "t2").call(|_| Ok("r2".to_string())));
        
        let tools = Arc::new(registry);
        
        memory.pending_tool_calls = vec![
            ToolCall { name: "t1".to_string(), args: HashMap::new(), id: Some("id1".to_string()) },
            ToolCall { name: "t2".to_string(), args: HashMap::new(), id: Some("id2".to_string()) },
        ];
        
        let state = ParallelActingState;
        let event = state.handle(&mut memory, &tools, &MockLlm, None).await;
        
        assert_eq!(event, Event::tool_success());
        assert_eq!(memory.parallel_results.len(), 2);
        assert!(memory.parallel_results.iter().all(|r| r.success));
        assert!(memory.pending_tool_calls.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_acting_partial_failure() {
        let mut memory = AgentMemory::new("test");
        let mut registry = ToolRegistry::new();
        
        registry.register_tool(Tool::new("t1", "t1").call(|_| Ok("r1".to_string())));
        registry.register_tool(Tool::new("t2", "t2").call(|_| Err("e2".to_string())));
        
        let tools = Arc::new(registry);
        
        memory.pending_tool_calls = vec![
            ToolCall { name: "t1".to_string(), args: HashMap::new(), id: Some("id1".to_string()) },
            ToolCall { name: "t2".to_string(), args: HashMap::new(), id: Some("id2".to_string()) },
        ];
        
        let state = ParallelActingState;
        let event = state.handle(&mut memory, &tools, &MockLlm, None).await;
        
        assert_eq!(event, Event::tool_success()); // Success if at least one succeeded
        assert_eq!(memory.parallel_results.len(), 2);
        assert!(memory.parallel_results.iter().any(|r| r.success));
        assert!(memory.parallel_results.iter().any(|r| !r.success));
    }

    #[tokio::test]
    async fn test_parallel_acting_total_failure() {
        let mut memory = AgentMemory::new("test");
        let mut registry = ToolRegistry::new();
        
        registry.register_tool(Tool::new("t1", "t1").call(|_| Err("e1".to_string())));
        registry.register_tool(Tool::new("t2", "t2").call(|_| Err("e2".to_string())));
        
        let tools = Arc::new(registry);
        
        memory.pending_tool_calls = vec![
            ToolCall { name: "t1".to_string(), args: HashMap::new(), id: Some("id1".to_string()) },
            ToolCall { name: "t2".to_string(), args: HashMap::new(), id: Some("id2".to_string()) },
        ];
        
        let state = ParallelActingState;
        let event = state.handle(&mut memory, &tools, &MockLlm, None).await;
        
        assert_eq!(event, Event::tool_failure());
        assert_eq!(memory.parallel_results.len(), 2);
        assert!(memory.parallel_results.iter().all(|r| !r.success));
    }
}
