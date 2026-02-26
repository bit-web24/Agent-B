use std::sync::Mutex;
use crate::llm::LlmCaller;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::LlmResponse;
use async_trait::async_trait;

pub struct MockLlmCaller {
    responses: Mutex<Vec<LlmResponse>>,
    call_log:  Mutex<Vec<(String, String)>>,  // (model, memory.task)
}

impl MockLlmCaller {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            call_log:  Mutex::new(Vec::new()),
        }
    }

    /// Returns the number of times call() was invoked
    pub fn call_count(&self) -> usize {
        self.call_log.lock().unwrap().len()
    }

    /// Returns the model string passed to the Nth call (0-indexed)
    pub fn model_for_call(&self, n: usize) -> Option<String> {
        self.call_log.lock().unwrap()
            .get(n)
            .map(|(model, _)| model.clone())
    }
}

#[async_trait]
impl crate::llm::AsyncLlmCaller for MockLlmCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        self.call(memory, tools, model)
    }

    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
    ) -> futures::stream::BoxStream<'a, Result<crate::types::LlmStreamChunk, String>> {
        let resp = self.call(memory, tools, model);
        use futures::stream::{self, StreamExt};
        match resp {
            Ok(r) => stream::once(async move { Ok(crate::types::LlmStreamChunk::Done(r)) }).boxed(),
            Err(e) => stream::once(async move { Err(e) }).boxed(),
        }
    }
}

impl LlmCaller for MockLlmCaller {
    fn call(
        &self,
        memory: &AgentMemory,
        _tools: &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        self.call_log.lock().unwrap()
            .push((model.to_string(), memory.task.clone()));

        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Err("MockLlmCaller: no more programmed responses".to_string());
        }
        Ok(responses.remove(0))
    }
}
