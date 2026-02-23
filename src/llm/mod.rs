use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::{LlmResponse, LlmStreamChunk};
use async_trait::async_trait;
use futures::stream::BoxStream;

mod openai;
mod anthropic;
mod mock;
mod retry;

pub use openai::OpenAiCaller;
pub use anthropic::AnthropicCaller;
pub use mock::MockLlmCaller;
pub use retry::RetryingLlmCaller;

/// The single interface between the state machine and any LLM provider.
///
/// # Contract
/// - Must be Send + Sync (used behind Box<dyn LlmCaller>)
/// - Returns Ok(LlmResponse) on any valid LLM interaction
/// - Returns Err(String) ONLY for unrecoverable failures:
///   - Network failure after retries exhausted
///   - Authentication failure
///   - Response unparseable as LlmResponse
/// - MUST build the tool schemas from `tools.schemas()` and include
///   them in every API call
/// - MUST build messages from `memory.build_messages()`
///
pub trait LlmCaller: Send + Sync {
    fn call(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String>;
}

/// Async version of LlmCaller for async runtimes.
#[async_trait]
pub trait AsyncLlmCaller: Send + Sync {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String>;

    /// Asynchronously streams chunks from the LLM.
    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
    ) -> BoxStream<'a, Result<LlmStreamChunk, String>>;
}

/// Extension trait: wraps an AsyncLlmCaller into a sync LlmCaller
/// using tokio::runtime::Handle::block_on.
pub struct SyncWrapper<T: AsyncLlmCaller>(pub T);

impl<T: AsyncLlmCaller> LlmCaller for SyncWrapper<T> {
    fn call(&self, memory: &AgentMemory, tools: &ToolRegistry, model: &str) -> Result<LlmResponse, String> {
        // block_in_place moves the current thread out of the async executor
        // context before blocking, preventing the "Cannot start a runtime
        // from within a runtime" panic when called from #[tokio::main].
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(self.0.call_async(memory, tools, model))
        })
    }
}

pub use SyncWrapper as LlmCallerExt;
