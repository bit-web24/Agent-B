use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::types::{AgentOutput, HistoryEntry, ToolCall, State};
use async_trait::async_trait;
use std::collections::HashMap;

pub struct ReflectingState;

#[async_trait]
impl AgentState for ReflectingState {
    fn name(&self) -> &'static str { "Reflecting" }

    async fn handle(
        &self,
        memory:    &mut AgentMemory,
        _tools:    &std::sync::Arc<ToolRegistry>,
        _llm:      &dyn AsyncLlmCaller,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<AgentOutput>>,
    ) -> Event {
        if let Some(tx) = output_tx {
            let _ = tx.send(AgentOutput::StateStarted(State::reflecting()));
            let _ = tx.send(AgentOutput::Action("Compressing history...".to_string()));
        }
        memory.log("Reflecting", "COMPRESS_START", &format!(
            "history_entries={}", memory.history.len()
        ));

        // Create summary of history
        let _history_json = serde_json::to_string_pretty(&memory.history)
            .unwrap_or_else(|_| "[]".to_string());

        let summary = format!(
            "Compressed {} tool call(s). Task: {}. Recent history available in context.",
            memory.history.len(),
            memory.task
        );

        // Replace history with single summary entry
        let summary_entry = HistoryEntry {
            step: memory.step,
            tool: ToolCall {
                name: "[SUMMARY]".to_string(),
                args: HashMap::new(),
                id:   None,
            },
            observation: summary,
            success: true,
        };

        memory.history = vec![summary_entry];
        memory.retry_count = 0;  // Reset retry budget

        memory.log("Reflecting", "COMPRESS_DONE", &format!(
            "compressed to {} entries", memory.history.len()
        ));

        Event::reflect_done()
    }
}
