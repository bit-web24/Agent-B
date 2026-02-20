use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;
use crate::types::{HistoryEntry, ToolCall};
use std::collections::HashMap;

pub struct ReflectingState;

impl AgentState for ReflectingState {
    fn name(&self) -> &'static str { "Reflecting" }

    fn handle(
        &self,
        memory: &mut AgentMemory,
        _tools: &ToolRegistry,
        _llm:   &dyn LlmCaller,
    ) -> Event {
        memory.log("Reflecting", "COMPRESS_START", &format!(
            "history_entries={}", memory.history.len()
        ));

        // Create summary of history
        let history_json = serde_json::to_string_pretty(&memory.history)
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
            },
            observation: summary,
            success: true,
        };

        memory.history = vec![summary_entry];
        memory.retry_count = 0;  // Reset retry budget

        memory.log("Reflecting", "COMPRESS_DONE", &format!(
            "compressed to {} entries", memory.history.len()
        ));

        Event::ReflectDone
    }
}
