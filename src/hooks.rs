//! Callback hooks for real-time agent observability.
//!
//! All methods have default no-op implementations — users only override what
//! they need.  Hook panics are caught and logged; they never crash the agent.

use crate::error::AgentError;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::types::LlmResponse;
use serde_json::Value;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Implement this trait to receive real-time notifications during agent
/// execution.  Every method has a default no-op so you only override the
/// events you care about.
pub trait AgentHooks: Send + Sync {
    /// Called *before* a state handler runs.
    fn on_state_enter(&self, _state: &str, _memory: &AgentMemory) {}

    /// Called *after* a state handler returns an event.
    fn on_state_exit(&self, _state: &str, _event: &Event, _memory: &AgentMemory) {}

    /// Called just before the LLM is invoked.
    fn on_llm_start(&self, _model: &str, _memory: &AgentMemory) {}

    /// Called after a successful LLM response.
    fn on_llm_end(&self, _model: &str, _response: &LlmResponse, _memory: &AgentMemory) {}

    /// Called when the LLM returns an error.
    fn on_llm_error(&self, _model: &str, _error: &str, _memory: &AgentMemory) {}

    /// Called just before a tool is executed.
    fn on_tool_start(
        &self,
        _tool_name: &str,
        _args: &HashMap<String, Value>,
        _memory: &AgentMemory,
    ) {
    }

    /// Called after a tool returns (success or failure).
    fn on_tool_end(&self, _tool_name: &str, _result: &str, _success: bool, _memory: &AgentMemory) {}

    /// Called once at the very start of `engine.run()`.
    fn on_agent_start(&self, _task: &str, _memory: &AgentMemory) {}

    /// Called once at the very end of `engine.run()`.
    fn on_agent_end(&self, _result: Result<&str, &AgentError>, _memory: &AgentMemory) {}

    /// Called when the introspection engine detects an anomaly.
    fn on_anomaly_detected(&self, _anomaly: &crate::introspection::Anomaly, _memory: &AgentMemory) {
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NoopHooks  (default when no hooks configured)
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-cost default — every method is a no-op.
pub struct NoopHooks;
impl AgentHooks for NoopHooks {}

// ─────────────────────────────────────────────────────────────────────────────
// CompositeHooks  (chains multiple hook implementations)
// ─────────────────────────────────────────────────────────────────────────────

/// Runs multiple `AgentHooks` in order.  If one hook panics, the panic is
/// caught and logged, and remaining hooks still execute.
pub struct CompositeHooks {
    hooks: Vec<Arc<dyn AgentHooks>>,
}

impl Default for CompositeHooks {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeHooks {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn add(mut self, hook: Arc<dyn AgentHooks>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Run a closure on every hook, catching panics.
    fn for_each<F: Fn(&dyn AgentHooks)>(&self, f: F) {
        for hook in &self.hooks {
            let hook_ref = hook.as_ref();
            let result = catch_unwind(AssertUnwindSafe(|| f(hook_ref)));
            if let Err(panic) = result {
                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                tracing::error!(error = %msg, "AgentHooks callback panicked — continuing");
            }
        }
    }
}

impl AgentHooks for CompositeHooks {
    fn on_state_enter(&self, state: &str, memory: &AgentMemory) {
        self.for_each(|h| h.on_state_enter(state, memory));
    }
    fn on_state_exit(&self, state: &str, event: &Event, memory: &AgentMemory) {
        self.for_each(|h| h.on_state_exit(state, event, memory));
    }
    fn on_llm_start(&self, model: &str, memory: &AgentMemory) {
        self.for_each(|h| h.on_llm_start(model, memory));
    }
    fn on_llm_end(&self, model: &str, response: &LlmResponse, memory: &AgentMemory) {
        self.for_each(|h| h.on_llm_end(model, response, memory));
    }
    fn on_llm_error(&self, model: &str, error: &str, memory: &AgentMemory) {
        self.for_each(|h| h.on_llm_error(model, error, memory));
    }
    fn on_tool_start(&self, tool_name: &str, args: &HashMap<String, Value>, memory: &AgentMemory) {
        self.for_each(|h| h.on_tool_start(tool_name, args, memory));
    }
    fn on_tool_end(&self, tool_name: &str, result: &str, success: bool, memory: &AgentMemory) {
        self.for_each(|h| h.on_tool_end(tool_name, result, success, memory));
    }
    fn on_agent_start(&self, task: &str, memory: &AgentMemory) {
        self.for_each(|h| h.on_agent_start(task, memory));
    }
    fn on_agent_end(&self, result: Result<&str, &AgentError>, memory: &AgentMemory) {
        // Clone into owned values so we can safely pass across panic boundaries.
        let ok_val: Option<String> = result.ok().map(|s| s.to_string());
        let err_val: Option<AgentError> = result.err().cloned();
        self.for_each(|h| {
            let r: Result<&str, &AgentError> = match (&ok_val, &err_val) {
                (Some(s), _) => Ok(s.as_str()),
                (_, Some(e)) => Err(e),
                _ => unreachable!(),
            };
            h.on_agent_end(r, memory);
        });
    }

    fn on_anomaly_detected(&self, anomaly: &crate::introspection::Anomaly, memory: &AgentMemory) {
        self.for_each(|h| h.on_anomaly_detected(anomaly, memory));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PrintHooks  (pretty-prints events to stdout for development)
// ─────────────────────────────────────────────────────────────────────────────

/// Development helper — prints hook events to stdout with ANSI colors.
pub struct PrintHooks;

impl AgentHooks for PrintHooks {
    fn on_state_enter(&self, state: &str, _memory: &AgentMemory) {
        println!("\x1b[36m▶ entering {}\x1b[0m", state);
    }
    fn on_state_exit(&self, state: &str, event: &Event, _memory: &AgentMemory) {
        println!("\x1b[36m◀ {} → {}\x1b[0m", state, event);
    }
    fn on_llm_start(&self, model: &str, _memory: &AgentMemory) {
        println!("\x1b[33m🤖 LLM call → {}\x1b[0m", model);
    }
    fn on_llm_end(&self, model: &str, response: &LlmResponse, _memory: &AgentMemory) {
        let kind = match response {
            LlmResponse::ToolCall { .. } => "tool_call",
            LlmResponse::ParallelToolCalls { .. } => "parallel_tool_calls",
            LlmResponse::FinalAnswer { .. } => "final_answer",
            LlmResponse::Structured { .. } => "structured",
        };
        println!("\x1b[33m✓ LLM ({}) → {}\x1b[0m", model, kind);
    }
    fn on_llm_error(&self, model: &str, error: &str, _memory: &AgentMemory) {
        println!("\x1b[31m✗ LLM ({}) error: {}\x1b[0m", model, error);
    }
    fn on_tool_start(
        &self,
        tool_name: &str,
        _args: &HashMap<String, Value>,
        _memory: &AgentMemory,
    ) {
        println!("\x1b[35m🔧 {} …\x1b[0m", tool_name);
    }
    fn on_tool_end(&self, tool_name: &str, _result: &str, success: bool, _memory: &AgentMemory) {
        let icon = if success { "✓" } else { "✗" };
        let color = if success { "32" } else { "31" };
        println!("\x1b[{}m{} {}\x1b[0m", color, icon, tool_name);
    }
    fn on_agent_start(&self, task: &str, _memory: &AgentMemory) {
        println!("\x1b[1m═══ Agent start: {}\x1b[0m", task);
    }
    fn on_agent_end(&self, result: Result<&str, &AgentError>, memory: &AgentMemory) {
        match result {
            Ok(answer) => println!(
                "\x1b[1;32m═══ Agent done ({} steps): {}…\x1b[0m",
                memory.step,
                &answer[..answer.len().min(80)]
            ),
            Err(e) => println!("\x1b[1;31m═══ Agent failed: {}\x1b[0m", e),
        }
    }

    fn on_anomaly_detected(&self, anomaly: &crate::introspection::Anomaly, _memory: &AgentMemory) {
        println!("\x1b[1;33m⚠️  [Anomaly] {}\x1b[0m", anomaly.to_note());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Panic-safe helpers used by engine/states to call hooks
// ─────────────────────────────────────────────────────────────────────────────

/// Call a hook method, catching any panic.  Used by engine and state handlers.
pub fn safe_hook<F: FnOnce()>(f: F) {
    let result = catch_unwind(AssertUnwindSafe(f));
    if let Err(panic) = result {
        let msg = if let Some(s) = panic.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        tracing::error!(error = %msg, "AgentHooks callback panicked — continuing");
    }
}
