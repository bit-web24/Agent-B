//! Conversation memory strategies for controlling what history gets sent to the LLM.
//!
//! Three built-in strategies:
//! - `FullMemory` — send the complete conversation history (default behaviour)
//! - `SlidingWindowMemory` — keep only the last N message turns
//! - `SummaryMemory` — keep a rolling LLM-generated summary + last N turns

use serde_json::Value;

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Controls how conversation history is prepared before being sent to the LLM.
///
/// Implementations receive the full message list and return a (possibly
/// trimmed) list that fits within context constraints.
pub trait MemoryStrategy: Send + Sync {
    /// Transform the raw messages list into the messages that will actually
    /// be sent to the LLM.  The first message (system prompt) is always
    /// preserved — strategies operate on the remaining messages.
    fn apply(&self, messages: Vec<Value>) -> Vec<Value>;

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;
}

// ─────────────────────────────────────────────────────────────────────────────
// FullMemory
// ─────────────────────────────────────────────────────────────────────────────

/// Pass the complete history through unchanged.  This is the default.
pub struct FullMemory;

impl MemoryStrategy for FullMemory {
    fn apply(&self, messages: Vec<Value>) -> Vec<Value> {
        messages
    }
    fn name(&self) -> &'static str {
        "full"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SlidingWindowMemory
// ─────────────────────────────────────────────────────────────────────────────

/// Keep the system prompt + last `window_size` messages.
///
/// Useful to limit context length when conversations are very long.
pub struct SlidingWindowMemory {
    window_size: usize,
}

impl SlidingWindowMemory {
    /// Create a sliding window strategy.
    ///
    /// * `window_size` — number of recent messages to keep (excluding system prompt).
    ///   A window of 0 sends only the system prompt.
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl MemoryStrategy for SlidingWindowMemory {
    fn apply(&self, messages: Vec<Value>) -> Vec<Value> {
        if messages.is_empty() {
            return messages;
        }

        // Always keep the system prompt (first message if role == "system")
        let has_system = messages
            .first()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            .map(|r| r == "system")
            .unwrap_or(false);

        if has_system {
            let system = messages[0].clone();
            let rest = &messages[1..];
            let start = rest.len().saturating_sub(self.window_size);
            let mut result = vec![system];
            result.extend_from_slice(&rest[start..]);
            result
        } else {
            let start = messages.len().saturating_sub(self.window_size);
            messages[start..].to_vec()
        }
    }

    fn name(&self) -> &'static str {
        "sliding_window"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SummaryMemory
// ─────────────────────────────────────────────────────────────────────────────

/// Keep the system prompt + a summary message + the last `recent_count` messages.
///
/// The summary is updated externally (e.g. by an LLM call at the end of each
/// agent step).  This struct only manages how the summary is injected into
/// the message list.
pub struct SummaryMemory {
    summary: std::sync::Mutex<String>,
    recent_count: usize,
}

impl SummaryMemory {
    /// Create a summary memory strategy.
    ///
    /// * `recent_count` — number of recent messages to keep alongside the summary.
    pub fn new(recent_count: usize) -> Self {
        Self {
            summary: std::sync::Mutex::new(String::new()),
            recent_count,
        }
    }

    /// Update the rolling summary text.
    pub fn set_summary(&self, summary: impl Into<String>) {
        *self.summary.lock().unwrap() = summary.into();
    }

    /// Get the current summary.
    pub fn get_summary(&self) -> String {
        self.summary.lock().unwrap().clone()
    }
}

impl MemoryStrategy for SummaryMemory {
    fn apply(&self, messages: Vec<Value>) -> Vec<Value> {
        if messages.is_empty() {
            return messages;
        }

        let summary = self.summary.lock().unwrap().clone();

        // Always keep system prompt if present
        let has_system = messages
            .first()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            .map(|r| r == "system")
            .unwrap_or(false);

        let mut result = Vec::new();

        if has_system {
            result.push(messages[0].clone());
        }

        // Inject summary as a system-level context message
        if !summary.is_empty() {
            result.push(serde_json::json!({
                "role": "system",
                "content": format!("[Conversation Summary]\n{}", summary)
            }));
        }

        // Add recent messages
        let non_system = if has_system {
            &messages[1..]
        } else {
            &messages[..]
        };
        let start = non_system.len().saturating_sub(self.recent_count);
        result.extend_from_slice(&non_system[start..]);

        result
    }

    fn name(&self) -> &'static str {
        "summary"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn system_msg() -> Value {
        serde_json::json!({"role": "system", "content": "You are an assistant."})
    }
    fn user_msg(n: usize) -> Value {
        serde_json::json!({"role": "user", "content": format!("Message {}", n)})
    }
    fn asst_msg(n: usize) -> Value {
        serde_json::json!({"role": "assistant", "content": format!("Reply {}", n)})
    }

    #[test]
    fn test_full_memory_passes_all() {
        let strategy = FullMemory;
        let msgs = vec![system_msg(), user_msg(1), asst_msg(1), user_msg(2)];
        let result = strategy.apply(msgs.clone());
        assert_eq!(result.len(), 4);
        assert_eq!(result, msgs);
    }

    #[test]
    fn test_sliding_window_keeps_n() {
        let strategy = SlidingWindowMemory::new(2);
        let msgs = vec![
            system_msg(),
            user_msg(1),
            asst_msg(1),
            user_msg(2),
            asst_msg(2),
            user_msg(3),
            asst_msg(3),
        ];
        let result = strategy.apply(msgs);
        // System + last 2 messages
        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["role"], "system");
        assert_eq!(
            result[1]["content"],
            "user_msg 3".to_string().replace("user_msg ", "Message ")
        );
    }

    #[test]
    fn test_sliding_window_smaller_than_window() {
        let strategy = SlidingWindowMemory::new(10);
        let msgs = vec![system_msg(), user_msg(1)];
        let result = strategy.apply(msgs.clone());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_sliding_window_zero() {
        let strategy = SlidingWindowMemory::new(0);
        let msgs = vec![system_msg(), user_msg(1), asst_msg(1)];
        let result = strategy.apply(msgs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "system");
    }

    #[test]
    fn test_summary_memory_with_summary() {
        let strategy = SummaryMemory::new(2);
        strategy.set_summary("User asked about Rust. Assistant explained ownership.");

        let msgs = vec![
            system_msg(),
            user_msg(1),
            asst_msg(1),
            user_msg(2),
            asst_msg(2),
            user_msg(3),
        ];
        let result = strategy.apply(msgs);

        // System + summary + last 2 messages
        assert_eq!(result.len(), 4);
        assert_eq!(result[0]["role"], "system");
        assert!(result[1]["content"]
            .as_str()
            .unwrap()
            .contains("[Conversation Summary]"));
        assert_eq!(result[2]["content"], "Reply 2");
        assert_eq!(result[3]["content"], "Message 3");
    }

    #[test]
    fn test_summary_memory_no_summary() {
        let strategy = SummaryMemory::new(2);
        let msgs = vec![system_msg(), user_msg(1), asst_msg(1), user_msg(2)];
        let result = strategy.apply(msgs);

        // System + last 2 (no summary injected when empty)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["role"], "system");
    }

    #[test]
    fn test_sliding_window_no_system() {
        let strategy = SlidingWindowMemory::new(2);
        let msgs = vec![user_msg(1), asst_msg(1), user_msg(2), asst_msg(2)];
        let result = strategy.apply(msgs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_summary_get_set() {
        let strategy = SummaryMemory::new(1);
        assert!(strategy.get_summary().is_empty());
        strategy.set_summary("Hello world");
        assert_eq!(strategy.get_summary(), "Hello world");
    }

    #[test]
    fn test_full_memory_empty() {
        let strategy = FullMemory;
        let result = strategy.apply(vec![]);
        assert!(result.is_empty());
    }
}
