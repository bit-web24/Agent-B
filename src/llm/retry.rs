use std::collections::HashMap;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::LlmResponse;

/// A wrapper around any `LlmCaller` that retries transient failures
/// with exponential back-off.
///
/// # Retry policy
/// - Up to `max_retries` attempts are made (1 original + `max_retries` retries)
/// - Back-off: 1 s → 2 s → 4 s → … (doubles each attempt, capped at 30 s)
/// - **Auth errors are never retried** (contains "401", "403", or "authentication")
/// - All other `Err(String)` from the inner caller are considered transient
///
/// # Example
/// ```no_run
/// # use agentsm::AgentBuilder;
/// AgentBuilder::new("task")
///     .openai("sk-...")
///     .retry_on_error(3);   // up to 3 retries on transient failures
/// ```
pub struct RetryingLlmCaller {
    inner:       Box<dyn super::LlmCaller>,
    max_retries: u32,
}

impl RetryingLlmCaller {
    pub fn new(inner: Box<dyn super::LlmCaller>, max_retries: u32) -> Self {
        Self { inner, max_retries }
    }

    fn is_auth_error(err: &str) -> bool {
        let lower = err.to_lowercase();
        lower.contains("401")
            || lower.contains("403")
            || lower.contains("authentication")
            || lower.contains("unauthorized")
            || lower.contains("forbidden")
            || lower.contains("invalid api key")
    }
}

impl super::LlmCaller for RetryingLlmCaller {
    fn call(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        let mut last_err = String::new();

        for attempt in 0..=self.max_retries {
            match self.inner.call(memory, tools, model) {
                Ok(resp) => return Ok(resp),
                Err(e) if Self::is_auth_error(&e) => {
                    // Auth errors are never retried — fail immediately
                    tracing::error!(error = %e, "LLM auth error — not retrying");
                    return Err(e);
                }
                Err(e) => {
                    last_err = e.clone();
                    if attempt < self.max_retries {
                        // Exponential back-off: 1s, 2s, 4s, … capped at 30s
                        let wait_secs = std::cmp::min(1u64 << attempt, 30);
                        tracing::warn!(
                            attempt = attempt + 1,
                            max     = self.max_retries,
                            wait_s  = wait_secs,
                            error   = %e,
                            "LLM transient error — retrying"
                        );
                        std::thread::sleep(std::time::Duration::from_secs(wait_secs));
                    }
                }
            }
        }

        Err(format!(
            "LLM failed after {} retries — last error: {}",
            self.max_retries, last_err
        ))
    }
}
