use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::{LlmResponse, LlmStreamChunk};
use async_trait::async_trait;
use futures::stream::BoxStream;

use std::sync::Arc;

/// A wrapper around any `AsyncLlmCaller` that retries transient failures
/// with exponential back-off.
pub struct RetryingLlmCaller {
    inner:       Arc<dyn super::AsyncLlmCaller>,
    max_retries: u32,
}

impl RetryingLlmCaller {
    pub fn new(inner: Arc<dyn super::AsyncLlmCaller>, max_retries: u32) -> Self {
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

    fn is_rate_limit_error(err: &str) -> bool {
        let lower = err.to_lowercase();
        lower.contains("429")
            || lower.contains("rate limit")
            || lower.contains("too many requests")
            || lower.contains("too_many_tokens_error")
            || lower.contains("token_quota_exceeded")
            || lower.contains("too_many_requests_error")
            || lower.contains("queue_exceeded")
            || lower.contains("limit exceeded")
    }
}

#[async_trait]
impl super::AsyncLlmCaller for RetryingLlmCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<crate::types::AgentOutput>>,
    ) -> Result<LlmResponse, String> {
        let mut last_err = String::new();
        let mut rate_limited = false;

        for attempt in 0..=self.max_retries {
            match self.inner.call_async(memory, tools, model, output_tx).await {
                Ok(resp) => return Ok(resp),
                Err(e) if Self::is_auth_error(&e) => {
                    tracing::error!(error = %e, "LLM auth error — not retrying");
                    return Err(e);
                }
                Err(e) => {
                    last_err = e.clone();
                    if Self::is_rate_limit_error(&e) {
                        rate_limited = true;
                    }

                    if attempt < self.max_retries {
                        // For rate limits, use a longer initial wait
                        let base_wait = if Self::is_rate_limit_error(&e) { 5 } else { 1 };
                        let wait_secs = std::cmp::min(base_wait << attempt, 60);
                        
                        if let Some(tx) = output_tx {
                            let msg = if Self::is_rate_limit_error(&e) {
                                format!("Rate limit hit (429). Waiting {}s before retry...", wait_secs)
                            } else {
                                format!("Transient error. Waiting {}s before retry...", wait_secs)
                            };
                            let _ = tx.send(crate::types::AgentOutput::Action(msg));
                        }

                        tracing::warn!(
                            attempt = attempt + 1,
                            max     = self.max_retries,
                            wait_s  = wait_secs,
                            error   = %e,
                            "LLM transient error — retrying"
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
                    }
                }
            }
        }

        let prefix = if rate_limited {
            "LLM RATE LIMIT EXCEEDED"
        } else {
            "LLM failed"
        };

        Err(format!(
            "{} after {} retries — last error: {}",
            prefix, self.max_retries, last_err
        ))
    }

    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<crate::types::AgentOutput>>,
    ) -> BoxStream<'a, Result<crate::types::LlmStreamChunk, String>> {
        // Retrying a stream is complex. For now, we just delegate to the inner caller.
        // If the initial connection fails, we could retry, but if it fails mid-stream, 
        // we'd lose state. Industry grade usually handles this at a higher level
        // or has complex chunk accumulation & recovery.
        self.inner.call_stream_async(memory, tools, model, output_tx)
    }
}
