use serde::{Deserialize, Serialize};

/// Tracks token usage for a single LLM call or an entire session.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens:  u32,
    pub output_tokens: u32,
    pub total_tokens:  u32,
}

impl TokenUsage {
    pub fn new(input: u32, output: u32) -> Self {
        Self {
            input_tokens:  input,
            output_tokens: output,
            total_tokens:  input + output,
        }
    }

    /// Accumulate usage from another call
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Defines limits on token usage for an agent session.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenBudget {
    pub max_total_tokens:  Option<u32>,
    pub max_input_tokens:  Option<u32>,
    pub max_output_tokens: Option<u32>,
}

impl TokenBudget {
    pub fn new(max_total: u32) -> Self {
        Self {
            max_total_tokens:  Some(max_total),
            max_input_tokens:  None,
            max_output_tokens: None,
        }
    }

    /// Checks if the given usage exceeds this budget.
    /// Returns true if any limit is exceeded.
    pub fn is_exceeded(&self, usage: TokenUsage) -> bool {
        if let Some(limit) = self.max_total_tokens {
            if usage.total_tokens > limit { return true; }
        }
        if let Some(limit) = self.max_input_tokens {
            if usage.input_tokens > limit { return true; }
        }
        if let Some(limit) = self.max_output_tokens {
            if usage.output_tokens > limit { return true; }
        }
        false
    }
}
