//! Adaptive Model Routing — dynamically select the LLM model based on runtime signals.
//!
//! Instead of static `task_type → model` mapping, the routing policy evaluates
//! conditions like confidence, step count, budget utilization, and tool failure
//! rate to pick the best model for each planning step.

use crate::memory::AgentMemory;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A condition evaluated against the agent's current memory state.
pub trait RoutingCondition: Send + Sync {
    fn evaluate(&self, memory: &AgentMemory) -> bool;
}

/// Blanket impl for closures.
impl<F> RoutingCondition for F
where
    F: Fn(&AgentMemory) -> bool + Send + Sync,
{
    fn evaluate(&self, memory: &AgentMemory) -> bool {
        self(memory)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in conditions
// ─────────────────────────────────────────────────────────────────────────────

/// Triggers when the last reported confidence is below a threshold.
#[derive(Clone)]
pub struct ConfidenceBelow(pub f64);

impl RoutingCondition for ConfidenceBelow {
    fn evaluate(&self, memory: &AgentMemory) -> bool {
        memory.confidence_score < self.0
    }
}

/// Triggers when the step count exceeds a threshold.
#[derive(Clone)]
pub struct StepAbove(pub usize);

impl RoutingCondition for StepAbove {
    fn evaluate(&self, memory: &AgentMemory) -> bool {
        memory.step > self.0
    }
}

/// Triggers when the budget utilization exceeds a percentage (0.0..=1.0).
#[derive(Clone)]
pub struct BudgetPctAbove(pub f64);

impl RoutingCondition for BudgetPctAbove {
    fn evaluate(&self, memory: &AgentMemory) -> bool {
        if let Some(ref budget) = memory.budget {
            if let Some(max) = budget.max_total_tokens {
                let used = memory.total_usage.total_tokens as f64;
                let max_f = max as f64;
                if max_f > 0.0 {
                    return (used / max_f) > self.0;
                }
            }
        }
        false
    }
}

/// Triggers when the recent tool failure rate exceeds a threshold.
/// Looks at the last `window` history entries.
#[derive(Clone)]
pub struct ToolFailureRateAbove {
    pub threshold: f64,
    pub window: usize,
}

impl RoutingCondition for ToolFailureRateAbove {
    fn evaluate(&self, memory: &AgentMemory) -> bool {
        if memory.history.is_empty() {
            return false;
        }
        let recent: Vec<_> = memory.history.iter().rev().take(self.window).collect();
        if recent.is_empty() {
            return false;
        }
        let failures = recent.iter().filter(|h| !h.success).count() as f64;
        let rate = failures / recent.len() as f64;
        rate > self.threshold
    }
}

/// Triggers when the step count is at or above a threshold (convenience alias).
#[derive(Clone)]
pub struct StepAtOrAbove(pub usize);

impl RoutingCondition for StepAtOrAbove {
    fn evaluate(&self, memory: &AgentMemory) -> bool {
        memory.step >= self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutingRule and RoutingPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// A single routing rule: if the condition matches, use the specified model.
pub struct RoutingRule {
    pub name: String,
    pub condition: Arc<dyn RoutingCondition>,
    pub model: String,
}

impl RoutingRule {
    pub fn new(
        name: impl Into<String>,
        condition: impl RoutingCondition + 'static,
        model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            condition: Arc::new(condition),
            model: model.into(),
        }
    }
}

impl Clone for RoutingRule {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            condition: self.condition.clone(),
            model: self.model.clone(),
        }
    }
}

/// A routing policy with ordered rules and a default model.
/// Rules are evaluated top-to-bottom; first match wins.
#[derive(Clone)]
pub struct RoutingPolicy {
    rules: Vec<RoutingRule>,
    default_model: String,
}

impl RoutingPolicy {
    /// Create a new policy with a default model for when no rule matches.
    pub fn new(default_model: impl Into<String>) -> Self {
        Self {
            rules: Vec::new(),
            default_model: default_model.into(),
        }
    }

    /// Add a routing rule. Rules are evaluated in insertion order.
    pub fn add_rule(mut self, rule: RoutingRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Convenience: add a rule using a built-in condition + target model.
    pub fn when(
        self,
        name: impl Into<String>,
        condition: impl RoutingCondition + 'static,
        model: impl Into<String>,
    ) -> Self {
        self.add_rule(RoutingRule::new(name, condition, model))
    }

    /// Convenience builders for common conditions.
    pub fn when_confidence_below(self, threshold: f64, model: impl Into<String>) -> Self {
        self.when(
            format!("confidence_below_{}", threshold),
            ConfidenceBelow(threshold),
            model,
        )
    }

    pub fn when_step_above(self, step: usize, model: impl Into<String>) -> Self {
        self.when(format!("step_above_{}", step), StepAbove(step), model)
    }

    pub fn when_budget_pct_above(self, pct: f64, model: impl Into<String>) -> Self {
        self.when(
            format!("budget_pct_above_{}", pct),
            BudgetPctAbove(pct),
            model,
        )
    }

    pub fn when_tool_failure_rate_above(
        self,
        threshold: f64,
        window: usize,
        model: impl Into<String>,
    ) -> Self {
        self.when(
            format!("tool_failure_rate_above_{}", threshold),
            ToolFailureRateAbove { threshold, window },
            model,
        )
    }

    /// Evaluate all rules against the current memory. Returns the model to use.
    pub fn resolve(&self, memory: &AgentMemory) -> &str {
        for rule in &self.rules {
            if rule.condition.evaluate(memory) {
                tracing::info!(
                    routing_rule = %rule.name,
                    model = %rule.model,
                    "Adaptive routing: rule matched"
                );
                return &rule.model;
            }
        }
        &self.default_model
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::AgentMemory;

    fn make_memory() -> AgentMemory {
        AgentMemory::new("test task")
    }

    #[test]
    fn test_default_model_when_no_rules() {
        let policy = RoutingPolicy::new("gpt-4o-mini");
        let m = make_memory();
        assert_eq!(policy.resolve(&m), "gpt-4o-mini");
    }

    #[test]
    fn test_first_matching_rule_wins() {
        let policy = RoutingPolicy::new("gpt-4o-mini")
            .when("always_true", |_: &AgentMemory| true, "gpt-4o")
            .when("also_true", |_: &AgentMemory| true, "claude-opus");
        let m = make_memory();
        assert_eq!(policy.resolve(&m), "gpt-4o"); // first match wins
    }

    #[test]
    fn test_no_match_uses_default() {
        let policy =
            RoutingPolicy::new("gpt-4o-mini").when("never_true", |_: &AgentMemory| false, "gpt-4o");
        let m = make_memory();
        assert_eq!(policy.resolve(&m), "gpt-4o-mini");
    }

    #[test]
    fn test_confidence_below_triggers() {
        let policy = RoutingPolicy::new("gpt-4o-mini").when_confidence_below(0.5, "gpt-4o");
        let mut m = make_memory();
        m.confidence_score = 0.3;
        assert_eq!(policy.resolve(&m), "gpt-4o");
    }

    #[test]
    fn test_confidence_above_threshold_no_trigger() {
        let policy = RoutingPolicy::new("gpt-4o-mini").when_confidence_below(0.5, "gpt-4o");
        let mut m = make_memory();
        m.confidence_score = 0.8;
        assert_eq!(policy.resolve(&m), "gpt-4o-mini");
    }

    #[test]
    fn test_step_above_triggers() {
        let policy = RoutingPolicy::new("gpt-4o-mini").when_step_above(5, "claude-opus");
        let mut m = make_memory();
        m.step = 7;
        assert_eq!(policy.resolve(&m), "claude-opus");
    }

    #[test]
    fn test_budget_pct_triggers() {
        let policy = RoutingPolicy::new("gpt-4o").when_budget_pct_above(0.8, "gpt-4o-mini");
        let mut m = make_memory();
        m.budget = Some(crate::budget::TokenBudget {
            max_total_tokens: Some(1000),
            max_input_tokens: None,
            max_output_tokens: None,
        });
        m.total_usage.total_tokens = 900; // 90% used
        assert_eq!(policy.resolve(&m), "gpt-4o-mini");
    }

    #[test]
    fn test_tool_failure_rate_triggers() {
        use crate::types::{HistoryEntry, ToolCall};
        let policy =
            RoutingPolicy::new("gpt-4o-mini").when_tool_failure_rate_above(0.5, 4, "gpt-4o");
        let mut m = make_memory();
        // 3 failures out of 4 = 75%
        for i in 0..4 {
            m.history.push(HistoryEntry {
                step: i,
                tool: ToolCall {
                    id: Some(format!("t{}", i)),
                    name: "search".into(),
                    args: Default::default(),
                },
                observation: "ERROR: timeout".into(),
                success: i == 0, // only first one succeeds
            });
        }
        assert_eq!(policy.resolve(&m), "gpt-4o");
    }

    #[test]
    fn test_multiple_conditions_priority() {
        let policy = RoutingPolicy::new("gpt-4o-mini")
            .when_step_above(10, "claude-opus") // won't trigger at step 7
            .when_confidence_below(0.5, "gpt-4o"); // will trigger
        let mut m = make_memory();
        m.step = 7;
        m.confidence_score = 0.3;
        assert_eq!(policy.resolve(&m), "gpt-4o"); // second rule matches
    }
}
