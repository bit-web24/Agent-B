//! Self-Healing Policies — declarative recovery strategies.
//!
//! Users define triggers (what went wrong) and actions (how to recover)
//! as a `HealingPolicy`. The engine evaluates the policy after each step
//! and applies the first matching action.

use crate::memory::AgentMemory;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Triggers
// ─────────────────────────────────────────────────────────────────────────────

/// What condition triggers a healing action.
#[derive(Debug, Clone)]
pub enum HealingTrigger {
    /// Tool returned an error containing this substring.
    ToolError(String),
    /// Same tool+args called N times consecutively.
    RepeatedToolCall(usize),
    /// N consecutive tool failures (any tool).
    ConsecutiveFailures(usize),
    /// Confidence dropped below this threshold.
    ConfidenceBelow(f64),
    /// Token budget usage exceeded this fraction (0.0–1.0).
    BudgetPctAbove(f64),
    /// Loop detected by introspection (any anomaly note containing "loop" or repeated).
    LoopDetected,
}

// ─────────────────────────────────────────────────────────────────────────────
// Actions
// ─────────────────────────────────────────────────────────────────────────────

/// What to do when a trigger fires.
#[derive(Debug, Clone)]
pub enum HealingAction {
    /// Retry the last step with exponential backoff.
    RetryWithBackoff { max: usize, base_ms: u64 },
    /// Blacklist the offending tool and continue.
    BlacklistAndContinue,
    /// Switch the model for subsequent planning steps.
    SwitchModel(String),
    /// Inject a reflection note and retry planning.
    ReflectAndRetry,
    /// Summarize current findings and finish immediately.
    SummarizeAndFinish,
    /// Break out of detected loop with a note to LLM.
    BreakAndContinue,
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule
// ─────────────────────────────────────────────────────────────────────────────

/// A single healing rule: trigger + action + optional cooldown.
#[derive(Debug, Clone)]
pub struct HealingRule {
    pub trigger: HealingTrigger,
    pub action: HealingAction,
    /// Minimum time between applications of this rule.
    pub cooldown: Option<Duration>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy
// ─────────────────────────────────────────────────────────────────────────────

/// Ordered list of healing rules. First match wins.
#[derive(Debug, Clone)]
pub struct HealingPolicy {
    rules: Vec<HealingRule>,
    /// Track when each rule last fired (by index).
    last_fired: Vec<Option<Instant>>,
}

impl HealingPolicy {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            last_fired: Vec::new(),
        }
    }

    /// Add a rule with no cooldown.
    pub fn on(mut self, trigger: HealingTrigger, action: HealingAction) -> Self {
        self.rules.push(HealingRule {
            trigger,
            action,
            cooldown: None,
        });
        self.last_fired.push(None);
        self
    }

    /// Add a rule with a cooldown.
    pub fn on_with_cooldown(
        mut self,
        trigger: HealingTrigger,
        action: HealingAction,
        cooldown: Duration,
    ) -> Self {
        self.rules.push(HealingRule {
            trigger,
            action,
            cooldown: Some(cooldown),
        });
        self.last_fired.push(None);
        self
    }

    // ── Convenience builders ─────────────────────────────────────────────

    pub fn on_tool_error(self, substring: &str, action: HealingAction) -> Self {
        self.on(HealingTrigger::ToolError(substring.to_string()), action)
    }

    pub fn on_repeated_tool_call(self, n: usize, action: HealingAction) -> Self {
        self.on(HealingTrigger::RepeatedToolCall(n), action)
    }

    pub fn on_consecutive_failures(self, n: usize, action: HealingAction) -> Self {
        self.on(HealingTrigger::ConsecutiveFailures(n), action)
    }

    pub fn on_confidence_below(self, threshold: f64, action: HealingAction) -> Self {
        self.on(HealingTrigger::ConfidenceBelow(threshold), action)
    }

    pub fn on_budget_pct_above(self, pct: f64, action: HealingAction) -> Self {
        self.on(HealingTrigger::BudgetPctAbove(pct), action)
    }

    pub fn on_loop_detected(self, action: HealingAction) -> Self {
        self.on(HealingTrigger::LoopDetected, action)
    }

    /// Evaluate the policy against current memory state.
    /// Returns the first matching action, or None.
    pub fn evaluate(&mut self, memory: &AgentMemory) -> Option<HealingAction> {
        let now = Instant::now();

        for (i, rule) in self.rules.iter().enumerate() {
            // Check cooldown
            if let Some(cooldown) = rule.cooldown {
                if let Some(last) = self.last_fired[i] {
                    if now.duration_since(last) < cooldown {
                        continue;
                    }
                }
            }

            if self.check_trigger(&rule.trigger, memory) {
                self.last_fired[i] = Some(now);
                return Some(rule.action.clone());
            }
        }
        None
    }

    fn check_trigger(&self, trigger: &HealingTrigger, memory: &AgentMemory) -> bool {
        match trigger {
            HealingTrigger::ToolError(substr) => {
                // Check if the most recent tool call failed with an error containing substr
                if let Some(last) = memory.history.last() {
                    if !last.success && last.observation.contains(substr.as_str()) {
                        return true;
                    }
                }
                false
            }
            HealingTrigger::RepeatedToolCall(n) => {
                let len = memory.history.len();
                if len < *n {
                    return false;
                }
                let last = &memory.history[len - 1];
                let count = memory
                    .history
                    .iter()
                    .rev()
                    .take(*n)
                    .filter(|h| h.tool.name == last.tool.name)
                    .count();
                count >= *n
            }
            HealingTrigger::ConsecutiveFailures(n) => {
                let len = memory.history.len();
                if len < *n {
                    return false;
                }
                memory.history.iter().rev().take(*n).all(|h| !h.success)
            }
            HealingTrigger::ConfidenceBelow(threshold) => memory.confidence_score < *threshold,
            HealingTrigger::BudgetPctAbove(pct) => {
                if let Some(ref budget) = memory.budget {
                    if let Some(max) = budget.max_total_tokens {
                        let used = memory.total_usage.total_tokens as f64;
                        let max_f = max as f64;
                        if max_f > 0.0 {
                            return (used / max_f) > *pct;
                        }
                    }
                }
                false
            }
            HealingTrigger::LoopDetected => {
                // Check anomaly_notes for loop indicators
                memory.anomaly_notes.iter().any(|note| {
                    let lower = note.to_lowercase();
                    lower.contains("loop")
                        || lower.contains("repeated")
                        || lower.contains("same")
                        || lower.contains("times with similar")
                        || lower.contains("oscillating")
                })
            }
        }
    }
}

impl Default for HealingPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HealingOutcome — result that engine acts on
// ─────────────────────────────────────────────────────────────────────────────

/// What the engine should do after a healing action is applied.
#[derive(Debug, Clone)]
pub enum HealingOutcome {
    /// Continue normal execution.
    Continue,
    /// Force the agent to finish with current best answer.
    ForceFinish,
    /// Retry current step.
    Retry,
}

/// Apply a healing action to the agent memory. Returns what the engine should do next.
pub fn apply_healing(action: &HealingAction, memory: &mut AgentMemory) -> HealingOutcome {
    match action {
        HealingAction::RetryWithBackoff { .. } => {
            // The actual backoff delay is handled by the engine/caller.
            // We just signal a retry.
            memory
                .anomaly_notes
                .push("Self-healing: retrying with backoff.".to_string());
            HealingOutcome::Retry
        }
        HealingAction::BlacklistAndContinue => {
            if let Some(last) = memory.history.last() {
                let tool_name = last.tool.name.clone();
                memory.blacklisted_tools.insert(tool_name.clone());
                memory.anomaly_notes.push(format!(
                    "Self-healing: blacklisted tool '{}'. Try a different approach.",
                    tool_name
                ));
            }
            HealingOutcome::Continue
        }
        HealingAction::SwitchModel(model) => {
            memory
                .anomaly_notes
                .push(format!("Self-healing: switching model to '{}'.", model));
            // The model switch is signaled through anomaly_notes; the routing
            // policy or planning state reads it. We also store a hint in config.
            memory
                .config
                .models
                .insert("default".to_string(), model.clone());
            HealingOutcome::Continue
        }
        HealingAction::ReflectAndRetry => {
            memory.anomaly_notes.push(
                "Self-healing: reflecting on failures before retrying. Review your approach and try something different.".to_string(),
            );
            HealingOutcome::Retry
        }
        HealingAction::SummarizeAndFinish => {
            // Set a final answer from the best available info
            if memory.final_answer.is_none() {
                let summary = if let Some(last) = memory.history.last() {
                    format!(
                        "Based on {} steps of analysis: {}",
                        memory.step, last.observation
                    )
                } else {
                    format!("Task completed after {} steps.", memory.step)
                };
                memory.final_answer = Some(summary);
            }
            memory
                .anomaly_notes
                .push("Self-healing: summarizing findings and finishing.".to_string());
            HealingOutcome::ForceFinish
        }
        HealingAction::BreakAndContinue => {
            memory.anomaly_notes.push(
                "Self-healing: detected loop. Breaking pattern — try a completely different strategy.".to_string(),
            );
            // Clear recent anomaly notes about loops to avoid re-triggering
            memory.anomaly_notes.retain(|n| {
                let lower = n.to_lowercase();
                !lower.contains("loop") || lower.contains("self-healing")
            });
            HealingOutcome::Continue
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::AgentMemory;
    use crate::types::{HistoryEntry, ToolCall};

    fn make_memory() -> AgentMemory {
        AgentMemory::new("test task")
    }

    fn add_tool_call(memory: &mut AgentMemory, name: &str, success: bool, obs: &str) {
        memory.history.push(HistoryEntry {
            step: memory.step,
            tool: ToolCall {
                id: None,
                name: name.to_string(),
                args: Default::default(),
            },
            observation: obs.to_string(),
            success,
        });
    }

    #[test]
    fn test_no_trigger_on_fresh_state() {
        let mut policy =
            HealingPolicy::new().on_consecutive_failures(3, HealingAction::ReflectAndRetry);
        let m = make_memory();
        assert!(policy.evaluate(&m).is_none());
    }

    #[test]
    fn test_tool_error_trigger() {
        let mut policy = HealingPolicy::new().on_tool_error(
            "rate_limit",
            HealingAction::RetryWithBackoff {
                max: 3,
                base_ms: 500,
            },
        );
        let mut m = make_memory();
        add_tool_call(&mut m, "api_call", false, "Error: rate_limit exceeded");
        let action = policy.evaluate(&m);
        assert!(matches!(
            action,
            Some(HealingAction::RetryWithBackoff { .. })
        ));
    }

    #[test]
    fn test_tool_error_no_match() {
        let mut policy = HealingPolicy::new().on_tool_error(
            "rate_limit",
            HealingAction::RetryWithBackoff {
                max: 3,
                base_ms: 500,
            },
        );
        let mut m = make_memory();
        add_tool_call(&mut m, "api_call", false, "Error: not_found");
        assert!(policy.evaluate(&m).is_none());
    }

    #[test]
    fn test_repeated_tool_call_trigger() {
        let mut policy =
            HealingPolicy::new().on_repeated_tool_call(3, HealingAction::BlacklistAndContinue);
        let mut m = make_memory();
        for _ in 0..3 {
            add_tool_call(&mut m, "search", true, "result");
        }
        let action = policy.evaluate(&m);
        assert!(matches!(action, Some(HealingAction::BlacklistAndContinue)));
    }

    #[test]
    fn test_consecutive_failures_trigger() {
        let mut policy =
            HealingPolicy::new().on_consecutive_failures(3, HealingAction::SummarizeAndFinish);
        let mut m = make_memory();
        for _ in 0..3 {
            add_tool_call(&mut m, "search", false, "failed");
        }
        let action = policy.evaluate(&m);
        assert!(matches!(action, Some(HealingAction::SummarizeAndFinish)));
    }

    #[test]
    fn test_consecutive_failures_broken_by_success() {
        let mut policy =
            HealingPolicy::new().on_consecutive_failures(3, HealingAction::SummarizeAndFinish);
        let mut m = make_memory();
        add_tool_call(&mut m, "search", false, "failed");
        add_tool_call(&mut m, "search", true, "ok");
        add_tool_call(&mut m, "search", false, "failed");
        assert!(policy.evaluate(&m).is_none());
    }

    #[test]
    fn test_confidence_below_trigger() {
        let mut policy = HealingPolicy::new()
            .on_confidence_below(0.3, HealingAction::SwitchModel("gpt-4o".into()));
        let mut m = make_memory();
        m.confidence_score = 0.2;
        let action = policy.evaluate(&m);
        assert!(matches!(action, Some(HealingAction::SwitchModel(_))));
    }

    #[test]
    fn test_budget_pct_trigger() {
        let mut policy =
            HealingPolicy::new().on_budget_pct_above(0.9, HealingAction::SummarizeAndFinish);
        let mut m = make_memory();
        m.budget = Some(crate::budget::TokenBudget {
            max_total_tokens: Some(1000),
            max_input_tokens: None,
            max_output_tokens: None,
        });
        m.total_usage.total_tokens = 950; // 95%
        let action = policy.evaluate(&m);
        assert!(matches!(action, Some(HealingAction::SummarizeAndFinish)));
    }

    #[test]
    fn test_loop_detected_trigger() {
        let mut policy = HealingPolicy::new().on_loop_detected(HealingAction::BreakAndContinue);
        let mut m = make_memory();
        m.anomaly_notes.push(
            "WARNING: You called 'search' 4 times with similar args. Try a different approach."
                .to_string(),
        );
        let action = policy.evaluate(&m);
        assert!(matches!(action, Some(HealingAction::BreakAndContinue)));
    }

    #[test]
    fn test_first_match_wins() {
        let mut policy = HealingPolicy::new()
            .on_consecutive_failures(2, HealingAction::ReflectAndRetry)
            .on_consecutive_failures(3, HealingAction::SummarizeAndFinish);
        let mut m = make_memory();
        for _ in 0..3 {
            add_tool_call(&mut m, "search", false, "failed");
        }
        // First rule matches (2 consecutive failures)
        let action = policy.evaluate(&m);
        assert!(matches!(action, Some(HealingAction::ReflectAndRetry)));
    }

    #[test]
    fn test_apply_blacklist() {
        let mut m = make_memory();
        add_tool_call(&mut m, "dangerous_tool", false, "error");
        let outcome = apply_healing(&HealingAction::BlacklistAndContinue, &mut m);
        assert!(matches!(outcome, HealingOutcome::Continue));
        assert!(m.blacklisted_tools.contains("dangerous_tool"));
    }

    #[test]
    fn test_apply_summarize_and_finish() {
        let mut m = make_memory();
        m.step = 5;
        add_tool_call(&mut m, "search", true, "Found: the answer is 42");
        let outcome = apply_healing(&HealingAction::SummarizeAndFinish, &mut m);
        assert!(matches!(outcome, HealingOutcome::ForceFinish));
        assert!(m.final_answer.is_some());
    }

    #[test]
    fn test_apply_switch_model() {
        let mut m = make_memory();
        let outcome = apply_healing(&HealingAction::SwitchModel("gpt-4o".into()), &mut m);
        assert!(matches!(outcome, HealingOutcome::Continue));
        assert_eq!(m.config.models.get("default").unwrap(), "gpt-4o");
    }
}
