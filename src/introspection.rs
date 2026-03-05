//! Agent Introspection / Meta-Reasoning — runtime anomaly detection.
//!
//! The `IntrospectionEngine` monitors execution patterns and detects:
//! - **Loops**: repeated identical or similar tool calls
//! - **Thrashing**: alternating between two tools without progress
//! - **Stalls**: N steps without new information / progress
//! - **Budget critical**: token budget nearing exhaustion
//! - **Confidence degradation**: confidence trending downward over time

use crate::memory::AgentMemory;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Anomaly types
// ─────────────────────────────────────────────────────────────────────────────

/// A detected pathological pattern in agent execution.
#[derive(Debug, Clone)]
pub enum Anomaly {
    /// Same tool+args called repeatedly.
    LoopDetected { tool: String, repeated: usize },
    /// Agent is stuck — N steps without any new successful tool result.
    ProgressStalled { steps_without_new_info: usize },
    /// Token budget approaching exhaustion.
    BudgetCritical { pct_used: f64 },
    /// Agent alternating between 2 tools without convergence.
    ThrashingDetected { tools_alternating: Vec<String> },
    /// Confidence trending downward over recent steps.
    ConfidenceDegrading { trend: f64 },
}

impl Anomaly {
    /// One-line human-readable note suitable for injection into LLM context.
    pub fn to_note(&self) -> String {
        match self {
            Anomaly::LoopDetected { tool, repeated } => {
                format!("WARNING: You called '{}' {} times with similar args. Try a different approach.", tool, repeated)
            }
            Anomaly::ProgressStalled {
                steps_without_new_info,
            } => {
                format!(
                    "WARNING: No new information in {} steps. Consider changing strategy.",
                    steps_without_new_info
                )
            }
            Anomaly::BudgetCritical { pct_used } => {
                format!(
                    "WARNING: {:.0}% of token budget consumed. Wrap up soon.",
                    pct_used * 100.0
                )
            }
            Anomaly::ThrashingDetected { tools_alternating } => {
                format!(
                    "WARNING: Oscillating between tools: {}. Break the pattern.",
                    tools_alternating.join(" ↔ ")
                )
            }
            Anomaly::ConfidenceDegrading { trend } => {
                format!(
                    "WARNING: Confidence declining (slope={:.2}). Reconsider your approach.",
                    trend
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the introspection engine.
#[derive(Debug, Clone)]
pub struct IntrospectionConfig {
    /// Detect loops when same tool is called N times in a window.
    pub loop_window: usize,
    /// Minimum repetitions to trigger loop anomaly.
    pub loop_threshold: usize,
    /// Detect stall when N consecutive steps have no successful tool result.
    pub stall_threshold: usize,
    /// Warn when budget usage exceeds this fraction.
    pub budget_warn_pct: f64,
    /// Detect thrashing when 2 tools alternate N times.
    pub thrash_window: usize,
    /// Confidence degradation: sliding window for trend calculation.
    pub confidence_window: usize,
    /// Confidence slope threshold (negative value) to trigger anomaly.
    pub confidence_slope_threshold: f64,
}

impl Default for IntrospectionConfig {
    fn default() -> Self {
        Self {
            loop_window: 6,
            loop_threshold: 3,
            stall_threshold: 4,
            budget_warn_pct: 0.75,
            thrash_window: 6,
            confidence_window: 5,
            confidence_slope_threshold: -0.1,
        }
    }
}

impl IntrospectionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn loop_detection(mut self, window: usize, threshold: usize) -> Self {
        self.loop_window = window;
        self.loop_threshold = threshold;
        self
    }

    pub fn stall_detection(mut self, threshold: usize) -> Self {
        self.stall_threshold = threshold;
        self
    }

    pub fn budget_awareness(mut self, warn_pct: f64) -> Self {
        self.budget_warn_pct = warn_pct;
        self
    }

    pub fn thrash_detection(mut self, window: usize) -> Self {
        self.thrash_window = window;
        self
    }

    pub fn confidence_tracking(mut self, window: usize, slope_threshold: f64) -> Self {
        self.confidence_window = window;
        self.confidence_slope_threshold = slope_threshold;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IntrospectionEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Record of a tool call for pattern analysis.
#[derive(Debug, Clone)]
struct ToolCallRecord {
    tool_name: String,
    args_hash: u64,
    success: bool,
}

/// Runtime meta-reasoner that watches tool call patterns.
#[derive(Debug, Clone)]
pub struct IntrospectionEngine {
    config: IntrospectionConfig,
    call_history: VecDeque<ToolCallRecord>,
    confidence_history: VecDeque<f64>,
    max_history: usize,
}

impl IntrospectionEngine {
    pub fn new(config: IntrospectionConfig) -> Self {
        Self {
            config,
            call_history: VecDeque::with_capacity(32),
            confidence_history: VecDeque::with_capacity(16),
            max_history: 30,
        }
    }

    /// Record a tool call for pattern analysis.
    pub fn record_tool_call(&mut self, tool_name: &str, args: &serde_json::Value, success: bool) {
        // Simple hash of args for equality comparison
        let args_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            args.to_string().hash(&mut hasher);
            hasher.finish()
        };

        if self.call_history.len() >= self.max_history {
            self.call_history.pop_front();
        }
        self.call_history.push_back(ToolCallRecord {
            tool_name: tool_name.to_string(),
            args_hash,
            success,
        });
    }

    /// Record the current confidence score.
    pub fn record_confidence(&mut self, score: f64) {
        if self.confidence_history.len() >= self.config.confidence_window * 2 {
            self.confidence_history.pop_front();
        }
        self.confidence_history.push_back(score);
    }

    /// Analyze current execution state. Returns all detected anomalies.
    pub fn analyze(&mut self, memory: &AgentMemory) -> Vec<Anomaly> {
        // Record confidence for this step
        self.record_confidence(memory.confidence_score);

        // Record latest tool calls from memory history that we haven't seen
        let history_len = memory.history.len();
        let tracked = self.call_history.len();
        if history_len > tracked {
            // Only record new entries
            for entry in memory.history.iter().skip(tracked) {
                let args_val = serde_json::to_value(&entry.tool.args).unwrap_or_default();
                self.record_tool_call(&entry.tool.name, &args_val, entry.success);
            }
        }

        let mut anomalies = Vec::new();

        // 1. Loop detection
        if let Some(anomaly) = self.detect_loop() {
            anomalies.push(anomaly);
        }

        // 2. Progress stall
        if let Some(anomaly) = self.detect_stall() {
            anomalies.push(anomaly);
        }

        // 3. Budget critical
        if let Some(anomaly) = self.detect_budget_critical(memory) {
            anomalies.push(anomaly);
        }

        // 4. Thrashing
        if let Some(anomaly) = self.detect_thrashing() {
            anomalies.push(anomaly);
        }

        // 5. Confidence degradation
        if let Some(anomaly) = self.detect_confidence_degradation() {
            anomalies.push(anomaly);
        }

        anomalies
    }

    // ── Detectors ────────────────────────────────────────────────────────────

    fn detect_loop(&self) -> Option<Anomaly> {
        if self.call_history.len() < self.config.loop_threshold {
            return None;
        }

        let window: Vec<_> = self
            .call_history
            .iter()
            .rev()
            .take(self.config.loop_window)
            .collect();

        // Check if the most recent tool+args appears N times
        if let Some(last) = window.first() {
            let count = window
                .iter()
                .filter(|r| r.tool_name == last.tool_name && r.args_hash == last.args_hash)
                .count();
            if count >= self.config.loop_threshold {
                return Some(Anomaly::LoopDetected {
                    tool: last.tool_name.clone(),
                    repeated: count,
                });
            }
        }
        None
    }

    fn detect_stall(&self) -> Option<Anomaly> {
        if self.call_history.len() < self.config.stall_threshold {
            return None;
        }

        let recent: Vec<_> = self
            .call_history
            .iter()
            .rev()
            .take(self.config.stall_threshold)
            .collect();

        let all_failed = recent.iter().all(|r| !r.success);
        if all_failed {
            return Some(Anomaly::ProgressStalled {
                steps_without_new_info: recent.len(),
            });
        }
        None
    }

    fn detect_budget_critical(&self, memory: &AgentMemory) -> Option<Anomaly> {
        if let Some(ref budget) = memory.budget {
            if let Some(max) = budget.max_total_tokens {
                let used = memory.total_usage.total_tokens as f64;
                let max_f = max as f64;
                if max_f > 0.0 {
                    let pct = used / max_f;
                    if pct > self.config.budget_warn_pct {
                        return Some(Anomaly::BudgetCritical { pct_used: pct });
                    }
                }
            }
        }
        None
    }

    fn detect_thrashing(&self) -> Option<Anomaly> {
        if self.call_history.len() < self.config.thrash_window {
            return None;
        }

        let recent: Vec<_> = self
            .call_history
            .iter()
            .rev()
            .take(self.config.thrash_window)
            .map(|r| r.tool_name.as_str())
            .collect();

        // Check for A-B-A-B pattern
        if recent.len() >= 4 {
            let a = recent[0];
            let b = recent[1];
            if a != b {
                let alternating =
                    recent.iter().enumerate().all(
                        |(i, name)| {
                            if i % 2 == 0 {
                                *name == a
                            } else {
                                *name == b
                            }
                        },
                    );
                if alternating {
                    return Some(Anomaly::ThrashingDetected {
                        tools_alternating: vec![a.to_string(), b.to_string()],
                    });
                }
            }
        }
        None
    }

    fn detect_confidence_degradation(&self) -> Option<Anomaly> {
        let n = self.confidence_history.len();
        if n < self.config.confidence_window {
            return None;
        }

        // Simple linear regression slope over last N confidence values
        let window: Vec<f64> = self
            .confidence_history
            .iter()
            .rev()
            .take(self.config.confidence_window)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let n_f = window.len() as f64;
        let mean_x = (n_f - 1.0) / 2.0;
        let mean_y: f64 = window.iter().sum::<f64>() / n_f;

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, y) in window.iter().enumerate() {
            let x = i as f64;
            num += (x - mean_x) * (y - mean_y);
            den += (x - mean_x) * (x - mean_x);
        }

        if den.abs() < f64::EPSILON {
            return None;
        }

        let slope = num / den;
        if slope < self.config.confidence_slope_threshold {
            return Some(Anomaly::ConfidenceDegrading { trend: slope });
        }
        None
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

    fn make_tool_call(name: &str, success: bool) -> HistoryEntry {
        HistoryEntry {
            step: 0,
            tool: ToolCall {
                id: None,
                name: name.to_string(),
                args: Default::default(),
            },
            observation: if success { "ok" } else { "ERROR" }.into(),
            success,
        }
    }

    #[test]
    fn test_no_anomalies_on_fresh_state() {
        let mut engine = IntrospectionEngine::new(IntrospectionConfig::default());
        let m = make_memory();
        let anomalies = engine.analyze(&m);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_loop_detection() {
        let mut engine = IntrospectionEngine::new(IntrospectionConfig::new().loop_detection(6, 3));
        let mut m = make_memory();
        // Add 3 identical tool calls
        for _ in 0..3 {
            m.history.push(make_tool_call("search", true));
        }
        let anomalies = engine.analyze(&m);
        assert!(anomalies.iter().any(|a| matches!(a, Anomaly::LoopDetected { tool, repeated } if tool == "search" && *repeated >= 3)));
    }

    #[test]
    fn test_no_loop_with_different_tools() {
        let mut engine = IntrospectionEngine::new(IntrospectionConfig::new().loop_detection(6, 3));
        let mut m = make_memory();
        m.history.push(make_tool_call("search", true));
        m.history.push(make_tool_call("read", true));
        m.history.push(make_tool_call("write", true));
        let anomalies = engine.analyze(&m);
        assert!(!anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::LoopDetected { .. })));
    }

    #[test]
    fn test_stall_detection() {
        let mut engine = IntrospectionEngine::new(IntrospectionConfig::new().stall_detection(3));
        let mut m = make_memory();
        // 3 consecutive failures
        for _ in 0..3 {
            m.history.push(make_tool_call("search", false));
        }
        let anomalies = engine.analyze(&m);
        assert!(anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::ProgressStalled { .. })));
    }

    #[test]
    fn test_no_stall_with_successes() {
        let mut engine = IntrospectionEngine::new(IntrospectionConfig::new().stall_detection(3));
        let mut m = make_memory();
        m.history.push(make_tool_call("search", false));
        m.history.push(make_tool_call("search", true)); // breaks the stall
        m.history.push(make_tool_call("search", false));
        let anomalies = engine.analyze(&m);
        assert!(!anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::ProgressStalled { .. })));
    }

    #[test]
    fn test_budget_critical() {
        let mut engine =
            IntrospectionEngine::new(IntrospectionConfig::new().budget_awareness(0.75));
        let mut m = make_memory();
        m.budget = Some(crate::budget::TokenBudget {
            max_total_tokens: Some(1000),
            max_input_tokens: None,
            max_output_tokens: None,
        });
        m.total_usage.total_tokens = 800; // 80% > 75%
        let anomalies = engine.analyze(&m);
        assert!(anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::BudgetCritical { pct_used } if *pct_used > 0.75)));
    }

    #[test]
    fn test_thrashing_detection() {
        let mut engine = IntrospectionEngine::new(IntrospectionConfig::new().thrash_detection(6));
        let mut m = make_memory();
        // A-B-A-B-A-B pattern
        for _ in 0..3 {
            m.history.push(make_tool_call("search", true));
            m.history.push(make_tool_call("read", true));
        }
        let anomalies = engine.analyze(&m);
        assert!(anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::ThrashingDetected { .. })));
    }

    #[test]
    fn test_confidence_degradation() {
        let mut engine =
            IntrospectionEngine::new(IntrospectionConfig::new().confidence_tracking(5, -0.1));
        let mut m = make_memory();
        // Simulate degrading confidence over 5 steps
        let scores = [0.9, 0.75, 0.6, 0.45, 0.3];
        for (i, &score) in scores.iter().enumerate() {
            m.confidence_score = score;
            m.history.push(make_tool_call("search", true));
            m.history[i].step = i;
            let anomalies = engine.analyze(&m);
            if i == scores.len() - 1 {
                // Should detect degradation on the last step
                assert!(anomalies
                    .iter()
                    .any(|a| matches!(a, Anomaly::ConfidenceDegrading { .. })));
            }
        }
    }

    #[test]
    fn test_anomaly_to_note() {
        let anomaly = Anomaly::LoopDetected {
            tool: "search".to_string(),
            repeated: 3,
        };
        let note = anomaly.to_note();
        assert!(note.contains("search"));
        assert!(note.contains("3 times"));
    }

    #[test]
    fn test_multiple_anomalies_simultaneously() {
        let mut engine = IntrospectionEngine::new(
            IntrospectionConfig::new()
                .loop_detection(4, 3)
                .stall_detection(3),
        );
        let mut m = make_memory();
        // 3 identical failing calls → both loop and stall
        for _ in 0..3 {
            m.history.push(make_tool_call("search", false));
        }
        let anomalies = engine.analyze(&m);
        assert!(anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::LoopDetected { .. })));
        assert!(anomalies
            .iter()
            .any(|a| matches!(a, Anomaly::ProgressStalled { .. })));
    }
}
