//! Agent Forking — speculative execution with parallel paths.
//!
//! The agent can fork into N parallel branches, each exploring a different
//! approach. Results are scored and the best is selected via a merge strategy.

use crate::memory::AgentMemory;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// ForkScorer
// ─────────────────────────────────────────────────────────────────────────────

/// Score a completed fork. Higher = better. Range: 0.0..=1.0
pub trait ForkScorer: Send + Sync {
    fn score(&self, memory: &AgentMemory) -> f64;
}

/// Scores by the agent's final confidence value.
#[derive(Debug, Clone)]
pub struct ConfidenceScorer;

impl ForkScorer for ConfidenceScorer {
    fn score(&self, memory: &AgentMemory) -> f64 {
        memory.confidence_score.clamp(0.0, 1.0)
    }
}

/// Scores by the fraction of tool calls that succeeded.
#[derive(Debug, Clone)]
pub struct ToolSuccessRateScorer;

impl ForkScorer for ToolSuccessRateScorer {
    fn score(&self, memory: &AgentMemory) -> f64 {
        if memory.history.is_empty() {
            return 0.5;
        }
        let successes = memory.history.iter().filter(|h| h.success).count();
        successes as f64 / memory.history.len() as f64
    }
}

/// Scores by efficiency: quality per step used.
/// quality = confidence * success_rate, efficiency = quality / steps
#[derive(Debug, Clone)]
pub struct StepEfficiencyScorer;

impl ForkScorer for StepEfficiencyScorer {
    fn score(&self, memory: &AgentMemory) -> f64 {
        let success_rate = if memory.history.is_empty() {
            0.5
        } else {
            let s = memory.history.iter().filter(|h| h.success).count();
            s as f64 / memory.history.len() as f64
        };
        let quality = memory.confidence_score * success_rate;
        if memory.step == 0 {
            return quality;
        }
        // Normalize: fewer steps = higher score
        // Use 1 / (1 + steps/10) as a decay factor
        let step_factor = 1.0 / (1.0 + memory.step as f64 / 10.0);
        (quality * step_factor).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MergeStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// How to select/combine results from multiple forks.
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Return answer from highest-scoring fork.
    BestFinalAnswer,
    /// Return fork with highest average confidence.
    MostConfident,
    /// Return first fork that reaches Done (by completion order).
    FirstToFinish,
    /// Answer with most agreement across forks.
    ConsensusMajority,
}

// ─────────────────────────────────────────────────────────────────────────────
// ForkConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for agent forking.
#[derive(Clone)]
pub struct ForkConfig {
    /// Number of parallel branches to spawn.
    pub num_branches: usize,
    /// Max steps per fork before scoring.
    pub max_depth: usize,
    /// Scoring function for each fork.
    pub scorer: Arc<dyn ForkScorer>,
    /// How to merge/select results.
    pub merge: MergeStrategy,
}

impl std::fmt::Debug for ForkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForkConfig")
            .field("num_branches", &self.num_branches)
            .field("max_depth", &self.max_depth)
            .field("merge", &self.merge)
            .finish()
    }
}

impl ForkConfig {
    pub fn new(num_branches: usize) -> Self {
        Self {
            num_branches,
            max_depth: 20,
            scorer: Arc::new(ConfidenceScorer),
            merge: MergeStrategy::BestFinalAnswer,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn scorer(mut self, scorer: Arc<dyn ForkScorer>) -> Self {
        self.scorer = scorer;
        self
    }

    pub fn merge(mut self, strategy: MergeStrategy) -> Self {
        self.merge = strategy;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ForkResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result from a single fork execution.
#[derive(Debug, Clone)]
pub struct ForkResult {
    pub branch_id: usize,
    pub answer: Option<String>,
    pub score: f64,
    pub steps_taken: usize,
    pub memory: AgentMemory,
    pub error: Option<String>,
}

/// Select the best result from multiple fork results.
pub fn select_best(results: &mut Vec<ForkResult>, strategy: &MergeStrategy) -> Option<ForkResult> {
    if results.is_empty() {
        return None;
    }

    match strategy {
        MergeStrategy::BestFinalAnswer => {
            // Pick the one with highest score that has an answer
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results
                .iter()
                .find(|r| r.answer.is_some())
                .cloned()
                .or_else(|| results.first().cloned())
        }
        MergeStrategy::MostConfident => {
            results.sort_by(|a, b| {
                b.memory
                    .confidence_score
                    .partial_cmp(&a.memory.confidence_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.first().cloned()
        }
        MergeStrategy::FirstToFinish => {
            // Results are already in completion order
            results
                .iter()
                .find(|r| r.answer.is_some())
                .cloned()
                .or_else(|| results.first().cloned())
        }
        MergeStrategy::ConsensusMajority => {
            // Group by answer, pick the one with most occurrences
            use std::collections::HashMap;
            let mut counts: HashMap<String, (usize, f64)> = HashMap::new();
            for r in results.iter() {
                if let Some(ref ans) = r.answer {
                    let entry = counts.entry(ans.clone()).or_insert((0, 0.0));
                    entry.0 += 1;
                    if r.score > entry.1 {
                        entry.1 = r.score;
                    }
                }
            }
            if let Some((best_answer, _)) = counts.iter().max_by_key(|(_, (count, _))| *count) {
                results
                    .iter()
                    .find(|r| r.answer.as_ref() == Some(best_answer))
                    .cloned()
            } else {
                results.first().cloned()
            }
        }
    }
}

/// Fork agent memory into N independent branches.
/// Each branch gets a clean trace and reset step counter.
pub fn fork_memory(
    memory: &AgentMemory,
    num_branches: usize,
    max_depth: usize,
) -> Vec<AgentMemory> {
    (0..num_branches)
        .map(|i| {
            let mut forked = memory.clone();
            // Reset step counter for this fork so it runs at most max_depth steps
            forked.config.max_steps = forked.step + max_depth;
            // Clear anomaly notes for fresh start
            forked.anomaly_notes.clear();
            // Tag the fork
            forked.anomaly_notes.push(format!(
                "You are fork branch {} of {}. Explore a distinct approach.",
                i + 1,
                num_branches
            ));
            forked
        })
        .collect()
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

    fn add_tool_call(memory: &mut AgentMemory, name: &str, success: bool) {
        memory.history.push(HistoryEntry {
            step: memory.step,
            tool: ToolCall {
                id: None,
                name: name.to_string(),
                args: Default::default(),
            },
            observation: "result".to_string(),
            success,
        });
    }

    #[test]
    fn test_confidence_scorer() {
        let mut m = make_memory();
        m.confidence_score = 0.85;
        assert!((ConfidenceScorer.score(&m) - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tool_success_rate_scorer() {
        let mut m = make_memory();
        add_tool_call(&mut m, "search", true);
        add_tool_call(&mut m, "search", true);
        add_tool_call(&mut m, "search", false);
        let score = ToolSuccessRateScorer.score(&m);
        assert!((score - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_step_efficiency_scorer() {
        let mut m = make_memory();
        m.confidence_score = 1.0;
        m.step = 0;
        add_tool_call(&mut m, "search", true);
        let s1 = StepEfficiencyScorer.score(&m);

        let mut m2 = make_memory();
        m2.confidence_score = 1.0;
        m2.step = 20;
        add_tool_call(&mut m2, "search", true);
        let s2 = StepEfficiencyScorer.score(&m2);

        // Fewer steps should score higher
        assert!(s1 > s2);
    }

    #[test]
    fn test_fork_config_builder() {
        let config = ForkConfig::new(3)
            .max_depth(10)
            .merge(MergeStrategy::MostConfident);
        assert_eq!(config.num_branches, 3);
        assert_eq!(config.max_depth, 10);
        assert!(matches!(config.merge, MergeStrategy::MostConfident));
    }

    #[test]
    fn test_fork_memory() {
        let m = make_memory();
        let forks = fork_memory(&m, 3, 10);
        assert_eq!(forks.len(), 3);
        // Each fork should have its own tag
        assert!(forks[0].anomaly_notes[0].contains("fork branch 1"));
        assert!(forks[1].anomaly_notes[0].contains("fork branch 2"));
        assert!(forks[2].anomaly_notes[0].contains("fork branch 3"));
        // Max steps should be offset
        assert_eq!(forks[0].config.max_steps, 10);
    }

    #[test]
    fn test_select_best_by_score() {
        let mut results = vec![
            ForkResult {
                branch_id: 0,
                answer: Some("answer A".into()),
                score: 0.6,
                steps_taken: 5,
                memory: make_memory(),
                error: None,
            },
            ForkResult {
                branch_id: 1,
                answer: Some("answer B".into()),
                score: 0.9,
                steps_taken: 3,
                memory: make_memory(),
                error: None,
            },
        ];
        let best = select_best(&mut results, &MergeStrategy::BestFinalAnswer).unwrap();
        assert_eq!(best.answer.unwrap(), "answer B");
    }

    #[test]
    fn test_select_best_most_confident() {
        let mut m1 = make_memory();
        m1.confidence_score = 0.5;
        let mut m2 = make_memory();
        m2.confidence_score = 0.95;

        let mut results = vec![
            ForkResult {
                branch_id: 0,
                answer: Some("low conf".into()),
                score: 0.5,
                steps_taken: 5,
                memory: m1,
                error: None,
            },
            ForkResult {
                branch_id: 1,
                answer: Some("high conf".into()),
                score: 0.9,
                steps_taken: 3,
                memory: m2,
                error: None,
            },
        ];
        let best = select_best(&mut results, &MergeStrategy::MostConfident).unwrap();
        assert_eq!(best.answer.unwrap(), "high conf");
    }

    #[test]
    fn test_select_consensus() {
        let mut results = vec![
            ForkResult {
                branch_id: 0,
                answer: Some("42".into()),
                score: 0.7,
                steps_taken: 5,
                memory: make_memory(),
                error: None,
            },
            ForkResult {
                branch_id: 1,
                answer: Some("42".into()),
                score: 0.8,
                steps_taken: 4,
                memory: make_memory(),
                error: None,
            },
            ForkResult {
                branch_id: 2,
                answer: Some("99".into()),
                score: 0.9,
                steps_taken: 3,
                memory: make_memory(),
                error: None,
            },
        ];
        let best = select_best(&mut results, &MergeStrategy::ConsensusMajority).unwrap();
        // "42" appears twice vs "99" once
        assert_eq!(best.answer.unwrap(), "42");
    }

    #[test]
    fn test_select_no_answers() {
        let mut results = vec![ForkResult {
            branch_id: 0,
            answer: None,
            score: 0.3,
            steps_taken: 5,
            memory: make_memory(),
            error: Some("failed".into()),
        }];
        let best = select_best(&mut results, &MergeStrategy::BestFinalAnswer);
        assert!(best.is_some()); // Returns the failed fork as best effort
    }

    #[test]
    fn test_select_empty_results() {
        let mut results: Vec<ForkResult> = vec![];
        let best = select_best(&mut results, &MergeStrategy::BestFinalAnswer);
        assert!(best.is_none());
    }
}
