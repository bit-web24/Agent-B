//! Plan-and-Execute — structured, inspectable, revisable execution plans.
//!
//! The agent creates an explicit step-by-step plan, executes each step,
//! and can revise the plan when conditions change.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// PlanStep
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a single plan step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    InProgress,
    Done,
    Failed,
    Skipped,
}

/// A single step in the agent's plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: usize,
    pub description: String,
    /// Hint about which tool to use (optional).
    pub tool_hint: Option<String>,
    pub status: StepStatus,
    /// Result of executing this step.
    pub result: Option<String>,
}

impl PlanStep {
    pub fn new(id: usize, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            tool_hint: None,
            status: StepStatus::Pending,
            result: None,
        }
    }

    pub fn with_tool_hint(mut self, tool: impl Into<String>) -> Self {
        self.tool_hint = Some(tool.into());
        self
    }

    pub fn is_actionable(&self) -> bool {
        matches!(self.status, StepStatus::Pending | StepStatus::InProgress)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AgentPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A structured execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPlan {
    pub steps: Vec<PlanStep>,
    pub current_step: usize,
    pub revision_count: usize,
    pub created_at: DateTime<Utc>,
}

impl AgentPlan {
    pub fn new(steps: Vec<PlanStep>) -> Self {
        Self {
            steps,
            current_step: 0,
            revision_count: 0,
            created_at: Utc::now(),
        }
    }

    /// Create a plan from a list of description strings.
    pub fn from_descriptions(descriptions: Vec<String>) -> Self {
        let steps = descriptions
            .into_iter()
            .enumerate()
            .map(|(i, desc)| PlanStep::new(i, desc))
            .collect();
        Self::new(steps)
    }

    /// Get the current step (if any remain).
    pub fn current(&self) -> Option<&PlanStep> {
        self.steps.get(self.current_step)
    }

    /// Get the current step mutably.
    pub fn current_mut(&mut self) -> Option<&mut PlanStep> {
        self.steps.get_mut(self.current_step)
    }

    /// Mark current step as in-progress.
    pub fn start_current(&mut self) {
        if let Some(step) = self.current_mut() {
            step.status = StepStatus::InProgress;
        }
    }

    /// Mark current step as done with a result and advance.
    pub fn complete_current(&mut self, result: String) {
        if let Some(step) = self.current_mut() {
            step.status = StepStatus::Done;
            step.result = Some(result);
        }
        self.current_step += 1;
    }

    /// Mark current step as failed and advance.
    pub fn fail_current(&mut self, reason: String) {
        if let Some(step) = self.current_mut() {
            step.status = StepStatus::Failed;
            step.result = Some(reason);
        }
        self.current_step += 1;
    }

    /// Skip current step and advance.
    pub fn skip_current(&mut self, reason: String) {
        if let Some(step) = self.current_mut() {
            step.status = StepStatus::Skipped;
            step.result = Some(reason);
        }
        self.current_step += 1;
    }

    /// Whether all steps have been executed.
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.steps.len()
    }

    /// Number of steps remaining.
    pub fn remaining(&self) -> usize {
        self.steps.len().saturating_sub(self.current_step)
    }

    /// Percentage complete (0.0 – 1.0).
    pub fn progress(&self) -> f64 {
        if self.steps.is_empty() {
            return 1.0;
        }
        self.current_step as f64 / self.steps.len() as f64
    }

    /// Revise the plan: replace remaining steps with new ones.
    /// Keeps already-completed steps intact.
    pub fn revise(&mut self, new_remaining: Vec<PlanStep>) {
        // Keep completed steps
        self.steps.truncate(self.current_step);
        // Re-number and append new steps
        let offset = self.steps.len();
        for (i, mut step) in new_remaining.into_iter().enumerate() {
            step.id = offset + i;
            self.steps.push(step);
        }
        self.revision_count += 1;
    }

    /// Compact summary for injection into LLM context.
    /// Shows current step + next 2 steps.
    pub fn to_context_string(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!(
            "PLAN ({}/{} steps, {} revisions):",
            self.current_step,
            self.steps.len(),
            self.revision_count
        ));

        for (i, step) in self.steps.iter().enumerate() {
            let marker = match step.status {
                StepStatus::Done => "✓",
                StepStatus::Failed => "✗",
                StepStatus::Skipped => "⊘",
                StepStatus::InProgress => "►",
                StepStatus::Pending => "○",
            };

            // Show completed steps concisely, current + next 2 in detail
            if i < self.current_step {
                parts.push(format!("  {} [{}] {}", marker, i, step.description));
            } else if i < self.current_step + 3 {
                let hint = step
                    .tool_hint
                    .as_ref()
                    .map(|t| format!(" (tool: {})", t))
                    .unwrap_or_default();
                parts.push(format!("  {} [{}] {}{}", marker, i, step.description, hint));
            }
        }

        if self.current_step + 3 < self.steps.len() {
            parts.push(format!(
                "  ... {} more steps",
                self.steps.len() - self.current_step - 3
            ));
        }

        parts.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanRevisionTrigger
// ─────────────────────────────────────────────────────────────────────────────

/// Conditions that trigger a plan revision.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanRevisionTrigger {
    /// Revise when a tool call fails.
    ToolFailure,
    /// Revise when observation contains new substantial information.
    NewInformation,
    /// Re-plan after every N completed steps.
    EveryNSteps(usize),
    /// Manual revision (agent or user requested).
    Manual,
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanningMode
// ─────────────────────────────────────────────────────────────────────────────

/// How the agent plans its execution.
#[derive(Debug, Clone)]
#[derive(Default)]
pub enum PlanningMode {
    /// Default: implicit plan via LLM context (existing behavior).
    #[default]
    Implicit,
    /// Explicit plan: agent generates a structured plan first.
    Explicit {
        max_plan_steps: usize,
        revision_triggers: Vec<PlanRevisionTrigger>,
    },
}


// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> AgentPlan {
        AgentPlan::from_descriptions(vec![
            "Search for information".into(),
            "Analyze results".into(),
            "Write summary".into(),
            "Review output".into(),
        ])
    }

    #[test]
    fn test_plan_creation() {
        let plan = sample_plan();
        assert_eq!(plan.steps.len(), 4);
        assert_eq!(plan.current_step, 0);
        assert_eq!(plan.revision_count, 0);
        assert!(!plan.is_complete());
        assert_eq!(plan.remaining(), 4);
    }

    #[test]
    fn test_plan_step_lifecycle() {
        let mut plan = sample_plan();

        // Start first step
        plan.start_current();
        assert_eq!(plan.current().unwrap().status, StepStatus::InProgress);

        // Complete first step
        plan.complete_current("Found 10 results".into());
        assert_eq!(plan.steps[0].status, StepStatus::Done);
        assert_eq!(plan.current_step, 1);
        assert_eq!(plan.remaining(), 3);
    }

    #[test]
    fn test_plan_fail_and_skip() {
        let mut plan = sample_plan();

        plan.fail_current("API error".into());
        assert_eq!(plan.steps[0].status, StepStatus::Failed);
        assert_eq!(plan.current_step, 1);

        plan.skip_current("Not needed".into());
        assert_eq!(plan.steps[1].status, StepStatus::Skipped);
        assert_eq!(plan.current_step, 2);
    }

    #[test]
    fn test_plan_completion() {
        let mut plan = sample_plan();
        for i in 0..4 {
            plan.complete_current(format!("Done step {}", i));
        }
        assert!(plan.is_complete());
        assert_eq!(plan.remaining(), 0);
        assert!((plan.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_progress() {
        let mut plan = sample_plan();
        assert!((plan.progress() - 0.0).abs() < f64::EPSILON);

        plan.complete_current("done".into());
        assert!((plan.progress() - 0.25).abs() < f64::EPSILON);

        plan.complete_current("done".into());
        assert!((plan.progress() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_revision() {
        let mut plan = sample_plan();
        // Complete first 2 steps
        plan.complete_current("result1".into());
        plan.complete_current("result2".into());
        assert_eq!(plan.current_step, 2);

        // Revise remaining steps
        plan.revise(vec![
            PlanStep::new(0, "New step A"),
            PlanStep::new(0, "New step B"),
        ]);

        assert_eq!(plan.steps.len(), 4); // 2 completed + 2 new
        assert_eq!(plan.revision_count, 1);
        assert_eq!(plan.steps[2].description, "New step A");
        assert_eq!(plan.steps[2].id, 2);
        assert_eq!(plan.steps[3].description, "New step B");
        assert_eq!(plan.steps[3].id, 3);
        assert_eq!(plan.remaining(), 2);
    }

    #[test]
    fn test_context_string() {
        let mut plan = sample_plan();
        plan.complete_current("done".into());
        plan.start_current();

        let ctx = plan.to_context_string();
        assert!(ctx.contains("PLAN"));
        assert!(ctx.contains("1/4 steps"));
        assert!(ctx.contains("✓"));
        assert!(ctx.contains("►"));
    }

    #[test]
    fn test_context_string_with_tool_hints() {
        let plan = AgentPlan::new(vec![
            PlanStep::new(0, "Search").with_tool_hint("web_search"),
            PlanStep::new(1, "Summarize").with_tool_hint("llm_summarize"),
        ]);
        let ctx = plan.to_context_string();
        assert!(ctx.contains("(tool: web_search)"));
    }

    #[test]
    fn test_empty_plan() {
        let plan = AgentPlan::from_descriptions(vec![]);
        assert!(plan.is_complete());
        assert_eq!(plan.remaining(), 0);
        assert!((plan.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_step_with_tool_hint() {
        let step = PlanStep::new(0, "Search for data").with_tool_hint("search");
        assert_eq!(step.tool_hint.as_deref(), Some("search"));
        assert!(step.is_actionable());
    }

    #[test]
    fn test_step_not_actionable_after_done() {
        let mut step = PlanStep::new(0, "Test");
        step.status = StepStatus::Done;
        assert!(!step.is_actionable());
    }
}
