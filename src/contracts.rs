//! Execution Contracts — declarative safety constraints on agent execution.
//!
//! Three contract types:
//! - `TransitionGuard`  — pre-condition checked before a specific state transition
//! - `Invariant`        — global condition checked after every step
//! - `PostCondition`    — checked before the final answer is emitted

use crate::memory::AgentMemory;
use crate::types::State;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A callable contract predicate.  Returns `true` if the contract holds.
pub trait ContractFn: Send + Sync {
    fn check(&self, memory: &AgentMemory) -> bool;
}

/// Blanket impl so users can pass plain closures.
impl<F> ContractFn for F
where
    F: Fn(&AgentMemory) -> bool + Send + Sync,
{
    fn check(&self, memory: &AgentMemory) -> bool {
        self(memory)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard failure actions
// ─────────────────────────────────────────────────────────────────────────────

/// What happens when a `TransitionGuard` rejects a transition.
#[derive(Debug, Clone)]
pub enum GuardFailAction {
    /// Block the transition; stay in the current state.
    /// The state will be re-handled on the next iteration.
    BlockTransition,
    /// Emit a custom event into the engine (allows custom recovery sub-graphs).
    EmitEvent(String),
    /// Hard abort — agent terminates with `ContractViolation`.
    FatalError,
}

/// What happens when an `Invariant` breaks.
#[derive(Debug, Clone)]
pub enum InvariantFailAction {
    LogWarning,
    FatalError,
    EmitEvent(String),
}

/// What happens when a `PostCondition` fails.
#[derive(Debug, Clone)]
pub enum PostConditionFailAction {
    /// Re-run the planning loop so the LLM can produce a better answer.
    RetryPlanning,
    FatalError,
}

// ─────────────────────────────────────────────────────────────────────────────
// Contract result
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned after evaluating a contract.
#[derive(Debug)]
pub enum ContractOutcome {
    /// Contract satisfied — proceed normally.
    Pass,
    /// Contract failed — take the associated action.
    Fail(ContractFailure),
}

/// Detailed contract failure information.
#[derive(Debug, Clone)]
pub struct ContractFailure {
    pub contract_name: String,
    pub message: String,
    pub action: ContractViolationAction,
}

#[derive(Debug, Clone)]
pub enum ContractViolationAction {
    Block,
    EmitEvent(String),
    FatalError,
    RetryPlanning,
}

// ─────────────────────────────────────────────────────────────────────────────
// TransitionGuard
// ─────────────────────────────────────────────────────────────────────────────

/// A pre-condition checked before a state transition is applied.
///
/// If `from` or `to` is `None`, it acts as a wildcard.
pub struct TransitionGuard {
    /// Human-readable name for logging.
    pub name: String,
    /// Source state filter. `None` = any source.
    pub from: Option<String>,
    /// Target state filter. `None` = any target.
    pub to: Option<String>,
    /// The predicate.
    pub condition: Arc<dyn ContractFn>,
    /// What to do on failure.
    pub on_fail: GuardFailAction,
    /// Counter to detect guards that block indefinitely.
    block_count: std::sync::atomic::AtomicUsize,
    /// Max consecutive blocks before escalating to FatalError.
    pub max_blocks: usize,
}

impl TransitionGuard {
    pub fn new(
        name: impl Into<String>,
        condition: impl ContractFn + 'static,
        on_fail: GuardFailAction,
    ) -> Self {
        Self {
            name: name.into(),
            from: None,
            to: None,
            condition: Arc::new(condition),
            on_fail,
            block_count: std::sync::atomic::AtomicUsize::new(0),
            max_blocks: 5,
        }
    }

    pub fn from_state(mut self, from: impl Into<String>) -> Self {
        self.from = Some(from.into());
        self
    }

    pub fn to_state(mut self, to: impl Into<String>) -> Self {
        self.to = Some(to.into());
        self
    }

    pub fn max_blocks(mut self, n: usize) -> Self {
        self.max_blocks = n;
        self
    }

    /// Returns true if this guard applies to the given transition.
    fn matches(&self, from: &str, to: &str) -> bool {
        self.from.as_deref().map(|f| f == from).unwrap_or(true)
            && self.to.as_deref().map(|t| t == to).unwrap_or(true)
    }

    /// Evaluate the guard. Returns `ContractOutcome`.
    pub fn evaluate(&self, from: &str, to: &str, memory: &AgentMemory) -> ContractOutcome {
        if !self.matches(from, to) {
            return ContractOutcome::Pass;
        }

        if self.condition.check(memory) {
            // Reset block counter on success
            self.block_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            ContractOutcome::Pass
        } else {
            let count = self
                .block_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;

            // If we've blocked too many times, escalate to fatal
            let action = if count >= self.max_blocks {
                tracing::error!(
                    guard = %self.name,
                    count,
                    "Guard blocked transition {} times — escalating to FatalError",
                    count
                );
                ContractViolationAction::FatalError
            } else {
                match &self.on_fail {
                    GuardFailAction::BlockTransition => ContractViolationAction::Block,
                    GuardFailAction::EmitEvent(e) => ContractViolationAction::EmitEvent(e.clone()),
                    GuardFailAction::FatalError => ContractViolationAction::FatalError,
                }
            };

            tracing::warn!(
                guard = %self.name,
                from,
                to,
                block_count = count,
                "Transition guard FAILED"
            );

            ContractOutcome::Fail(ContractFailure {
                contract_name: self.name.clone(),
                message: format!(
                    "Guard '{}' rejected transition {} → {}",
                    self.name, from, to
                ),
                action,
            })
        }
    }
}

/// Manual Clone — resets block_count to 0 on clone (guards start fresh on cloned builders).
impl Clone for TransitionGuard {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            from: self.from.clone(),
            to: self.to.clone(),
            condition: self.condition.clone(),
            on_fail: self.on_fail.clone(),
            block_count: std::sync::atomic::AtomicUsize::new(0),
            max_blocks: self.max_blocks,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Invariant
// ─────────────────────────────────────────────────────────────────────────────

/// A global constraint checked after every step.
pub struct Invariant {
    pub name: String,
    pub condition: Arc<dyn ContractFn>,
    pub on_fail: InvariantFailAction,
}

impl Invariant {
    pub fn new(
        name: impl Into<String>,
        condition: impl ContractFn + 'static,
        on_fail: InvariantFailAction,
    ) -> Self {
        Self {
            name: name.into(),
            condition: Arc::new(condition),
            on_fail,
        }
    }

    pub fn evaluate(&self, memory: &AgentMemory) -> ContractOutcome {
        if self.condition.check(memory) {
            ContractOutcome::Pass
        } else {
            let action = match &self.on_fail {
                InvariantFailAction::LogWarning => {
                    tracing::warn!(invariant = %self.name, "Invariant VIOLATED (warn-only)");
                    return ContractOutcome::Fail(ContractFailure {
                        contract_name: self.name.clone(),
                        message: format!("Invariant '{}' violated", self.name),
                        action: ContractViolationAction::Block, // treated as warning only
                    });
                }
                InvariantFailAction::FatalError => ContractViolationAction::FatalError,
                InvariantFailAction::EmitEvent(e) => ContractViolationAction::EmitEvent(e.clone()),
            };

            tracing::error!(invariant = %self.name, "Invariant VIOLATED");
            ContractOutcome::Fail(ContractFailure {
                contract_name: self.name.clone(),
                message: format!("Invariant '{}' violated", self.name),
                action,
            })
        }
    }
}

impl Clone for Invariant {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            condition: self.condition.clone(),
            on_fail: self.on_fail.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PostCondition
// ─────────────────────────────────────────────────────────────────────────────

/// A condition checked just before the final answer is emitted.
pub struct PostCondition {
    pub name: String,
    pub condition: Arc<dyn ContractFn>,
    pub on_fail: PostConditionFailAction,
}

impl PostCondition {
    pub fn new(
        name: impl Into<String>,
        condition: impl ContractFn + 'static,
        on_fail: PostConditionFailAction,
    ) -> Self {
        Self {
            name: name.into(),
            condition: Arc::new(condition),
            on_fail,
        }
    }

    pub fn evaluate(&self, memory: &AgentMemory) -> ContractOutcome {
        if self.condition.check(memory) {
            ContractOutcome::Pass
        } else {
            let action = match &self.on_fail {
                PostConditionFailAction::RetryPlanning => ContractViolationAction::RetryPlanning,
                PostConditionFailAction::FatalError => ContractViolationAction::FatalError,
            };
            tracing::warn!(postcondition = %self.name, "PostCondition FAILED");
            ContractOutcome::Fail(ContractFailure {
                contract_name: self.name.clone(),
                message: format!("PostCondition '{}' failed", self.name),
                action,
            })
        }
    }
}

impl Clone for PostCondition {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            condition: self.condition.clone(),
            on_fail: self.on_fail.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContractSet — the main container
// ─────────────────────────────────────────────────────────────────────────────

/// Holds all registered contracts for an agent.
#[derive(Default, Clone)]
pub struct ContractSet {
    guards: Vec<TransitionGuard>,
    invariants: Vec<Invariant>,
    postconditions: Vec<PostCondition>,
}

impl ContractSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_guard(mut self, guard: TransitionGuard) -> Self {
        self.guards.push(guard);
        self
    }

    pub fn add_invariant(mut self, inv: Invariant) -> Self {
        self.invariants.push(inv);
        self
    }

    pub fn add_postcondition(mut self, pc: PostCondition) -> Self {
        self.postconditions.push(pc);
        self
    }

    /// Evaluate all transition guards for a given (from, to) pair.
    /// Returns the first failure found, or `None` if all pass.
    pub fn check_guards(
        &self,
        from: &State,
        to: &State,
        memory: &AgentMemory,
    ) -> Option<ContractFailure> {
        for guard in &self.guards {
            if let ContractOutcome::Fail(failure) =
                guard.evaluate(from.as_str(), to.as_str(), memory)
            {
                return Some(failure);
            }
        }
        None
    }

    /// Evaluate all invariants. Returns the first failure found.
    pub fn check_invariants(&self, memory: &AgentMemory) -> Option<ContractFailure> {
        for inv in &self.invariants {
            if let ContractOutcome::Fail(failure) = inv.evaluate(memory) {
                // LogWarning outcomes are not fatal — skip them
                if matches!(failure.action, ContractViolationAction::Block) {
                    continue;
                }
                return Some(failure);
            }
        }
        None
    }

    /// Evaluate all postconditions. Returns the first failure found.
    pub fn check_postconditions(&self, memory: &AgentMemory) -> Option<ContractFailure> {
        for pc in &self.postconditions {
            if let ContractOutcome::Fail(failure) = pc.evaluate(memory) {
                return Some(failure);
            }
        }
        None
    }

    pub fn has_contracts(&self) -> bool {
        !self.guards.is_empty() || !self.invariants.is_empty() || !self.postconditions.is_empty()
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
    fn test_guard_passes_when_condition_true() {
        let guard = TransitionGuard::new(
            "always_pass",
            |_: &AgentMemory| true,
            GuardFailAction::BlockTransition,
        );
        let m = make_memory();
        assert!(matches!(
            guard.evaluate("Planning", "Acting", &m),
            ContractOutcome::Pass
        ));
    }

    #[test]
    fn test_guard_fails_with_block() {
        let guard = TransitionGuard::new(
            "always_fail",
            |_: &AgentMemory| false,
            GuardFailAction::BlockTransition,
        );
        let m = make_memory();
        let outcome = guard.evaluate("Planning", "Acting", &m);
        assert!(matches!(
            outcome,
            ContractOutcome::Fail(ContractFailure {
                action: ContractViolationAction::Block,
                ..
            })
        ));
    }

    #[test]
    fn test_guard_escalates_after_max_blocks() {
        let guard = TransitionGuard::new(
            "block_escalate",
            |_: &AgentMemory| false,
            GuardFailAction::BlockTransition,
        )
        .max_blocks(3);
        let m = make_memory();
        // First 2 → Block
        for _ in 0..2 {
            assert!(matches!(
                guard.evaluate("Planning", "Acting", &m),
                ContractOutcome::Fail(ContractFailure {
                    action: ContractViolationAction::Block,
                    ..
                })
            ));
        }
        // 3rd → FatalError (escalation)
        assert!(matches!(
            guard.evaluate("Planning", "Acting", &m),
            ContractOutcome::Fail(ContractFailure {
                action: ContractViolationAction::FatalError,
                ..
            })
        ));
    }

    #[test]
    fn test_guard_wildcard_from() {
        let guard = TransitionGuard::new(
            "any_to_done",
            |_: &AgentMemory| false,
            GuardFailAction::FatalError,
        )
        .to_state("Done");
        let m = make_memory();
        // Matches any → Done
        assert!(matches!(
            guard.evaluate("Foo", "Done", &m),
            ContractOutcome::Fail(_)
        ));
        // Does not match → other
        assert!(matches!(
            guard.evaluate("Foo", "Bar", &m),
            ContractOutcome::Pass
        ));
    }

    #[test]
    fn test_invariant_passes() {
        let inv = Invariant::new(
            "step_ok",
            |m: &AgentMemory| m.step < 100,
            InvariantFailAction::FatalError,
        );
        let m = make_memory();
        assert!(matches!(inv.evaluate(&m), ContractOutcome::Pass));
    }

    #[test]
    fn test_invariant_fails_fatal() {
        let inv = Invariant::new(
            "step_zero",
            |m: &AgentMemory| m.step > 999,
            InvariantFailAction::FatalError,
        );
        let m = make_memory();
        assert!(matches!(
            inv.evaluate(&m),
            ContractOutcome::Fail(ContractFailure {
                action: ContractViolationAction::FatalError,
                ..
            })
        ));
    }

    #[test]
    fn test_postcondition_fails_retry() {
        let pc = PostCondition::new(
            "answer_long_enough",
            |m: &AgentMemory| {
                m.final_answer
                    .as_ref()
                    .map(|a| a.len() >= 20)
                    .unwrap_or(false)
            },
            PostConditionFailAction::RetryPlanning,
        );
        let m = make_memory(); // no final_answer
        assert!(matches!(
            pc.evaluate(&m),
            ContractOutcome::Fail(ContractFailure {
                action: ContractViolationAction::RetryPlanning,
                ..
            })
        ));
    }

    #[test]
    fn test_contract_set_check_guards_first_failure() {
        let cs = ContractSet::new()
            .add_guard(TransitionGuard::new(
                "pass",
                |_: &AgentMemory| true,
                GuardFailAction::BlockTransition,
            ))
            .add_guard(TransitionGuard::new(
                "fail",
                |_: &AgentMemory| false,
                GuardFailAction::FatalError,
            ));
        let m = make_memory();
        let failure = cs.check_guards(&State::planning(), &State::acting(), &m);
        assert!(failure.is_some());
        assert_eq!(failure.unwrap().contract_name, "fail");
    }

    #[test]
    fn test_contract_set_all_pass() {
        let cs = ContractSet::new()
            .add_guard(TransitionGuard::new(
                "p1",
                |_: &AgentMemory| true,
                GuardFailAction::BlockTransition,
            ))
            .add_invariant(Invariant::new(
                "p2",
                |_: &AgentMemory| true,
                InvariantFailAction::FatalError,
            ));
        let m = make_memory();
        assert!(cs
            .check_guards(&State::planning(), &State::acting(), &m)
            .is_none());
        assert!(cs.check_invariants(&m).is_none());
    }
}
