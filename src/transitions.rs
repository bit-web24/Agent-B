use std::collections::HashMap;
use crate::types::State;
use crate::events::Event;

pub type TransitionTable = HashMap<(State, Event), State>;

/// Builds the complete, immutable transition table.
/// This function defines ALL legal behaviors of the agent.
/// Any (State, Event) pair not in this table is illegal and
/// will cause AgentEngine::run() to return AgentError::InvalidTransition.
pub fn build_transition_table() -> TransitionTable {
    let mut t = HashMap::new();

    // ── IDLE ─────────────────────────────────────────────
    t.insert((State::Idle,       Event::Start),           State::Planning);

    // ── PLANNING ─────────────────────────────────────────
    t.insert((State::Planning,   Event::LlmToolCall),     State::Acting);
    t.insert((State::Planning,   Event::LlmFinalAnswer),  State::Done);
    t.insert((State::Planning,   Event::MaxSteps),        State::Error);
    t.insert((State::Planning,   Event::LowConfidence),   State::Reflecting);
    t.insert((State::Planning,   Event::AnswerTooShort),  State::Planning);
    t.insert((State::Planning,   Event::ToolBlacklisted), State::Planning);
    t.insert((State::Planning,   Event::FatalError),      State::Error);

    // ── ACTING ───────────────────────────────────────────
    t.insert((State::Acting,     Event::ToolSuccess),     State::Observing);
    t.insert((State::Acting,     Event::ToolFailure),     State::Observing);
    t.insert((State::Acting,     Event::FatalError),      State::Error);

    // ── OBSERVING ────────────────────────────────────────
    t.insert((State::Observing,  Event::Continue),        State::Planning);
    t.insert((State::Observing,  Event::NeedsReflection), State::Reflecting);

    // ── REFLECTING ───────────────────────────────────────
    t.insert((State::Reflecting, Event::ReflectDone),     State::Planning);

    // Note: DONE and ERROR are terminal — no outgoing transitions.
    // Engine checks State::is_terminal() and exits before table lookup.

    t
}

/// Validates that a given (state, event) pair is legal.
pub fn is_valid_transition(table: &TransitionTable, state: &State, event: &Event) -> bool {
    table.contains_key(&(state.clone(), event.clone()))
}
