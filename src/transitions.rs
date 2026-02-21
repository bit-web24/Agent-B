use std::collections::HashMap;
use crate::types::State;
use crate::events::Event;

pub type TransitionTable = HashMap<(State, Event), State>;

/// Builds the complete default transition table.
/// This defines the standard ReAct-cycle behavior of the agent.
/// Any (State, Event) pair not in this table is illegal and
/// will cause AgentEngine::run() to return AgentError::InvalidTransition.
///
/// Users can extend this table with custom transitions via
/// `AgentBuilder::transition()`.
pub fn build_transition_table() -> TransitionTable {
    let mut t = HashMap::new();

    // ── IDLE ─────────────────────────────────────────────
    t.insert((State::idle(),       Event::start()),           State::planning());

    // ── PLANNING ─────────────────────────────────────────
    t.insert((State::planning(),   Event::llm_tool_call()),     State::acting());
    t.insert((State::planning(),   Event::llm_final_answer()),  State::done());
    t.insert((State::planning(),   Event::max_steps()),        State::error());
    t.insert((State::planning(),   Event::low_confidence()),   State::reflecting());
    t.insert((State::planning(),   Event::answer_too_short()),  State::planning());
    t.insert((State::planning(),   Event::tool_blacklisted()), State::planning());
    t.insert((State::planning(),   Event::fatal_error()),      State::error());

    // ── ACTING ───────────────────────────────────────────
    t.insert((State::acting(),     Event::tool_success()),     State::observing());
    t.insert((State::acting(),     Event::tool_failure()),     State::observing());
    t.insert((State::acting(),     Event::fatal_error()),      State::error());

    // ── OBSERVING ────────────────────────────────────────
    t.insert((State::observing(),  Event::r#continue()),        State::planning());
    t.insert((State::observing(),  Event::needs_reflection()), State::reflecting());

    // ── REFLECTING ───────────────────────────────────────
    t.insert((State::reflecting(), Event::reflect_done()),     State::planning());

    // Note: DONE and ERROR are terminal — no outgoing transitions.
    // Engine checks State::is_terminal() and exits before table lookup.

    t
}

/// Validates that a given (state, event) pair is legal.
pub fn is_valid_transition(table: &TransitionTable, state: &State, event: &Event) -> bool {
    table.contains_key(&(state.clone(), event.clone()))
}
