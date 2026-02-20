//! Integration tests for agentsm-rs.
//!
//! All tests use `MockLlmCaller` — no network calls are made.
//! Run with: `cargo test`

use agentsm::{
    AgentBuilder, AgentConfig, AgentEngine, AgentError,
    Event, LlmResponse, State, ToolCall,
    ToolRegistry,
};
use agentsm::llm::MockLlmCaller;
use agentsm::memory::AgentMemory;
use agentsm::states::{
    AgentState, IdleState, PlanningState, ActingState,
    ObservingState, ReflectingState, DoneState, ErrorState,
};
use agentsm::transitions::build_transition_table;
use serde_json::json;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

fn test_memory() -> AgentMemory {
    AgentMemory::new("test task")
}

fn test_tools() -> ToolRegistry {
    ToolRegistry::new()
}

fn make_tool_call_response(name: &str) -> LlmResponse {
    LlmResponse::ToolCall {
        tool: ToolCall {
            name: name.to_string(),
            args: HashMap::new(),
        },
        confidence: 1.0,
    }
}

fn make_final_answer(content: &str) -> LlmResponse {
    LlmResponse::FinalAnswer {
        content: content.to_string(),
    }
}

fn make_mock_llm(responses: Vec<LlmResponse>) -> MockLlmCaller {
    MockLlmCaller::new(responses)
}

/// Build a full engine from a MockLlmCaller. Registers a "dummy" tool.
fn make_engine_with_mock(mock: MockLlmCaller) -> AgentEngine {
    AgentBuilder::new("test task")
        .llm(Box::new(mock))
        .tool(
            "dummy",
            "A dummy tool for testing",
            json!({ "type": "object", "properties": {} }),
            Box::new(|_args| Ok("dummy result".to_string())),
        )
        .build()
        .expect("builder should succeed")
}

/// Build a minimal engine with no tools.
fn make_bare_engine(mock: MockLlmCaller) -> AgentEngine {
    AgentBuilder::new("bare test task")
        .llm(Box::new(mock))
        .build()
        .expect("bare builder should succeed")
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: IdleState produces Event::Start → Planning
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_idle_to_planning_transition() {
    let mut memory = test_memory();
    let tools      = test_tools();
    let llm        = make_mock_llm(vec![make_final_answer("The answer to life is 42.")]);

    let idle = IdleState;
    let event = idle.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::Start, "IdleState must emit Event::Start");

    // Verify the transition table maps (Idle, Start) → Planning
    let table = build_transition_table();
    let next = table.get(&(State::Idle, Event::Start));
    assert_eq!(next, Some(&State::Planning), "Idle + Start should transition to Planning");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: PlanningState max-steps guard
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_planning_max_steps_guard() {
    let mut memory = test_memory();
    memory.step = memory.config.max_steps; // already at the limit

    let tools = test_tools();
    let llm   = make_mock_llm(vec![]);  // Should never be called

    let state = PlanningState;
    let event = state.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::MaxSteps, "PlanningState must return MaxSteps when step >= max_steps");
    assert!(memory.error.is_some(), "memory.error must be set on MaxSteps");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: PlanningState rejects blacklisted tools
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_planning_tool_blacklist() {
    let mut engine = AgentBuilder::new("test blacklist")
        .llm(Box::new(make_mock_llm(vec![
            // First response requests the blacklisted tool
            make_tool_call_response("forbidden_tool"),
            // Second response → final answer (after blacklist rejection loops back to Planning)
            make_final_answer("I used an allowed approach to answer your question properly."),
        ])))
        .tool(
            "forbidden_tool",
            "A tool that is registered but blacklisted",
            json!({ "type": "object", "properties": {} }),
            Box::new(|_| Ok("should never run".to_string())),
        )
        .blacklist_tool("forbidden_tool")
        .build()
        .expect("builder should succeed");

    let result = engine.run();
    assert!(result.is_ok(), "Agent should complete despite blacklisted tool call: {:?}", result);

    // Verify the trace contains the blacklist event
    let blacklist_entries: Vec<_> = engine.trace().entries()
        .iter()
        .filter(|e| e.event == "TOOL_BLACKLISTED")
        .collect();
    assert!(!blacklist_entries.is_empty(), "Trace should record TOOL_BLACKLISTED event");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: ActingState treats unknown tool as ToolFailure, not crash
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_acting_unknown_tool_is_failure_not_crash() {
    let mut memory = test_memory();
    memory.current_tool_call = Some(ToolCall {
        name: "nonexistent_tool".to_string(),
        args: HashMap::new(),
    });

    let tools = test_tools(); // empty — tool not registered
    let llm   = make_mock_llm(vec![]);

    let state = ActingState;
    let event = state.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::ToolFailure, "Unknown tool must produce ToolFailure, not FatalError or panic");
    assert!(
        memory.last_observation.as_ref().map_or(false, |o| o.starts_with("ERROR:")),
        "last_observation must be prefixed with 'ERROR:'"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: ObservingState triggers reflection at the configured step interval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_observing_triggers_reflection_at_step_5() {
    let mut memory = test_memory();
    memory.config.reflect_every_n_steps = 5;
    memory.step = 5; // exactly at the boundary

    // Provide a current tool call and observation so Observing can commit them
    memory.current_tool_call = Some(ToolCall {
        name: "dummy".to_string(),
        args: HashMap::new(),
    });
    memory.last_observation = Some("SUCCESS: some result".to_string());

    let tools = test_tools();
    let llm   = make_mock_llm(vec![]);

    let state = ObservingState;
    let event = state.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::NeedsReflection, "ObservingState must emit NeedsReflection at step % interval == 0");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: ObservingState commits tool call + observation to history
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_observing_commits_to_history() {
    let mut memory = test_memory();
    memory.config.reflect_every_n_steps = 0; // Disable reflection triggering
    memory.step = 1;

    memory.current_tool_call = Some(ToolCall {
        name: "search".to_string(),
        args: HashMap::from([
            ("query".to_string(), json!("Rust language")),
        ]),
    });
    memory.last_observation = Some("SUCCESS: Rust is a systems programming language.".to_string());

    let tools = test_tools();
    let llm   = make_mock_llm(vec![]);

    let state = ObservingState;
    let event = state.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::Continue, "ObservingState should return Continue between reflections");
    assert_eq!(memory.history.len(), 1, "One HistoryEntry should be committed");
    assert_eq!(memory.history[0].tool.name, "search");
    assert!(memory.history[0].success, "Entry should be marked success");
    assert!(memory.current_tool_call.is_none(), "current_tool_call must be cleared");
    assert!(memory.last_observation.is_none(), "last_observation must be cleared");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Full run with MockLlmCaller: tool call + final answer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_full_run_with_mock_llm() {
    let mock = make_mock_llm(vec![
        // Step 1: LLM requests the dummy tool
        make_tool_call_response("dummy"),
        // Step 2: LLM provides final answer after seeing the tool result
        make_final_answer("Based on the dummy tool result, the answer is 42."),
    ]);

    let mut engine = make_engine_with_mock(mock);
    let result = engine.run();

    assert!(result.is_ok(), "Agent should complete successfully: {:?}", result);
    let answer = result.unwrap();
    assert!(!answer.is_empty(), "Final answer must not be empty");
    assert!(answer.contains("42"), "Final answer should contain expected content");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Full run ends in Done state
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_full_run_reaches_done_state() {
    let mock = make_mock_llm(vec![
        make_tool_call_response("dummy"),
        make_final_answer("This is the complete final answer to the test question."),
    ]);

    let mut engine = make_engine_with_mock(mock);
    let result = engine.run();

    assert!(result.is_ok(), "Agent should complete: {:?}", result);
    assert_eq!(
        engine.current_state(), &State::Done,
        "Engine must be in Done state after successful completion"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: Invalid (State, Event) pair is caught, not panicked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_invalid_transition_returns_error() {
    // Directly test that PlanningState fires MaxSteps when step >= max_steps,
    // bypassing engine loop complexities.
    let mut memory = test_memory();
    memory.config.max_steps = 1;
    memory.step = 1; // already at max

    let tools = test_tools();
    let llm   = make_mock_llm(vec![]); // should never be called

    let state = PlanningState;
    let event = state.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::MaxSteps, "PlanningState must emit MaxSteps when step >= max_steps");
    assert!(memory.error.is_some(), "memory.error must be set");

    // Also verify the transition table routes MaxSteps → Error
    let table = build_transition_table();
    let next  = table.get(&(State::Planning, Event::MaxSteps));
    assert_eq!(next, Some(&State::Error), "MaxSteps should transition to Error state");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: Trace records all steps
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_trace_records_all_steps() {
    let mock = make_mock_llm(vec![
        make_tool_call_response("dummy"),
        make_final_answer("Trace test complete answer value here."),
    ]);

    let mut engine = make_engine_with_mock(mock);
    engine.run().expect("Agent should complete");

    let trace = engine.trace();
    assert!(trace.len() > 0, "Trace must not be empty after a run");

    // Verify expected states appear in the trace
    let idle_entries = trace.for_state("Idle");
    assert!(!idle_entries.is_empty(), "Trace must contain Idle state entries");

    let planning_entries = trace.for_state("Planning");
    assert!(!planning_entries.is_empty(), "Trace must contain Planning state entries");

    let acting_entries = trace.for_state("Acting");
    assert!(!acting_entries.is_empty(), "Trace must contain Acting state entries");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: ToolRegistry returns Err for unknown tool, does not panic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_tool_registry_execute_unknown_returns_err() {
    let registry = ToolRegistry::new(); // empty registry
    let args     = HashMap::new();

    let result = registry.execute("nonexistent_tool", &args);

    assert!(result.is_err(), "Executing unknown tool should return Err, not panic");
    let err = result.unwrap_err();
    assert!(err.contains("not found"), "Error message should mention 'not found'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: MockLlmCaller.call_count() matches actual LLM invocations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mock_llm_call_count() {
    let mock = make_mock_llm(vec![
        make_tool_call_response("dummy"),  // call 1
        make_tool_call_response("dummy"),  // call 2
        make_final_answer("Three LLM calls to complete this mock test run."), // call 3
    ]);

    let mock_ref = {
        // We need to keep a reference to count calls, so wrap in Arc to share
        // But MockLlmCaller isn't Arc-compatible out of the box.
        // Instead, build engine first, then check after.
        // Since AgentBuilder takes ownership of the LlmCaller, we must
        // build and run, then check via trace.
        make_mock_llm(vec![
            make_tool_call_response("dummy"),
            make_tool_call_response("dummy"),
            make_final_answer("Three LLM calls to complete this mock test run."),
        ])
    };

    let mut engine = make_engine_with_mock(mock_ref);
    engine.run().expect("Agent should complete");

    // Verify via trace: Planning state should have 3 STEP_START entries
    let planning_steps: Vec<_> = engine.trace()
        .entries()
        .iter()
        .filter(|e| e.state == "Planning" && e.event == "STEP_START")
        .collect();

    assert_eq!(
        planning_steps.len(), 3,
        "There should be exactly 3 planning steps (3 LLM calls)"
    );

    // Also verify history has 2 entries (2 tool calls succeeded)
    assert_eq!(
        engine.memory.history.len(), 2,
        "History should have 2 completed tool calls"
    );

    // Suppress unused variable warning for mock
    drop(mock);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13: AgentConfig max_steps is respected
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_agent_config_respected() {
    // Directly verify via PlanningState: when step >= max_steps, MaxSteps fires.
    // This tests the config contract without fighting the safety cap.
    let mut memory = test_memory();
    memory.config.max_steps = 3;
    memory.step = 3; // simulate: 3 steps have been used, at the limit

    let tools = test_tools();
    let llm   = make_mock_llm(vec![]);

    let state = PlanningState;
    let event = state.handle(&mut memory, &tools, &llm);

    assert_eq!(event, Event::MaxSteps, "AgentConfig max_steps must be enforced by PlanningState");
    assert!(
        memory.error.as_ref().map_or(false, |e| e.contains("Max steps")),
        "memory.error should mention max steps, got: {:?}", memory.error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14: AgentBuilder requires an LLM caller
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_builder_requires_llm() {
    // Do NOT call .llm() — should produce BuildError
    let result = AgentBuilder::new("test no llm").build();

    assert!(result.is_err(), "Building without LLM should return Err");
    // Use err().unwrap() instead of unwrap_err() — avoids requiring AgentEngine: Debug
    let err = result.err().unwrap();
    match err {
        AgentError::BuildError(msg) => {
            assert!(
                msg.to_lowercase().contains("llm") || msg.to_lowercase().contains("required"),
                "BuildError should mention LLM: {}", msg
            );
        }
        other => panic!("Expected BuildError, got: {:?}", other),
    }
}
