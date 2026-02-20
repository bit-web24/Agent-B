use std::collections::HashMap;
use crate::states::AgentState;
use crate::types::State;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::LlmCaller;
use crate::transitions::TransitionTable;
use crate::trace::Trace;
use crate::error::AgentError;

pub struct AgentEngine {
    pub memory:     AgentMemory,
    pub tools:      ToolRegistry,
    pub llm:        Box<dyn LlmCaller>,
    state:          State,
    transitions:    TransitionTable,
    handlers:       HashMap<&'static str, Box<dyn AgentState>>,
}

impl AgentEngine {
    /// Creates a new engine. Prefer using AgentBuilder for ergonomic construction.
    pub fn new(
        memory:      AgentMemory,
        tools:       ToolRegistry,
        llm:         Box<dyn LlmCaller>,
        transitions: TransitionTable,
        handlers:    HashMap<&'static str, Box<dyn AgentState>>,
    ) -> Self {
        Self {
            memory,
            tools,
            llm,
            state: State::Idle,
            transitions,
            handlers,
        }
    }

    /// Run the agent to completion.
    /// Returns Ok(final_answer) or Err(AgentError).
    pub fn run(&mut self) -> Result<String, AgentError> {
        let safety_cap = self.memory.config.max_steps * 3;
        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > safety_cap {
                return Err(AgentError::SafetyCapExceeded(iterations));
            }

            tracing::info!(state = %self.state, iteration = iterations, "agent loop tick");

            // Exit condition: terminal state
            if self.state.is_terminal() {
                return match self.state {
                    State::Done  => Ok(self.memory.final_answer.clone()
                        .unwrap_or_else(|| "[No answer produced]".to_string())),
                    State::Error => Err(AgentError::AgentFailed(
                        self.memory.error.clone()
                            .unwrap_or_else(|| "Unknown error".to_string())
                    )),
                    _ => unreachable!(),
                };
            }

            // Get handler for current state
            let state_name = self.state.as_str();
            let handler = self.handlers.get(state_name)
                .ok_or_else(|| AgentError::NoHandlerForState(state_name.to_string()))?;

            // Execute state — get event
            let event = handler.handle(&mut self.memory, &self.tools, self.llm.as_ref());

            tracing::debug!(state = %self.state, event = %event, "state produced event");

            // Look up transition
            let key = (self.state.clone(), event.clone());
            let next_state = self.transitions.get(&key)
                .cloned()
                .ok_or_else(|| AgentError::InvalidTransition {
                    from:  self.state.clone(),
                    event: event.clone(),
                })?;

            tracing::info!(from = %self.state, event = %event, to = %next_state, "transition");
            println!("  ══ {} --{}--> {} ══", self.state, event, next_state);

            self.state = next_state;
        }
    }

    /// Returns a reference to the full execution trace.
    pub fn trace(&self) -> &Trace {
        &self.memory.trace
    }

    /// Returns the current state (useful for inspection after run).
    pub fn current_state(&self) -> &State {
        &self.state
    }
}
