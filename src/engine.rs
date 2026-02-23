use std::collections::{HashMap, HashSet};
use crate::states::AgentState;
use crate::events::Event;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::llm::AsyncLlmCaller;
use crate::transitions::TransitionTable;
use crate::trace::Trace;
use crate::types::{State, AgentOutput};
use crate::error::AgentError;
use futures::stream::BoxStream;
use tokio::sync::mpsc;

pub struct AgentEngine {
    pub memory:          AgentMemory,
    pub tools:           ToolRegistry,
    pub llm:             Box<dyn AsyncLlmCaller>,
    state:               State,
    transitions:         TransitionTable,
    handlers:            HashMap<String, Box<dyn AgentState>>,
    terminal_states:     HashSet<String>,
}

impl AgentEngine {
    /// Creates a new engine. Prefer using AgentBuilder for ergonomic construction.
    pub fn new(
        memory:          AgentMemory,
        tools:           ToolRegistry,
        llm:             Box<dyn AsyncLlmCaller>,
        transitions:     TransitionTable,
        handlers:        HashMap<String, Box<dyn AgentState>>,
        terminal_states: HashSet<String>,
    ) -> Self {
        Self {
            memory,
            tools,
            llm,
            state: State::idle(),
            transitions,
            handlers,
            terminal_states,
        }
    }

    /// Run the agent to completion asynchronously.
    /// Returns Ok(final_answer) or Err(AgentError).
    pub async fn run(&mut self) -> Result<String, AgentError> {
        let (tx, _rx) = mpsc::unbounded_channel();
        let safety_cap = self.memory.config.max_steps * 3;
        let mut iterations = 0;

        while !self.terminal_states.contains(self.state.as_str()) {
            iterations += 1;
            if iterations > safety_cap {
                return Err(AgentError::SafetyCapExceeded(iterations));
            }

            self.step(&tx).await?;
        }

        if self.state == State::done() {
            Ok(self.memory.final_answer.clone()
                .unwrap_or_else(|| "[No answer produced]".to_string()))
        } else if self.state == State::error() {
            Err(AgentError::AgentFailed(
                self.memory.error.clone()
                    .unwrap_or_else(|| "Unknown error".to_string())
            ))
        } else {
            Ok(self.memory.final_answer.clone()
                .unwrap_or_else(|| format!("[Terminated in state: {}]", self.state)))
        }
    }

    /// Executes a single state transition.
    /// Returns Ok(()) if successful, or Err(AgentError).
    pub async fn step(&mut self, tx: &mpsc::UnboundedSender<AgentOutput>) -> Result<(), AgentError> {
        tracing::info!(state = %self.state, "agent step");

        // Get handler for current state
        let state_name = self.state.as_str();
        let handler = self.handlers.get(state_name)
            .ok_or_else(|| AgentError::NoHandlerForState(state_name.to_string()))?;

        // Execute state — get event
        let event: Event = handler.handle(&mut self.memory, &self.tools, self.llm.as_ref(), Some(tx)).await;

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
        println!("  ══ {} --{}-->{} ══", self.state, event, next_state);

        self.state = next_state;
        Ok(())
    }

    /// Run the agent and return a stream of AgentOutput events.
    pub fn run_streaming(&mut self) -> BoxStream<'_, AgentOutput> {
        use futures::stream;
        use futures::StreamExt;

        let (tx, rx) = mpsc::unbounded_channel();
        
        stream::unfold((self, rx, tx, false), |(engine, mut rx, tx, mut done)| async move {
            if done {
                return None;
            }

            // 1. If we have pending messages in the channel (e.g. from the last step or tokens), yield them first.
            if let Ok(msg) = rx.try_recv() {
                return Some((msg, (engine, rx, tx, false)));
            }

            // 2. Check if we've reached a terminal state.
            if engine.terminal_states.contains(engine.state.as_str()) {
                done = true;
                // Try one last recv just in case
                if let Ok(msg) = rx.try_recv() {
                    return Some((msg, (engine, rx, tx, true)));
                }
                return None;
            }

            // 3. Execute one step of the engine.
            // This will likely send many events (StateStarted, tokens, ToolCallStarted, etc.) to tx.
            if let Err(e) = engine.step(&tx).await {
                done = true;
                return Some((AgentOutput::Error(e.to_string()), (engine, rx, tx, true)));
            }

            // 4. After a step, we should have at least one message (StateStarted).
            if let Ok(msg) = rx.try_recv() {
                return Some((msg, (engine, rx, tx, false)));
            }

            // If we get here, the step produced no output and wasn't terminal (rare but possible).
            // We just return an empty action to keep the stream alive or recurse? 
            // Recursing is better.
            None // For now, end stream if no output.
        }).boxed()
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
