use crate::checkpoint::{AgentCheckpoint, CheckpointStore};
use crate::contracts::{ContractSet, ContractViolationAction};
use crate::error::AgentError;
use crate::events::Event;
use crate::hooks::{safe_hook, AgentHooks};
use crate::llm::AsyncLlmCaller;
use crate::memory::AgentMemory;
use crate::states::AgentState;
use crate::tools::ToolRegistry;
use crate::trace::Trace;
use crate::transitions::TransitionTable;
use crate::types::{AgentOutput, State};
use futures::stream::BoxStream;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct AgentEngine {
    pub memory: AgentMemory,
    pub tools: Arc<ToolRegistry>,
    pub llm: Arc<dyn AsyncLlmCaller>,
    pub state: State,
    transitions: TransitionTable,
    handlers: HashMap<String, Arc<dyn AgentState>>,
    terminal_states: HashSet<String>,
    pub session_id: String,
    pub checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    pub hooks: Arc<dyn AgentHooks>,
    pub contracts: ContractSet,
    pub introspection: Option<crate::introspection::IntrospectionEngine>,
    pub healing_policy: Option<crate::healing::HealingPolicy>,
}

impl AgentEngine {
    /// Creates a new engine. Prefer using AgentBuilder for ergonomic construction.
    pub fn new(
        memory: AgentMemory,
        tools: Arc<ToolRegistry>,
        llm: Arc<dyn AsyncLlmCaller>,
        transitions: TransitionTable,
        handlers: HashMap<String, Arc<dyn AgentState>>,
        terminal_states: HashSet<String>,
        session_id: String,
        checkpoint_store: Option<Arc<dyn CheckpointStore>>,
        hooks: Arc<dyn AgentHooks>,
        contracts: ContractSet,
        introspection: Option<crate::introspection::IntrospectionEngine>,
        healing_policy: Option<crate::healing::HealingPolicy>,
    ) -> Self {
        Self {
            memory,
            tools,
            llm,
            state: State::idle(),
            transitions,
            handlers,
            terminal_states,
            session_id,
            checkpoint_store,
            hooks,
            contracts,
            introspection,
            healing_policy,
        }
    }

    /// Run the agent to completion asynchronously.
    /// Returns Ok(final_answer) or Err(AgentError).
    pub async fn run(&mut self) -> Result<String, AgentError> {
        // Inject hooks into memory so state handlers can access them
        self.memory.hooks = self.hooks.clone();

        let (tx, _rx) = mpsc::unbounded_channel();
        let safety_cap = self.memory.config.max_steps * 3;
        let mut iterations = 0;
        let mut postcondition_retries = 0;
        let max_postcondition_retries = 3;

        // Hook: agent start
        let hooks = self.hooks.clone();
        let task = self.memory.task.clone();
        safe_hook(|| hooks.on_agent_start(&task, &self.memory));

        'outer: loop {
            while !self.terminal_states.contains(self.state.as_str()) {
                iterations += 1;
                if iterations > safety_cap {
                    let err = AgentError::SafetyCapExceeded(iterations);
                    let hooks = self.hooks.clone();
                    safe_hook(|| hooks.on_agent_end(Err(&err), &self.memory));
                    return Err(err);
                }

                self.step(&tx).await?;

                // Contract: check invariants after every step
                if let Some(failure) = self.contracts.check_invariants(&self.memory) {
                    match failure.action {
                        ContractViolationAction::FatalError => {
                            return Err(AgentError::ContractViolation {
                                name: failure.contract_name,
                                message: failure.message,
                            });
                        }
                        ContractViolationAction::EmitEvent(ref evt) => {
                            tracing::warn!(invariant = %failure.contract_name, event = %evt, "Invariant triggered event");
                            let key = (self.state.clone(), Event::new(evt));
                            if let Some(next) = self.transitions.get(&key).cloned() {
                                self.state = next;
                            }
                        }
                        _ => {} // Block/LogWarning — already logged, continue
                    }
                }
            }

            // Contract: check postconditions before emitting final answer
            if self.state == State::done() {
                if let Some(failure) = self.contracts.check_postconditions(&self.memory) {
                    match failure.action {
                        ContractViolationAction::RetryPlanning => {
                            postcondition_retries += 1;
                            if postcondition_retries > max_postcondition_retries {
                                return Err(AgentError::ContractViolation {
                                    name: failure.contract_name,
                                    message: format!(
                                        "{} (exceeded {} retries)",
                                        failure.message, max_postcondition_retries
                                    ),
                                });
                            }
                            tracing::warn!(postcondition = %failure.contract_name, retry = postcondition_retries, "PostCondition failed — retrying planning");
                            self.state = State::planning();
                            self.memory.final_answer = None;
                            continue 'outer;
                        }
                        ContractViolationAction::FatalError => {
                            return Err(AgentError::ContractViolation {
                                name: failure.contract_name,
                                message: failure.message,
                            });
                        }
                        _ => {}
                    }
                }
            }

            break; // All postconditions pass (or no postconditions) — exit loop
        }

        let result = if self.state == State::done() {
            Ok(self
                .memory
                .final_answer
                .clone()
                .unwrap_or_else(|| "[No answer produced]".to_string()))
        } else if self.state == State::error() {
            Err(AgentError::AgentFailed(
                self.memory
                    .error
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        } else {
            Ok(self
                .memory
                .final_answer
                .clone()
                .unwrap_or_else(|| format!("[Terminated in state: {}]", self.state)))
        };

        // Hook: agent end
        let hooks = self.hooks.clone();
        match &result {
            Ok(answer) => safe_hook(|| hooks.on_agent_end(Ok(answer.as_str()), &self.memory)),
            Err(e) => safe_hook(|| hooks.on_agent_end(Err(e), &self.memory)),
        }

        result
    }

    /// Executes a single state transition.
    /// Returns Ok(()) if successful, or Err(AgentError).
    pub async fn step(
        &mut self,
        tx: &mpsc::UnboundedSender<AgentOutput>,
    ) -> Result<(), AgentError> {
        tracing::info!(state = %self.state, "agent step");

        // Get handler for current state
        let state_name = self.state.as_str();
        let handler = self
            .handlers
            .get(state_name)
            .ok_or_else(|| AgentError::NoHandlerForState(state_name.to_string()))?;

        // Hook: state enter
        let hooks = self.hooks.clone();
        let sn = state_name.to_string();
        safe_hook(|| hooks.on_state_enter(&sn, &self.memory));

        // Execute state — get event
        let event: Event = handler
            .handle(&mut self.memory, &self.tools, self.llm.as_ref(), Some(tx))
            .await;

        // Hook: state exit
        let hooks = self.hooks.clone();
        let sn = state_name.to_string();
        safe_hook(|| hooks.on_state_exit(&sn, &event, &self.memory));

        tracing::debug!(state = %self.state, event = %event, "state produced event");

        // Introspection: analyze recent execution history
        let anomalies = if let Some(intro) = &mut self.introspection {
            intro.analyze(&self.memory)
        } else {
            Vec::new()
        };
        for anomaly in anomalies {
            let note = anomaly.to_note();
            tracing::warn!(anomaly = ?anomaly, "Introspection anomaly: {}", note);

            // Add to LLM context notes if not already present
            if !self.memory.anomaly_notes.contains(&note) {
                self.memory.anomaly_notes.push(note);
            }

            // Fire hook
            let hooks = self.hooks.clone();
            safe_hook(|| hooks.on_anomaly_detected(&anomaly, &self.memory));
        }

        // Self-healing: evaluate healing policy if configured
        if let Some(policy) = &mut self.healing_policy {
            if let Some(action) = policy.evaluate(&self.memory) {
                tracing::info!(action = ?action, "Self-healing triggered");
                let outcome = crate::healing::apply_healing(&action, &mut self.memory);
                match outcome {
                    crate::healing::HealingOutcome::ForceFinish => {
                        // Jump to Done state
                        self.state = State::done();
                        return Ok(());
                    }
                    crate::healing::HealingOutcome::Retry => {
                        // Stay in current state — don't apply transition
                        return Ok(());
                    }
                    crate::healing::HealingOutcome::Continue => {
                        // Fall through to normal transition
                    }
                }
            }
        }

        // Look up transition
        let key = (self.state.clone(), event.clone());
        let next_state =
            self.transitions
                .get(&key)
                .cloned()
                .ok_or_else(|| AgentError::InvalidTransition {
                    from: self.state.clone(),
                    event: event.clone(),
                })?;

        // Contract: check transition guards before applying
        if let Some(failure) = self
            .contracts
            .check_guards(&self.state, &next_state, &self.memory)
        {
            match failure.action {
                ContractViolationAction::Block => {
                    tracing::warn!(
                        guard = %failure.contract_name,
                        "Guard blocked transition {} → {} — staying in {}",
                        self.state, next_state, self.state
                    );
                    // Don't apply transition; state stays the same
                    return Ok(());
                }
                ContractViolationAction::EmitEvent(ref evt) => {
                    // Redirect to a different transition via the custom event
                    let alt_key = (self.state.clone(), Event::new(evt));
                    if let Some(alt_next) = self.transitions.get(&alt_key).cloned() {
                        tracing::info!(guard = %failure.contract_name, event = %evt, to = %alt_next, "Guard redirected transition");
                        self.state = alt_next;
                        return Ok(());
                    } else {
                        // No transition for the emitted event — treat as block
                        tracing::warn!(guard = %failure.contract_name, event = %evt, "Guard emitted event but no transition found — blocking");
                        return Ok(());
                    }
                }
                ContractViolationAction::FatalError => {
                    return Err(AgentError::ContractViolation {
                        name: failure.contract_name,
                        message: failure.message,
                    });
                }
                _ => {} // RetryPlanning only for postconditions
            }
        }

        tracing::info!(from = %self.state, event = %event, to = %next_state, "transition");
        println!("  ══ {} --{}-->{} ══", self.state, event, next_state);

        // Plan-and-Execute: advance plan step on Observing→Planning transitions
        let from_state = self.state.clone();
        self.state = next_state;

        if from_state.as_str() == "Observing" && self.state.as_str() == "Planning" {
            if let Some(ref mut plan) = self.memory.current_plan {
                if !plan.is_complete() {
                    // Check if last tool call succeeded or failed
                    if let Some(last) = self.memory.history.last() {
                        if last.success {
                            let obs = last.observation.chars().take(200).collect::<String>();
                            plan.complete_current(obs);
                        } else {
                            let obs = last.observation.chars().take(200).collect::<String>();
                            plan.fail_current(obs);
                        }
                    }
                }
            }
        }

        // Save checkpoint
        if let Some(store) = &self.checkpoint_store {
            let checkpoint: AgentCheckpoint = AgentCheckpoint {
                checkpoint_id: uuid::Uuid::new_v4().to_string(),
                session_id: self.session_id.clone(),
                state: self.state.clone(),
                memory: self.memory.clone(),
                timestamp: chrono::Utc::now(),
            };
            let _ = store.save(checkpoint).await;
        }

        Ok(())
    }

    /// Run the agent and return a stream of AgentOutput events.
    pub fn run_streaming(&mut self) -> BoxStream<'_, AgentOutput> {
        use futures::stream;
        use futures::StreamExt;

        let (tx, rx) = mpsc::unbounded_channel();

        stream::unfold(
            (self, rx, tx, false),
            |(engine, mut rx, tx, mut done)| async move {
                if done {
                    return None;
                }

                // 1. If we have pending messages in the channel (e.g. from the last step or tokens), yield them first.
                if let Ok(msg) = rx.try_recv() {
                    return Some((msg, (engine, rx, tx, false)));
                }

                // 2. Check if we've reached a terminal state AND channel is empty.
                if engine.terminal_states.contains(engine.state.as_str()) {
                    if let Ok(msg) = rx.try_recv() {
                        return Some((msg, (engine, rx, tx, false)));
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
            },
        )
        .boxed()
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
