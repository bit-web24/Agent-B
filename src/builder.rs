use std::collections::HashMap;
use crate::engine::AgentEngine;
use crate::error::AgentError;
use crate::memory::AgentMemory;
use crate::tools::{ToolRegistry, ToolFn};
use crate::llm::LlmCaller;
use crate::states::{
    AgentState, IdleState, PlanningState, ActingState,
    ObservingState, ReflectingState, DoneState, ErrorState,
};
use crate::transitions::build_transition_table;
use crate::types::AgentConfig;

pub struct AgentBuilder {
    memory:  AgentMemory,
    tools:   ToolRegistry,
    llm:     Option<Box<dyn LlmCaller>>,
    config:  Option<AgentConfig>,
}

impl AgentBuilder {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            memory: AgentMemory::new(task),
            tools:  ToolRegistry::new(),
            llm:    None,
            config: None,
        }
    }

    pub fn task_type(mut self, t: impl Into<String>) -> Self {
        self.memory.task_type = t.into(); self
    }

    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.memory.system_prompt = p.into(); self
    }

    pub fn llm(mut self, llm: Box<dyn LlmCaller>) -> Self {
        self.llm = Some(llm); self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config); self
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.memory.config.max_steps = n; self
    }

    pub fn tool(
        mut self,
        name:        impl Into<String>,
        description: impl Into<String>,
        schema:      serde_json::Value,
        func:        ToolFn,
    ) -> Self {
        self.tools.register(name, description, schema, func);
        self
    }

    pub fn blacklist_tool(mut self, name: impl Into<String>) -> Self {
        self.memory.blacklist_tool(name); self
    }

    /// Builds the AgentEngine with all default state handlers.
    pub fn build(mut self) -> Result<AgentEngine, AgentError> {
        let llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required".to_string()))?;

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        // Register default state handlers
        let mut handlers: HashMap<&'static str, Box<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle",       Box::new(IdleState));
        handlers.insert("Planning",   Box::new(PlanningState));
        handlers.insert("Acting",     Box::new(ActingState));
        handlers.insert("Observing",  Box::new(ObservingState));
        handlers.insert("Reflecting", Box::new(ReflectingState));
        handlers.insert("Done",       Box::new(DoneState));
        handlers.insert("Error",      Box::new(ErrorState));

        Ok(AgentEngine::new(
            self.memory,
            self.tools,
            llm,
            build_transition_table(),
            handlers,
        ))
    }

    /// Builds with custom state handlers — for advanced users extending the library.
    /// Any entry in `extra_handlers` will *replace* the default handler for that state name.
    pub fn build_with_handlers(
        mut self,
        extra_handlers: HashMap<&'static str, Box<dyn AgentState>>,
    ) -> Result<AgentEngine, AgentError> {
        let llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required".to_string()))?;

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        // Start with the default set of state handlers.
        let mut handlers: HashMap<&'static str, Box<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle",       Box::new(IdleState));
        handlers.insert("Planning",   Box::new(PlanningState));
        handlers.insert("Acting",     Box::new(ActingState));
        handlers.insert("Observing",  Box::new(ObservingState));
        handlers.insert("Reflecting", Box::new(ReflectingState));
        handlers.insert("Done",       Box::new(DoneState));
        handlers.insert("Error",      Box::new(ErrorState));

        // Merge in any custom overrides — overwriting defaults of the same name.
        for (key, handler) in extra_handlers {
            handlers.insert(key, handler);
        }

        Ok(AgentEngine::new(
            self.memory,
            self.tools,
            llm,
            build_transition_table(),
            handlers,
        ))
    }
}
