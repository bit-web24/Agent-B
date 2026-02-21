use std::collections::{HashMap, HashSet};
use crate::engine::AgentEngine;
use crate::error::AgentError;
use crate::memory::AgentMemory;
use crate::tools::{ToolRegistry, ToolFn, Tool};
use crate::llm::{LlmCaller, LlmCallerExt, OpenAiCaller, AnthropicCaller, RetryingLlmCaller};
use crate::states::{
    AgentState, IdleState, PlanningState, ActingState,
    ObservingState, ReflectingState, DoneState, ErrorState,
};
use crate::transitions::build_transition_table;
use crate::types::{AgentConfig, State};
use crate::events::Event;

pub struct AgentBuilder {
    memory:             AgentMemory,
    tools:              ToolRegistry,
    llm:                Option<Box<dyn LlmCaller>>,
    config:             Option<AgentConfig>,
    retry_count:        Option<u32>,
    custom_handlers:    HashMap<String, Box<dyn AgentState>>,
    custom_transitions: Vec<(State, Event, State)>,
    terminal_states:    HashSet<String>,
}

impl AgentBuilder {
    pub fn new(task: impl Into<String>) -> Self {
        // Default terminal states
        let mut terminal = HashSet::new();
        terminal.insert("Done".to_string());
        terminal.insert("Error".to_string());

        Self {
            memory:             AgentMemory::new(task),
            tools:              ToolRegistry::new(),
            llm:                None,
            config:             None,
            retry_count:        None,
            custom_handlers:    HashMap::new(),
            custom_transitions: Vec::new(),
            terminal_states:    terminal,
        }
    }

    pub fn task_type(mut self, t: impl Into<String>) -> Self {
        self.memory.task_type = t.into(); self
    }

    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.memory.system_prompt = p.into(); self
    }

    // ── LLM provider setters ──────────────────────────────────────────────────

    /// Set the LLM caller explicitly.
    ///
    /// The escape-hatch for any provider not covered by the convenience methods.
    pub fn llm(mut self, llm: Box<dyn LlmCaller>) -> Self {
        self.llm = Some(llm); self
    }

    /// Use the standard OpenAI API.
    ///
    /// Reads `OPENAI_API_KEY` from the environment if you pass `""`,
    /// or pass an explicit key.
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task").openai("sk-...");
    /// // or rely on OPENAI_API_KEY env var (pass empty string is not valid — set .llm() instead)
    /// ```
    pub fn openai(mut self, api_key: impl Into<String>) -> Self {
        let key = api_key.into();
        let caller = if key.is_empty() {
            OpenAiCaller::new()
        } else {
            OpenAiCaller::with_base_url("https://api.openai.com/v1", key)
        };
        self.llm = Some(Box::new(LlmCallerExt(caller)));
        self
    }

    /// Use Groq's ultra-fast inference API (OpenAI-compatible).
    ///
    /// Requires a Groq API key from <https://console.groq.com>.
    /// Set a model via `.model("llama-3.3-70b-versatile")` etc.
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .groq("gsk_...")
    ///     .model("llama-3.3-70b-versatile");
    /// ```
    pub fn groq(mut self, api_key: impl Into<String>) -> Self {
        let caller = OpenAiCaller::with_base_url(
            "https://api.groq.com/openai/v1",
            api_key,
        );
        self.llm = Some(Box::new(LlmCallerExt(caller)));
        self
    }

    /// Use a local Ollama instance (OpenAI-compatible API).
    ///
    /// `base_url` defaults to `"http://localhost:11434/v1"` if empty.
    /// Set the model name via `.model("llama3.2")` etc.
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .ollama("")                // http://localhost:11434/v1
    ///     .model("llama3.2");
    /// ```
    pub fn ollama(mut self, base_url: impl Into<String>) -> Self {
        let url = {
            let s = base_url.into();
            if s.is_empty() { "http://localhost:11434/v1".to_string() } else { s }
        };
        let caller = OpenAiCaller::with_base_url(url, "ollama");
        self.llm = Some(Box::new(LlmCallerExt(caller)));
        self
    }

    /// Use the Anthropic API (Claude models).
    ///
    /// Reads `ANTHROPIC_API_KEY` from the environment if you pass `""`.
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .anthropic("sk-ant-...")
    ///     .model("claude-opus-4-6");
    /// ```
    pub fn anthropic(mut self, api_key: impl Into<String>) -> Self {
        let key   = api_key.into();
        let result = if key.is_empty() {
            AnthropicCaller::from_env()
        } else {
            Ok(AnthropicCaller::new(key))
        };
        // Store error for build() to surface — we can't return Result here
        match result {
            Ok(caller) => self.llm = Some(Box::new(LlmCallerExt(caller))),
            Err(_) => {}  // will fail at build() with "LLM caller is required"
        }
        self
    }

    // ── Retry policy ────────────────────────────────────────────────────────

    /// Wrap the current LLM caller with automatic retry on transient errors.
    ///
    /// - Retries up to `n` times with exponential back-off (1s, 2s, 4s, … cap 30s)
    /// - Auth errors (401/403) are never retried
    /// - Must be called **after** a provider method (`.openai()`, `.groq()`, etc.)
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .groq("gsk_...")
    ///     .model("llama-3.3-70b-versatile")
    ///     .retry_on_error(3);
    /// ```
    pub fn retry_on_error(mut self, n: u32) -> Self {
        self.retry_count = Some(n);
        self
    }

    // ── Configuration ────────────────────────────────────────────────────────

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config); self
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.memory.config.max_steps = n; self
    }

    /// Set the model used for all planning steps (sets `"default"` key).
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task").model("gpt-4o");
    /// AgentBuilder::new("task").model("claude-sonnet-4-6");
    /// AgentBuilder::new("task").model("llama3.2");
    /// ```
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.memory.config.models.insert("default".to_string(), model.into());
        self
    }

    /// Set the model for a specific task type.
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .model("gpt-4o")
    ///     .model_for("calculation", "gpt-4o-mini")
    ///     .model_for("research",    "gpt-4o");
    /// ```
    pub fn model_for(mut self, task_type: impl Into<String>, model: impl Into<String>) -> Self {
        self.memory.config.models.insert(task_type.into(), model.into());
        self
    }

    /// Supply the full model map all at once.
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// use std::collections::HashMap;
    /// let models: HashMap<String, String> = [
    ///     ("default".into(),     "mistral-large-latest".into()),
    ///     ("calculation".into(), "mistral-small-latest".into()),
    /// ].into();
    /// AgentBuilder::new("task").models(models);
    /// ```
    pub fn models(mut self, models: std::collections::HashMap<String, String>) -> Self {
        self.memory.config.models = models;
        self
    }

    // ── Tool registration ────────────────────────────────────────────────────

    /// Register a raw tool (name, description, JSON Schema, function).
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

    /// Register a tool built with the `Tool` builder — ergonomic alternative to `.tool()`.
    ///
    /// ```no_run
    /// # use agentsm::{AgentBuilder, Tool};
    /// # use std::collections::HashMap;
    /// AgentBuilder::new("task")
    ///     .add_tool(
    ///         Tool::new("search", "Search the web")
    ///             .param("query", "string", "The search query")
    ///             .call(|args| Ok(format!("results for {}", args["query"])))
    ///     );
    /// ```
    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.register_tool(tool);
        self
    }

    pub fn blacklist_tool(mut self, name: impl Into<String>) -> Self {
        self.memory.blacklist_tool(name); self
    }

    // ── Custom graph building (LangGraph-style) ──────────────────────────────

    /// Register a custom state handler.
    ///
    /// The handler's `name()` method must return the same string you use
    /// in `.transition()` calls. If the name matches a built-in state
    /// (e.g. `"Planning"`), the custom handler **replaces** the default one.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// # struct ResearchingState;
    /// # impl agentsm::states::AgentState for ResearchingState {
    /// #   fn name(&self) -> &'static str { "Researching" }
    /// #   fn handle(&self, _: &mut agentsm::AgentMemory, _: &agentsm::ToolRegistry, _: &dyn agentsm::LlmCaller) -> agentsm::Event { agentsm::Event::new("ResearchDone") }
    /// # }
    /// AgentBuilder::new("task")
    ///     .openai("sk-...")
    ///     .state("Researching", Box::new(ResearchingState));
    /// ```
    pub fn state(mut self, name: impl Into<String>, handler: Box<dyn AgentState>) -> Self {
        self.custom_handlers.insert(name.into(), handler);
        self
    }

    /// Add a transition edge: when the agent is in `from_state` and the
    /// handler emits `on_event`, transition to `to_state`.
    ///
    /// If the `(from, event)` pair already exists in the default table,
    /// the custom transition **overwrites** it.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .openai("sk-...")
    ///     .transition("Planning",    "NeedsResearch", "Researching")
    ///     .transition("Researching", "ResearchDone",  "Planning");
    /// ```
    pub fn transition(
        mut self,
        from_state: impl Into<String>,
        on_event:   impl Into<String>,
        to_state:   impl Into<String>,
    ) -> Self {
        self.custom_transitions.push((
            State::new(from_state),
            Event::new(on_event),
            State::new(to_state),
        ));
        self
    }

    /// Mark a state as terminal — the engine will exit when it reaches
    /// this state.
    ///
    /// `"Done"` and `"Error"` are terminal by default. Use this to add
    /// additional terminal states for custom graphs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use agentsm::AgentBuilder;
    /// AgentBuilder::new("task")
    ///     .openai("sk-...")
    ///     .terminal_state("CustomDone");
    /// ```
    pub fn terminal_state(mut self, name: impl Into<String>) -> Self {
        self.terminal_states.insert(name.into());
        self
    }

    // ── Build ────────────────────────────────────────────────────────────────

    /// Builds the AgentEngine with all default state handlers,
    /// plus any custom states, transitions, and terminal states.
    pub fn build(mut self) -> Result<AgentEngine, AgentError> {
        let mut llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required. Use .openai(), .groq(), .ollama(), .anthropic(), or .llm()".to_string()))?;

        // Wrap with retry if requested
        if let Some(n) = self.retry_count {
            llm = Box::new(RetryingLlmCaller::new(llm, n));
        }

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        // Register default state handlers
        let mut handlers: HashMap<String, Box<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle".to_string(),       Box::new(IdleState));
        handlers.insert("Planning".to_string(),   Box::new(PlanningState));
        handlers.insert("Acting".to_string(),     Box::new(ActingState));
        handlers.insert("Observing".to_string(),  Box::new(ObservingState));
        handlers.insert("Reflecting".to_string(), Box::new(ReflectingState));
        handlers.insert("Done".to_string(),       Box::new(DoneState));
        handlers.insert("Error".to_string(),      Box::new(ErrorState));

        // Merge custom handlers (overwriting defaults if same name)
        for (name, handler) in self.custom_handlers {
            handlers.insert(name, handler);
        }

        // Build transition table (defaults + custom)
        let mut transitions = build_transition_table();
        for (from, event, to) in self.custom_transitions {
            transitions.insert((from, event), to);
        }

        Ok(AgentEngine::new(
            self.memory,
            self.tools,
            llm,
            transitions,
            handlers,
            self.terminal_states,
        ))
    }

    /// Builds with custom state handlers — for advanced users extending the library.
    /// Any entry in `extra_handlers` will *replace* the default handler for that state name.
    pub fn build_with_handlers(
        mut self,
        extra_handlers: HashMap<String, Box<dyn AgentState>>,
    ) -> Result<AgentEngine, AgentError> {
        let mut llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required".to_string()))?;

        if let Some(n) = self.retry_count {
            llm = Box::new(RetryingLlmCaller::new(llm, n));
        }

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        // Start with the default set of state handlers.
        let mut handlers: HashMap<String, Box<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle".to_string(),       Box::new(IdleState));
        handlers.insert("Planning".to_string(),   Box::new(PlanningState));
        handlers.insert("Acting".to_string(),     Box::new(ActingState));
        handlers.insert("Observing".to_string(),  Box::new(ObservingState));
        handlers.insert("Reflecting".to_string(), Box::new(ReflectingState));
        handlers.insert("Done".to_string(),       Box::new(DoneState));
        handlers.insert("Error".to_string(),      Box::new(ErrorState));

        // Merge custom handlers from .state() calls
        for (name, handler) in self.custom_handlers {
            handlers.insert(name, handler);
        }

        // Merge in extra_handlers (these take highest priority)
        for (key, handler) in extra_handlers {
            handlers.insert(key, handler);
        }

        // Build transition table (defaults + custom)
        let mut transitions = build_transition_table();
        for (from, event, to) in self.custom_transitions {
            transitions.insert((from, event), to);
        }

        Ok(AgentEngine::new(
            self.memory,
            self.tools,
            llm,
            transitions,
            handlers,
            self.terminal_states,
        ))
    }
}
