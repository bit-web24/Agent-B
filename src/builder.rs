use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::engine::AgentEngine;
use crate::error::AgentError;
use crate::memory::AgentMemory;
use crate::tools::{ToolRegistry, ToolFn, Tool};
use crate::llm::{AsyncLlmCaller, OpenAiCaller, AnthropicCaller, RetryingLlmCaller};
use crate::states::{
    AgentState, IdleState, PlanningState, ActingState, ParallelActingState,
    ObservingState, ReflectingState, DoneState, ErrorState,
    WaitingForHumanState,
};
use crate::checkpoint::CheckpointStore;
use crate::budget::TokenBudget;
use crate::transitions::build_transition_table;
use crate::types::{AgentConfig, State};
use crate::events::Event;
use crate::mcp::{McpClient, bridge_mcp_tool};

#[derive(Clone)]
pub struct AgentBuilder {
    memory:             AgentMemory,
    tools:              ToolRegistry,
    llm:                Option<Arc<dyn AsyncLlmCaller>>,
    config:             Option<AgentConfig>,
    retry_count:        Option<u32>,
    custom_handlers:    HashMap<String, Arc<dyn AgentState>>,
    custom_transitions: Vec<(State, Event, State)>,
    terminal_states:    HashSet<String>,
    checkpoint_store:   Option<Arc<dyn CheckpointStore>>,
    session_id:         String,
    initial_state:      Option<State>,
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
            checkpoint_store:   None,
            session_id:         uuid::Uuid::new_v4().to_string(),
            initial_state:      None,
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
    pub fn llm(mut self, llm: Arc<dyn AsyncLlmCaller>) -> Self {
        self.llm = Some(llm); self
    }

    /// Use the standard OpenAI API.
    pub fn openai(mut self, api_key: impl Into<String>) -> Self {
        let key = api_key.into();
        let caller = if key.is_empty() {
            OpenAiCaller::new()
        } else {
            OpenAiCaller::with_base_url("https://api.openai.com/v1", key)
        };
        self.llm = Some(Arc::new(caller));
        self
    }

    /// Use Groq's ultra-fast inference API (OpenAI-compatible).
    pub fn groq(mut self, api_key: impl Into<String>) -> Self {
        let caller = OpenAiCaller::with_base_url(
            "https://api.groq.com/openai/v1",
            api_key.into(),
        );
        self.llm = Some(Arc::new(caller));
        self
    }

    /// Use a local Ollama instance (OpenAI-compatible API).
    pub fn ollama(mut self, base_url: impl Into<String>) -> Self {
        let url = {
            let s = base_url.into();
            if s.is_empty() { "http://localhost:11434/v1".to_string() } else { s }
        };
        let caller = OpenAiCaller::with_base_url(url, "ollama");
        self.llm = Some(Arc::new(caller));
        self
    }

    /// Set the maximum number of total tokens allowed for this agent session.
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.memory.budget = Some(TokenBudget::new(max));
        self
    }

    /// Set a detailed token budget.
    pub fn token_budget(mut self, budget: TokenBudget) -> Self {
        self.memory.budget = Some(budget);
        self
    }

    /// Use the Anthropic API (Claude models).
    pub fn anthropic(mut self, api_key: impl Into<String>) -> Self {
        let key   = api_key.into();
        let result = if key.is_empty() {
            AnthropicCaller::from_env()
        } else {
            Ok(AnthropicCaller::new(key))
        };
        if let Ok(caller) = result {
            self.llm = Some(Arc::new(caller));
        }
        self
    }

    // ── Retry policy ────────────────────────────────────────────────────────

    pub fn retry_on_error(mut self, n: u32) -> Self {
        self.retry_count = Some(n);
        self
    }

    // ── Configuration ────────────────────────────────────────────────────────

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config); self
    }

    /// Set the checkpoint store for persistence.
    pub fn checkpoint_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }

    /// Set a custom session ID.
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    /// Resume an agent from the latest checkpoint of a session.
    pub async fn resume(mut self, session_id: &str) -> Result<Self, AgentError> {
        let store = self.checkpoint_store.as_ref()
            .ok_or_else(|| AgentError::BuildError("Checkpoint store must be set before calling .resume()".to_string()))?;
        
        let checkpoint = store.load_latest(session_id).await
            .map_err(|e| AgentError::BuildError(format!("Failed to load checkpoint: {}", e)))?
            .ok_or_else(|| AgentError::BuildError(format!("No checkpoint found for session: {}", session_id)))?;
        
        self.memory = checkpoint.memory;
        self.session_id = checkpoint.session_id;
        self.initial_state = Some(checkpoint.state);

        Ok(self)
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.memory.config.max_steps = n; self
    }

    /// Enable or disable parallel tool execution.
    pub fn parallel_tools(mut self, enabled: bool) -> Self {
        self.memory.config.parallel_tools = enabled; 
        self
    }

    /// Require human approval for certain tools.
    pub fn approval_policy(mut self, policy: crate::human::ApprovalPolicy) -> Self {
        self.memory.approval_policy = policy;
        self
    }

    /// Callback for human approval.
    pub fn on_approval<F>(mut self, callback: F) -> Self 
    where F: Fn(crate::human::HumanApprovalRequest) -> crate::human::HumanDecision + Send + Sync + 'static {
        self.memory.approval_callback = Some(crate::memory::ApprovalCallback(std::sync::Arc::new(callback)));
        self
    }

    /// Set the model used for all planning steps (sets `"default"` key).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.memory.config.models.insert("default".to_string(), model.into());
        self
    }

    /// Set the model for a specific task type.
    pub fn model_for(mut self, task_type: impl Into<String>, model: impl Into<String>) -> Self {
        self.memory.config.models.insert(task_type.into(), model.into());
        self
    }

    /// Supply the full model map all at once.
    pub fn models(mut self, models: std::collections::HashMap<String, String>) -> Self {
        self.memory.config.models = models;
        self
    }

    // ── Tool registration ────────────────────────────────────────────────────

    /// Register a raw tool.
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

    /// Register a tool built with the `Tool` builder.
    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.register_tool(tool);
        self
    }

    /// Register an MCP server and all its tools.
    pub fn mcp_server(mut self, command: impl Into<String>, args: &[String]) -> Self {
        let cmd = command.into();
        let args = args.to_vec();

        tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            let client = handle.block_on(McpClient::new(&cmd, &args))
                .expect("Failed to initialize MCP client");

            let tools = handle.block_on(client.list_tools())
                .expect("Failed to list MCP tools");

            for mcp_tool in tools {
                let name = mcp_tool.name.clone();
                let desc = mcp_tool.description.clone().unwrap_or_default();
                let schema = mcp_tool.input_schema.clone().unwrap_or_default();
                let func = bridge_mcp_tool(Arc::clone(&client), name.clone());

                self.tools.register(name, desc, schema, func);
            }
        });

        self
    }

    pub fn blacklist_tool(mut self, name: impl Into<String>) -> Self {
        self.memory.blacklist_tool(name); self
    }

    // ── Custom graph building ────────────────────────────────────────────────

    pub fn state(mut self, name: impl Into<String>, handler: Arc<dyn AgentState>) -> Self {
        self.custom_handlers.insert(name.into(), handler);
        self
    }

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

    pub fn terminal_state(mut self, name: impl Into<String>) -> Self {
        self.terminal_states.insert(name.into());
        self
    }

    // ── Sub-Agents as Tools ──────────────────────────────────────────────

    /// Converts this builder into a tool that can be used by another agent.
    /// When called, it runs a new instance of the agent with the given task.
    pub fn as_tool(&self, name: impl Into<String>, description: impl Into<String>) -> Tool {
        let builder = self.clone();
        Tool::new(name, description)
            .param("task", "string", "The task to delegate to this specialized sub-agent")
            .call(move |args| {
                let task = args.get("task").and_then(|v| v.as_str())
                    .ok_or_else(|| "Missing required 'task' parameter for sub-agent".to_string())?;

                let mut sub_agent_builder = builder.clone();
                sub_agent_builder.memory.task = task.to_string();
                
                // We use block_in_place because sub-agents run synchronously within the tool call
                tokio::task::block_in_place(|| {
                    let mut engine = sub_agent_builder.build()
                        .map_err(|e| format!("Failed to build sub-agent: {}", e))?;
                    
                    let handle = tokio::runtime::Handle::current();
                    handle.block_on(engine.run())
                        .map_err(|e| format!("Sub-agent failed: {}", e))
                })
            })
    }

    /// Convenience method to register a sub-agent as a tool.
    pub fn add_subagent(self, name: impl Into<String>, description: impl Into<String>, subagent: AgentBuilder) -> Self {
        let name_s = name.into();
        let tool = subagent.as_tool(name_s.clone(), description);
        self.add_tool(tool)
    }

    // ── Build ────────────────────────────────────────────────────────────────

    pub fn build(mut self) -> Result<AgentEngine, AgentError> {
        let mut llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required.".to_string()))?;

        if let Some(n) = self.retry_count {
            llm = Arc::new(RetryingLlmCaller::new(llm, n));
        }

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        let mut handlers: HashMap<String, Arc<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle".to_string(),       Arc::new(IdleState));
        handlers.insert("Planning".to_string(),   Arc::new(PlanningState));
        handlers.insert("Acting".to_string(),     Arc::new(ActingState));
        handlers.insert("ParallelActing".to_string(), Arc::new(ParallelActingState));
        handlers.insert("Observing".to_string(),  Arc::new(ObservingState));
        handlers.insert("Reflecting".to_string(), Arc::new(ReflectingState));
        handlers.insert("Done".to_string(),       Arc::new(DoneState));
        handlers.insert("Error".to_string(),      Arc::new(ErrorState));
        handlers.insert("WaitingForHuman".to_string(), Arc::new(WaitingForHumanState));

        for (name, handler) in self.custom_handlers {
            handlers.insert(name, handler);
        }

        let mut transitions = build_transition_table();
        for (from, event, to) in self.custom_transitions {
            transitions.insert((from, event), to);
        }

        let mut engine = AgentEngine::new(
            self.memory,
            Arc::new(self.tools),
            llm,
            transitions,
            handlers,
            self.terminal_states,
            self.session_id,
            self.checkpoint_store,
        );

        if let Some(state) = self.initial_state {
            engine.state = state;
        }

        Ok(engine)
    }

    pub fn build_with_handlers(
        mut self,
        extra_handlers: HashMap<String, Arc<dyn AgentState>>,
    ) -> Result<AgentEngine, AgentError> {
        let mut llm = self.llm
            .ok_or_else(|| AgentError::BuildError("LLM caller is required".to_string()))?;

        if let Some(n) = self.retry_count {
            llm = Arc::new(RetryingLlmCaller::new(llm, n));
        }

        if let Some(config) = self.config {
            self.memory.config = config;
        }

        let mut handlers: HashMap<String, Arc<dyn AgentState>> = HashMap::new();
        handlers.insert("Idle".to_string(),       Arc::new(IdleState));
        handlers.insert("Planning".to_string(),   Arc::new(PlanningState));
        handlers.insert("Acting".to_string(),     Arc::new(ActingState));
        handlers.insert("ParallelActing".to_string(), Arc::new(ParallelActingState));
        handlers.insert("Observing".to_string(),  Arc::new(ObservingState));
        handlers.insert("Reflecting".to_string(), Arc::new(ReflectingState));
        handlers.insert("Done".to_string(),       Arc::new(DoneState));
        handlers.insert("Error".to_string(),      Arc::new(ErrorState));
        handlers.insert("WaitingForHuman".to_string(), Arc::new(WaitingForHumanState));

        for (name, handler) in self.custom_handlers {
            handlers.insert(name, handler);
        }

        for (key, handler) in extra_handlers {
            handlers.insert(key, handler);
        }

        let mut transitions = build_transition_table();
        for (from, event, to) in self.custom_transitions {
            transitions.insert((from, event), to);
        }

        let mut engine = AgentEngine::new(
            self.memory,
            Arc::new(self.tools),
            llm,
            transitions,
            handlers,
            self.terminal_states,
            self.session_id,
            self.checkpoint_store,
        );

        if let Some(state) = self.initial_state {
            engine.state = state;
        }

        Ok(engine)
    }
}
