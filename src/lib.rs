pub mod budget;
pub mod builder;
pub mod cache;
pub mod checkpoint;
pub mod contracts;
pub mod engine;
pub mod error;
pub mod events;
pub mod healing;
pub mod hooks;
pub mod human;
pub mod introspection;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod memory_strategy;
pub mod prompt;
pub mod routing;
pub mod states;
pub mod tools;
pub mod trace;
pub mod transitions;
pub mod types;

// Convenience re-exports at crate root
pub use builder::AgentBuilder;
pub use cache::{CacheStats, InMemoryCache, LlmCache, NoopCache};
pub use contracts::{
    ContractSet, ContractViolationAction, GuardFailAction, Invariant, InvariantFailAction,
    PostCondition, PostConditionFailAction, TransitionGuard,
};
pub use engine::AgentEngine;
pub use error::AgentError;
pub use events::Event;
pub use healing::{apply_healing, HealingAction, HealingOutcome, HealingPolicy, HealingTrigger};
pub use hooks::{AgentHooks, CompositeHooks, NoopHooks, PrintHooks};
pub use introspection::{Anomaly, IntrospectionConfig, IntrospectionEngine};
pub use llm::{AsyncLlmCaller, LlmCaller, LlmCallerExt, RetryingLlmCaller};
pub use memory::AgentMemory;
pub use memory_strategy::{FullMemory, MemoryStrategy, SlidingWindowMemory, SummaryMemory};
pub use prompt::{PromptError, PromptTemplate};
pub use routing::{
    BudgetPctAbove, ConfidenceBelow, RoutingCondition, RoutingPolicy, RoutingRule, StepAbove,
    ToolFailureRateAbove,
};
pub use tools::{Tool, ToolFn, ToolRegistry};
pub use trace::{Trace, TraceEntry};
pub use types::{
    AgentConfig, AgentOutput, HistoryEntry, LlmResponse, LlmStreamChunk, OutputSchema, State,
    ToolCall,
};
