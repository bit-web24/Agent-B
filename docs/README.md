# agentsm-rs Documentation

Welcome to the **agentsm-rs** documentation — a production-grade Agentic AI library for Rust built on the Hybrid State Machine pattern.

## Table of Contents

| Document | Description |
|---|---|
| [Getting Started](./getting-started.md) | Installation, quick start, first agent in 5 minutes |
| [Architecture](./architecture.md) | How the library works — state machine, engine, layers |
| [Core Concepts](./core-concepts.md) | States, Events, Memory, Transitions — the building blocks |
| [LLM Providers](./llm-providers.md) | OpenAI, Anthropic, OpenAI-compatible, and custom providers |
| [Tool System](./tool-system.md) | Registering tools, schemas, execution, error handling |
| [Configuration](./configuration.md) | AgentConfig, tuning steps, confidence, reflection |
| [Examples](./examples.md) | Annotated walkthrough of all bundled examples |
| [Testing Guide](./testing.md) | Using MockLlmCaller, writing unit and integration tests |
| [Advanced Usage](./advanced.md) | Custom states, custom transitions, multi-agent patterns |
| [API Reference](./api-reference.md) | Complete public API surface — all types and traits |
| [Error Handling](./error-handling.md) | AgentError variants, the failure-as-data philosophy |

## Design in One Sentence

> The **Transition Table** owns the graph. **State Traits** own the behavior. **You** own everything else.

## Three Laws

1. **The Transition Table is the Single Source of Truth.** Every legal state transition lives in one `HashMap`. If it's not in the table, it cannot happen.

2. **Failure is Data, Not Exceptions.** Tool errors and bad LLM outputs flow back to the agent as observations. The agent decides what to do next.

3. **The Engine is Dumb.** `AgentEngine` knows nothing about what states *do*. It only wires together: state handler → event → transition table → next state.
