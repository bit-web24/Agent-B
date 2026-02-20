//! # Basic Agent Example
//!
//! Demonstrates the minimal working agent using OpenAI and a single mock search tool.
//!
//! # Usage
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example basic_agent
//! RUST_LOG=debug OPENAI_API_KEY=sk-... cargo run --example basic_agent
//! ```

use agentsm::{AgentBuilder, AgentConfig};
use agentsm::llm::{OpenAiCaller, LlmCallerExt};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize structured logging — set RUST_LOG=debug|info|warn
    tracing_subscriber::fmt::init();

    println!("=== agentsm-rs Basic Agent Example ===\n");

    // Build an OpenAI LLM caller (reads OPENAI_API_KEY from environment)
    let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));

    // Configure the agent with a reasonable cap
    let config = AgentConfig {
        max_steps:             10,
        max_retries:           3,
        confidence_threshold:  0.4,
        reflect_every_n_steps: 5,
        min_answer_length:     20,
    };

    let mut engine = AgentBuilder::new(
            "What is the capital of France and what is its population?"
        )
        .task_type("research")
        .system_prompt(
            "You are a helpful research assistant. \
             Use the search tool to find information before answering."
        )
        .llm(llm)
        .config(config)
        .tool(
            "search",
            "Search the web for current information. Use for any factual queries.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let query = args.get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                // In production, call a real search API here (e.g. Serper, Tavily, Brave)
                Ok(format!(
                    "Search results for '{}': Paris is the capital of France. \
                     The city of Paris has a population of approximately 2.1 million people, \
                     while the Greater Paris metropolitan area has around 12 million inhabitants.",
                    query
                ))
            }),
        )
        .build()?;

    // Run the agent to completion
    match engine.run() {
        Ok(answer) => {
            println!("\n╔══════════════════════╗");
            println!("║    FINAL ANSWER      ║");
            println!("╚══════════════════════╝");
            println!("{}\n", answer);

            println!("╔══════════════════════╗");
            println!("║       TRACE          ║");
            println!("╚══════════════════════╝");
            engine.trace().print();
        }
        Err(e) => {
            eprintln!("Agent failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
