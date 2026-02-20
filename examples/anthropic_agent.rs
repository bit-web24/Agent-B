//! # Anthropic Agent Example
//!
//! Demonstrates using `AnthropicCaller` with Claude models to build an agent
//! that uses a knowledge base lookup tool.
//!
//! # Usage
//! ```bash
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_agent
//! RUST_LOG=info ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_agent
//! ```

use agentsm::AgentBuilder;
use agentsm::llm::{AnthropicCaller, LlmCallerExt};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== agentsm-rs Anthropic Agent Example ===\n");
    println!("Using: AnthropicCaller (Claude via native reqwest HTTP)\n");

    // Load the Anthropic API key from the ANTHROPIC_API_KEY environment variable
    let anthropic = AnthropicCaller::from_env()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Wrap the async Anthropic caller into a sync LlmCaller for the engine
    let llm = Box::new(LlmCallerExt(anthropic));

    let mut engine = AgentBuilder::new(
            "What are the key design principles of the Rust programming language, \
             and how does its ownership model prevent memory safety issues?"
        )
        .task_type("research")
        // Specify the Claude model explicitly — no model names are hardcoded in the library.
        // Swap out for any model your Anthropic plan supports:
        //   "claude-opus-4-6"           — highest quality
        //   "claude-sonnet-4-6"         — balanced
        //   "claude-haiku-4-5-20251001" — fast and cheap
        .model("claude-opus-4-6")
        .system_prompt(
            "You are an expert software engineer specializing in systems programming. \
             Use the knowledge_base tool to retrieve accurate technical information \
             before composing your answer. Provide thorough, well-structured responses."
        )
        .llm(llm)
        .max_steps(8)
        // ── Tool: Knowledge Base Lookup ───────────────────────────────────────
        .tool(
            "knowledge_base",
            "Retrieve technical documentation and articles from the knowledge base. \
             Use this to look up programming concepts, language features, and best practices.",
            json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The technical topic to look up, e.g. 'Rust ownership model'"
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "detailed", "comprehensive"],
                        "description": "How much detail to return"
                    }
                },
                "required": ["topic"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let topic  = args.get("topic").and_then(|v| v.as_str()).unwrap_or("unknown");
                let detail = args.get("detail_level").and_then(|v| v.as_str()).unwrap_or("summary");

                // Mock knowledge base — in production, query a vector DB or document store
                let content = match topic.to_lowercase().as_str() {
                    t if t.contains("ownership") || t.contains("memory") => {
                        "Rust's ownership model is based on three rules: \
                         (1) Each value has exactly one owner at a time. \
                         (2) When the owner goes out of scope, the value is dropped (RAII). \
                         (3) Ownership can be transferred (moved) or temporarily borrowed. \
                         Borrowing allows &T (shared, read-only) or &mut T (exclusive, mutable). \
                         The borrow checker enforces these at compile time with zero runtime cost, \
                         preventing use-after-free, double-free, and data races."
                    }
                    t if t.contains("rust") && t.contains("design") => {
                        "Rust's key design principles: \
                         Memory safety without garbage collection (ownership + borrow checker), \
                         Zero-cost abstractions (iterators, generics compile to optimal machine code), \
                         Fearless concurrency (the type system prevents data races), \
                         Expressive type system (enums, traits, pattern matching), \
                         Practical error handling (Result<T,E>, no exceptions), \
                         Interoperability with C via FFI, \
                         First-class tooling (Cargo, rustfmt, clippy)."
                    }
                    _ => {
                        "Documentation not found for this specific topic. \
                         Please try a more specific query about Rust features, \
                         ownership, lifetimes, traits, or other language concepts."
                    }
                };

                let response = match detail {
                    "comprehensive" => format!("[KB: {}]\n\n{}\n\n[Related: ownership, lifetimes, traits, concurrency]", topic, content),
                    "detailed"      => format!("[KB: {}]\n\n{}", topic, content),
                    _               => format!("[KB: {}] {}", topic, &content[..content.len().min(200)]),
                };

                Ok(response)
            }),
        )
        .build()?;

    match engine.run() {
        Ok(answer) => {
            println!("\n╔═══════════════════════════════════════╗");
            println!("║           FINAL ANSWER                ║");
            println!("╚═══════════════════════════════════════╝\n");
            println!("{}\n", answer);

            println!("╔═══════════════════════════════════════╗");
            println!("║         EXECUTION TRACE               ║");
            println!("╚═══════════════════════════════════════╝");
            engine.trace().print();
        }
        Err(e) => {
            eprintln!("Agent failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
