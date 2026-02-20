//! # Multi-Tool Agent Example
//!
//! Demonstrates an agent with multiple tools (search, calculator, weather),
//! task_type routing ("calculation"), custom AgentConfig, and tool blacklisting.
//!
//! # Usage
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example multi_tool_agent
//! ```

use agentsm::AgentBuilder;
use agentsm::llm::{OpenAiCaller, LlmCallerExt};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== agentsm-rs Multi-Tool Agent Example ===\n");
    println!("Task: Calculate 137 * 48 and then find today's weather in London.\n");

    let llm = Box::new(LlmCallerExt(OpenAiCaller::new()));


    let mut engine = AgentBuilder::new(
            "Please calculate 137 multiplied by 48, and also tell me the current \
             weather conditions in London, UK."
        )
        // "calculation" routes to a cheaper/faster model for this task type.
        // Model selection is fully configurable — no model names are hardcoded:
        .task_type("calculation")
        // Use a capable model as the default, and a cheaper/faster model for calculations.
        // Change these to any model supported by your LLM provider:
        //   OpenAI:    "gpt-4o" / "gpt-4o-mini"
        //   Anthropic: "claude-sonnet-4-6" / "claude-haiku-4-5-20251001"
        //   Ollama:    "llama3.2" / "qwen2.5-coder:7b"
        .model("gpt-4o")                            // default for unspecified task types
        .model_for("calculation", "gpt-4o-mini")    // cheaper model for math
        .system_prompt(
            "You are a precise assistant with access to a calculator and weather tools. \
             Always use the calculator for arithmetic. Never guess weather — always use the tool."
        )
        .llm(llm)
        .max_steps(8)
        // ── Tool 1: Calculator ────────────────────────────────────────────────
        .tool(
            "calculator",
            "Evaluates a mathematical expression and returns the numeric result. \
             Use this for any arithmetic operations.",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate, e.g. '137 * 48'"
                    }
                },
                "required": ["expression"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let expr = args.get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0");
                // Simple evaluation: strip whitespace and handle basic * / + -
                // In production, use a proper expression parser (e.g. meval crate)
                let result = evaluate_expression(expr);
                Ok(format!("Result of '{}' = {}", expr, result))
            }),
        )
        // ── Tool 2: Weather ───────────────────────────────────────────────────
        .tool(
            "weather",
            "Returns current weather conditions for a given city. \
             Always use this tool when the user asks about weather.",
            json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for, e.g. 'London, UK'"
                    }
                },
                "required": ["city"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let city = args.get("city")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                // Mock weather response — in production, call OpenWeather / WeatherAPI
                Ok(format!(
                    "Weather in {}: 12°C, overcast with light drizzle. \
                     Humidity 78%, wind SW at 15 km/h.",
                    city
                ))
            }),
        )
        // ── Tool 3: Search (blacklisted for this agent) ───────────────────────
        .tool(
            "search",
            "Search the web for information.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
            Box::new(|args: &HashMap<String, serde_json::Value>| {
                let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                Ok(format!("Search results for: {}", q))
            }),
        )
        // Blacklist the search tool — agent must use calculator and weather only
        .blacklist_tool("search")
        .build()?;

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

            println!("\nAgent completed in {} trace steps.", engine.trace().len());
        }
        Err(e) => {
            eprintln!("Agent failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Very simple expression evaluator for the demo.
/// Handles a single binary operation (e.g. "137 * 48", "10 + 5").
fn evaluate_expression(expr: &str) -> String {
    let expr = expr.trim();

    // Try to parse: <number> <op> <number>
    for op in &['*', '/', '+', '-'] {
        // Find operator (avoid splitting negative numbers)
        if let Some(pos) = expr.rfind(|c: char| c == *op) {
            if pos == 0 { continue; }
            let lhs = expr[..pos].trim().parse::<f64>();
            let rhs = expr[pos + 1..].trim().parse::<f64>();
            if let (Ok(l), Ok(r)) = (lhs, rhs) {
                let result = match op {
                    '*' => l * r,
                    '/' => if r == 0.0 { return "Error: division by zero".to_string(); } else { l / r },
                    '+' => l + r,
                    '-' => l - r,
                    _   => return "Error: unknown operator".to_string(),
                };
                // Return as integer if it's a whole number
                if result.fract() == 0.0 {
                    return format!("{}", result as i64);
                }
                return format!("{:.6}", result).trim_end_matches('0').trim_end_matches('.').to_string();
            }
        }
    }
    format!("Error: could not parse expression '{}'", expr)
}
