use agentsm::{AgentBuilder, AgentOutput, Tool};
use futures::StreamExt;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Agent-B Streaming Example...");

    // 1. Build the agent
    // We'll use OpenAI for this example. Ensure OPENAI_API_KEY is set.
    let mut agent = AgentBuilder::new("Explain the importance of Rust in system programming and then use the 'calculator' tool to add 123 and 456.")
        .openai("") // Reads from OPENAI_API_KEY env var
        .model("gpt-4o")
        .add_tool(
            Tool::new("calculator", "Adds two numbers")
                .param("a", "number", "First number")
                .param("b", "number", "Second number")
                .call(|args| {
                    let a = args["a"].as_f64().unwrap_or(0.0);
                    let b = args["b"].as_f64().unwrap_or(0.0);
                    Ok((a + b).to_string())
                })
        )
        .build()?;

    // 2. Run with streaming
    println!("\n--- Streaming Output ---\n");
    let mut stream = agent.run_streaming();

    while let Some(output) = stream.next().await {
        match output {
            AgentOutput::StateStarted(state) => {
                println!("\n\n[STATE] Entering: {}", state);
            }
            AgentOutput::LlmToken(token) => {
                print!("{}", token);
                stdout().flush()?;
            }
            AgentOutput::ToolCallDelta { name, args_json } => {
                // For a real-time UI, we'd update a partial view. 
                // Here we'll just show it was called if name is present.
                if let Some(_n) = name {
                    if !args_json.is_empty() && args_json.len() < 20 {
                        // Just an indicator
                        print!("⚡"); 
                        stdout().flush()?;
                    }
                }
            }
            AgentOutput::ToolCallStarted { name, args } => {
                println!("\n[TOOL CALL] {} with arguments: {:?}", name, args);
            }
            AgentOutput::ToolCallFinished { name, result, success } => {
                println!("[TOOL RESULT] {} (Success: {}): {}", name, success, result);
            }
            AgentOutput::Action(msg) => {
                println!("\n[ACTION] {}", msg);
            }
            AgentOutput::FinalAnswer(answer) => {
                println!("\n\n✅ [FINAL ANSWER]\n{}", answer);
            }
            AgentOutput::Error(err) => {
                eprintln!("\n❌ [ERROR] {}", err);
            }
        }
    }

    println!("\n--- Streaming Complete ---\n");
    Ok(())
}
