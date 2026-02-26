use agentsm::{AgentBuilder, Tool, AgentOutput};
use std::time::Duration;
use chrono::Utc;
use std::sync::{Arc, Mutex};
use futures::StreamExt;

#[tokio::test]
async fn test_parallel_tool_execution_real_api() {
    let api_key = match std::env::var("GROQ_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping real API test: GROQ_API_KEY not set");
            return;
        }
    };
    
    // We'll use a shared log to track when tools start and end
    let execution_log = Arc::new(Mutex::new(Vec::new()));
    
    let log_for_tool = execution_log.clone();

    let mut engine = AgentBuilder::new("Get the weather for both New York and San Francisco simultaneously. Use the get_weather tool for BOTH cities in ONE response.")
        .llm(Box::new(agentsm::llm::OpenAiCaller::with_base_url("https://api.groq.com/openai/v1", api_key)))
        .model("llama-3.3-70b-versatile")
        .parallel_tools(true)
        .add_tool(
            Tool::new("get_weather", "Get weather for a city")
                .param("city", "string", "The city name")
                .call(move |args| {
                    let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let city_owned = city.to_string();
                    
                    println!("--- Tool Starting: {} ---", city_owned);
                    {
                        let mut log = log_for_tool.lock().unwrap();
                        log.push((format!("start_{}", city_owned), Utc::now()));
                    }
                    
                    // Simulate I/O latency
                    std::thread::sleep(Duration::from_millis(1000));
                    
                    {
                        let mut log = log_for_tool.lock().unwrap();
                        log.push((format!("end_{}", city_owned), Utc::now()));
                    }
                    println!("--- Tool Finished: {} ---", city_owned);
                    
                    Ok(format!("The weather in {} is sunny, 22°C", city_owned))
                })
        )
        .build()
        .expect("Failed to build agent");

    println!("Running agent with parallel tools enabled...");
    let mut stream = engine.run_streaming();
    let mut final_answer = String::new();

    while let Some(output) = stream.next().await {
        match output {
            AgentOutput::StateStarted(state) => println!("State: {:?}", state),
            AgentOutput::LlmToken(token) => print!("{}", token),
            AgentOutput::ToolCallStarted { name, args } => println!("\n[Tool Call] {}: {:?}", name, args),
            AgentOutput::ToolCallFinished { name, result, .. } => println!("\n[Tool Result] {}: {}", name, result),
            AgentOutput::FinalAnswer(ans) => final_answer = ans,
            AgentOutput::Error(err) => {
                println!("\n[AGENT ERROR] {}", err);
                panic!("Agent Error: {}", err)
            },
            _ => {}
        }
    }
    
    drop(stream);
    
    if !final_answer.is_empty() {
        println!("\nFinal Answer: {}", final_answer);
    } else if let Some(err) = &engine.memory.error {
        println!("\n[ENGINE ERROR] {}", err);
    }
    
    let log = execution_log.lock().unwrap();
    let events: Vec<String> = log.iter().map(|(e, _)| e.clone()).collect();
    println!("Execution Events: {:?}", events);
    
    if log.len() < 4 {
        if let Some(err) = &engine.memory.error {
             panic!("Test failed with engine error: {}", err);
        }
    }
    
    // Verify that at least two tool calls were made
    assert!(log.len() >= 4, "Expected at least 2 tool calls (4 events). Found: {:?}", events);
    
    // To prove parallelism: one tool must have started BEFORE another finished.
    // i.e., at some point, there were two "started" without an intervening "ended".
    let mut overlapping = false;
    let mut active = 0;
    for (event, _) in log.iter() {
        if event.starts_with("start_") {
            active += 1;
            if active > 1 {
                overlapping = true;
            }
        } else if event.starts_with("end_") {
            active -= 1;
        }
    }
    
    assert!(overlapping, "Tools should have executed in parallel (overlapping intervals)");
    assert!(!final_answer.is_empty(), "Should have a final answer");
}
