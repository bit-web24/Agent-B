use agentsm::AgentBuilder;
use std::collections::HashMap;
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_integration() {
    // We need a tokio runtime for the bridge to work.
    // The AgentBuilder::mcp_server calls block_on/block_in_place.
    
    let builder = AgentBuilder::new("Test task")
        .mcp_server("python3", &["tests/mcp_server.py".to_string()]);
    
    let engine = builder.openai("sk-fake-key").build().unwrap();
    
    // Verify tool was registered
    assert!(engine.tools.has("echo"));
    
    // Execute the tool
    let mut args = HashMap::new();
    args.insert("message".to_string(), json!("Hello MCP"));
    
    let result = engine.tools.execute("echo", &args).unwrap();
    assert_eq!(result, "Echo: Hello MCP");
}
