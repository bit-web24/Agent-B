use agentsm::AgentBuilder;
use agentsm::llm::MockLlmCaller;
use agentsm::types::{LlmResponse, ToolCall};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_parallel_tool_execution() {
    let mock_responses = vec![
        LlmResponse::ParallelToolCalls {
            tools: vec![
                ToolCall {
                    name: "tool_a".to_string(),
                    args: HashMap::new(),
                    id: Some("id_a".to_string()),
                },
                ToolCall {
                    name: "tool_b".to_string(),
                    args: HashMap::new(),
                    id: Some("id_b".to_string()),
                },
            ],
            confidence: 1.0,
            usage: None,
        },
        LlmResponse::FinalAnswer {
            content: "Both tools finished.".to_string(),
            usage: None,
        },
    ];

    let mut agent = AgentBuilder::new("Run two tools")
        .llm(Arc::new(MockLlmCaller::new(mock_responses)))
        .tool("tool_a", "desc", serde_json::json!({}), Arc::new(|_| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok("A done".to_string())
        }))
        .tool("tool_b", "desc", serde_json::json!({}), Arc::new(|_| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok("B done".to_string())
        }))
        .parallel_tools(true)
        .build()
        .unwrap();

    let start = std::time::Instant::now();
    let answer = agent.run().await.unwrap();
    let duration = start.elapsed();

    assert_eq!(answer, "Both tools finished.");
    // Parallel should take ~100ms, sequential would take ~200ms
    assert!(duration.as_millis() < 180, "Parallel execution seems too slow: {}ms", duration.as_millis());
}
