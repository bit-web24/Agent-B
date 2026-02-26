use agentsm::AgentBuilder;
use agentsm::llm::MockLlmCaller;
use agentsm::types::{LlmResponse, ToolCall};
use agentsm::budget::TokenUsage;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_token_usage_accumulation() {
    let mock_responses = vec![
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: "dummy".to_string(),
                args: HashMap::new(),
                id: Some("call_1".to_string()),
            },
            confidence: 1.0,
            usage: Some(TokenUsage::new(10, 20)), // Total 30
        },
        LlmResponse::FinalAnswer {
            content: "Answer that is long enough to pass minimum length check.".to_string(),
            usage: Some(TokenUsage::new(5, 15)), // Total 20
        },
    ];

    let mut agent = AgentBuilder::new("Test usage")
        .llm(Arc::new(MockLlmCaller::new(mock_responses)))
        .tool("dummy", "desc", serde_json::json!({}), Arc::new(|_| Ok("res".to_string())))
        .build()
        .unwrap();

    let _answer = agent.run().await.unwrap();

    // Verify accumulation: 30 + 20 = 50
    assert_eq!(agent.memory.total_usage.input_tokens, 15);
    assert_eq!(agent.memory.total_usage.output_tokens, 35);
    assert_eq!(agent.memory.total_usage.total_tokens, 50);
}

#[tokio::test]
async fn test_budget_enforcement() {
    let mock_responses = vec![
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: "dummy".to_string(),
                args: HashMap::new(),
                id: Some("call_1".to_string()),
            },
            confidence: 1.0,
            usage: Some(TokenUsage::new(60, 0)), // 60 total
        },
        LlmResponse::FinalAnswer {
            content: "Should not be reached".to_string(),
            usage: None,
        },
    ];

    let mut agent = AgentBuilder::new("Test budget")
        .llm(Arc::new(MockLlmCaller::new(mock_responses)))
        .tool("dummy", "desc", serde_json::json!({}), Arc::new(|_| Ok("res".to_string())))
        .max_tokens(50) // Set budget to 50
        .build()
        .unwrap();

    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

    // Step 1: Idle -> Planning
    agent.step(&tx).await.unwrap();
    // Step 2: Planning (calls LLM, gets 60 tokens, budget 50 exceeded!) -> Error
    let result = agent.step(&tx).await;
    
    // In our implementation, we check budget BEFORE the call, 
    // and then update usage AFTER the call.
    // So the FIRST call (which uses 60) will SUCCEED in executing, 
    // but the NEXT call will be blocked.
    
    // Wait, let's verify where we check the budget.
    // planning.rs:
    // // 2. Guard: Token Budget
    // if let Some(budget) = memory.budget {
    //     if budget.is_exceeded(memory.total_usage) { ... return Event::fatal_error(); }
    // }
    
    // On the first Planning step, total_usage is 0. Budget 50 is NOT exceeded.
    // Planning calls LLM, receives 60 tokens. total_usage becomes 60.
    // Then it transitions to Acting -> Observing -> Planning.
    // On the SECOND Planning step, total_usage is 60. Budget 50 IS exceeded.
    
    assert!(result.is_ok()); // The step where it receives 60 succeeds
    
    // Acting
    agent.step(&tx).await.unwrap();
    // Observing
    agent.step(&tx).await.unwrap();
    
    // Next Planning should fail
    let final_step_result = agent.step(&tx).await;
    assert!(final_step_result.is_ok()); // step() returns Ok(Event) or Ok(()) usually
    
    assert_eq!(agent.current_state().as_str(), "Error");
    assert!(agent.memory.error.as_ref().unwrap().contains("budget exceeded"));
}
