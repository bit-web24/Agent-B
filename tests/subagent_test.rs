use agentsm::AgentBuilder;
use agentsm::llm::MockLlmCaller;
use agentsm::types::{LlmResponse, ToolCall};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test(flavor = "multi_thread")]
async fn test_subagent_delegation() {
    // 1. Setup a specialized calculator sub-agent
    // This sub-agent knows how to add numbers.
    let calc_responses = vec![
        LlmResponse::FinalAnswer {
            content: "The result is 42".to_string(),
            usage: None,
        },
    ];
    let calc_agent = AgentBuilder::new("Calculate sum")
        .llm(Arc::new(MockLlmCaller::new(calc_responses)));

    // 2. Setup a parent agent that uses the calculator sub-agent as a tool
    let parent_responses = vec![
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: "calculator".to_string(),
                args: {
                    let mut m = HashMap::new();
                    m.insert("task".to_string(), serde_json::Value::String("add 20 and 22".into()));
                    m
                },
                id: Some("call_calc_1".to_string()),
            },
            confidence: 1.0,
            usage: None,
        },
        LlmResponse::FinalAnswer {
            content: "The calculator said it's 42".to_string(),
            usage: None,
        },
    ];

    let mut parent = AgentBuilder::new("Ask calculator for a sum")
        .llm(Arc::new(MockLlmCaller::new(parent_responses)))
        .add_subagent("calculator", "A specialized calculator agent", calc_agent)
        .build()
        .unwrap();

    // 3. Run parent
    let answer = parent.run().await.unwrap();

    assert_eq!(answer, "The calculator said it's 42");
    
    // 4. Verify trace/history
    assert_eq!(parent.memory.history.len(), 1);
    assert_eq!(parent.memory.history[0].tool.name, "calculator");
    assert!(parent.memory.history[0].observation.contains("42"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_nested_subagent_delegation() {
    // Grandchild: just answers
    let grandchild_llm = Arc::new(MockLlmCaller::new(vec![
        LlmResponse::FinalAnswer { content: "I am the grandchild".to_string(), usage: None }
    ]));
    let grandchild = AgentBuilder::new("gc").llm(grandchild_llm);

    // Child: calls grandchild
    let child_llm = Arc::new(MockLlmCaller::new(vec![
        LlmResponse::ToolCall {
            tool: ToolCall { name: "grandchild".to_string(), args: HashMap::new(), id: Some("c1".to_string()) },
            confidence: 1.0,
            usage: None,
        },
        LlmResponse::FinalAnswer { content: "Grandchild said: ...".to_string(), usage: None }
    ]));
    let child = AgentBuilder::new("c")
        .llm(child_llm)
        .add_subagent("grandchild", "desc", grandchild);

    // Parent: calls child
    let parent_llm = Arc::new(MockLlmCaller::new(vec![
        LlmResponse::ToolCall {
            tool: ToolCall { name: "child".to_string(), args: HashMap::new(), id: Some("p1".to_string()) },
            confidence: 1.0,
            usage: None,
        },
        LlmResponse::FinalAnswer { content: "Child finished".to_string(), usage: None }
    ]));
    
    let mut parent = AgentBuilder::new("p")
        .llm(parent_llm)
        .add_subagent("child", "desc", child)
        .build()
        .unwrap();

    let answer = parent.run().await.unwrap();
    assert_eq!(answer, "Child finished");
}
