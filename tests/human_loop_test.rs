use agentsm::AgentBuilder;
use agentsm::llm::MockLlmCaller;
use agentsm::human::{ApprovalPolicy, RiskLevel, HumanDecision, HumanApprovalRequest};
use agentsm::types::{LlmResponse, ToolCall};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_human_approval_flow() {
    // 1. Define a high-risk tool
    let unsafe_tool = agentsm::Tool::new("delete_database", "Deletes all data")
        .call(|_| Ok("Database deleted".to_string()));

    // 2. Mock LLM that wants to call the unsafe tool
    let responses = vec![
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: "delete_database".to_string(),
                args: HashMap::new(),
                id: Some("call_1".to_string()),
            },
            confidence: 1.0,
            usage:      None,
        },
        LlmResponse::FinalAnswer {
            content: "I have deleted the database as requested.".to_string(),
            usage:   None,
        },
    ];
    let mock_llm = MockLlmCaller::new(responses);

    // 3. Setup approval policy: Require approval for "delete_database"
    let mut policy_map = HashMap::new();
    policy_map.insert("delete_database".to_string(), RiskLevel::Critical);
    let policy = ApprovalPolicy::ToolBased(policy_map);

    // 4. Build agent with HIP
    let mut agent = AgentBuilder::new("Delete the database")
        .llm(Arc::new(mock_llm))
        .add_tool(unsafe_tool)
        .approval_policy(policy)
        .on_approval(|req: HumanApprovalRequest| {
            println!("Human approving: {}", req.tool_name);
            HumanDecision::Approved
        })
        .build()
        .unwrap();

    // 5. Run agent
    let answer = agent.run().await.unwrap();
    assert_eq!(answer, "I have deleted the database as requested.");
    
    // Check history: should have 1 successful tool call
    assert_eq!(agent.memory.history.len(), 1);
    assert_eq!(agent.memory.history[0].tool.name, "delete_database");
    assert!(agent.memory.history[0].success);
}

#[tokio::test]
async fn test_human_rejection_flow() {
    let unsafe_tool = agentsm::Tool::new("delete_database", "Deletes all data")
        .call(|_| Ok("Database deleted".to_string()));

    let responses = vec![
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: "delete_database".to_string(),
                args: HashMap::new(),
                id: Some("call_1".to_string()),
            },
            confidence: 1.0,
            usage:      None,
        },
        LlmResponse::FinalAnswer {
            content: "I couldn't delete the database because you rejected it.".to_string(),
            usage:   None,
        },
    ];
    let mock_llm = MockLlmCaller::new(responses);

    let policy = ApprovalPolicy::AlwaysAsk;

    let mut agent = AgentBuilder::new("Delete the database")
        .llm(Arc::new(mock_llm))
        .add_tool(unsafe_tool)
        .approval_policy(policy)
        .on_approval(|req| {
            println!("Human rejecting: {}", req.tool_name);
            HumanDecision::Rejected("I don't trust you".to_string())
        })
        .build()
        .unwrap();

    let answer = agent.run().await.unwrap();
    assert!(answer.contains("rejected"));
    
    // Check history: should have 1 FAILED entry (rejected is failure)
    assert_eq!(agent.memory.history.len(), 1);
    assert!(!agent.memory.history[0].success);
    assert!(agent.memory.history[0].observation.contains("REJECTED"));
}

#[tokio::test]
async fn test_human_modification_flow() {
    let safe_tool = agentsm::Tool::new("list_files", "List files")
        .call(|_| Ok("file1, file2".to_string()));
    
    let responses = vec![
        LlmResponse::ToolCall {
            tool: ToolCall {
                name: "list_files".to_string(),
                args: HashMap::new(),
                id: Some("call_1".to_string()),
            },
            confidence: 1.0,
            usage:      None,
        },
        LlmResponse::FinalAnswer {
            content: "Here are the files...".to_string(),
            usage:   None,
        },
    ];
    let mock_llm = MockLlmCaller::new(responses);

    let policy = ApprovalPolicy::AlwaysAsk;

    let mut agent = AgentBuilder::new("List files")
        .llm(Arc::new(mock_llm))
        .add_tool(safe_tool)
        .approval_policy(policy)
        .on_approval(|_req| {
            // Human modifies the call to something else? 
            // Or just adds an argument.
            HumanDecision::Modified {
                tool_name: "list_files".to_string(),
                tool_args: {
                    let mut m = HashMap::new();
                    m.insert("dir".to_string(), serde_json::Value::String("/tmp".into()));
                    m
                }
            }
        })
        .build()
        .unwrap();

    let _answer = agent.run().await.unwrap();
    
    // Check history: the tool args should be modified
    assert_eq!(agent.memory.history.len(), 1);
    assert_eq!(agent.memory.history[0].tool.args.get("dir").unwrap().as_str().unwrap(), "/tmp");
}
