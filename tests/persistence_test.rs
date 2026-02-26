use agentsm::AgentBuilder;
use agentsm::llm::MockLlmCaller;
use agentsm::types::{LlmResponse, ToolCall};
use agentsm::checkpoint::{MemoryCheckpointStore, FileCheckpointStore, SqliteCheckpointStore, CheckpointStore, AgentCheckpoint};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn test_persistence_memory_store() {
    let store = Arc::new(MemoryCheckpointStore::new());
    let session_id = "test_session_1";

    // 1. First run: stops after one tool call
    {
        let mock_llm = vec![
            LlmResponse::ToolCall {
                tool: ToolCall {
                    name: "test_tool".to_string(),
                    args: HashMap::new(),
                    id: Some("call_1".to_string()),
                },
                confidence: 1.0,
                usage:      None,
            },
        ];
        
        let tool = agentsm::Tool::new("test_tool", "desc")
            .call(|_| Ok("result 1".to_string()));

        let mut agent = AgentBuilder::new("Task 1")
            .llm(Arc::new(MockLlmCaller::new(mock_llm)))
            .add_tool(tool)
            .checkpoint_store(store.clone())
            .session_id(session_id)
            .build()
            .unwrap();

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Step 1: Idle -> Planning
        agent.step(&tx).await.unwrap();
        // Step 2: Planning -> Acting
        agent.step(&tx).await.unwrap();
        // Step 3: Acting -> Observing
        agent.step(&tx).await.unwrap();
        // Step 4: Observing -> Planning
        agent.step(&tx).await.unwrap();
        
        assert_eq!(agent.current_state().as_str(), "Planning");
    }

    // 2. Resume run: verify history is preserved
    {
        let responses = vec![
            LlmResponse::FinalAnswer {
                content: "resumed answer that is long enough".to_string(),
                usage:   None,
            },
        ];
        
        let tool = agentsm::Tool::new("test_tool", "desc")
            .call(|_| Ok("result 1".to_string()));

        let mut agent = AgentBuilder::new("Dummy Task")
            .llm(Arc::new(MockLlmCaller::new(responses)))
            .add_tool(tool)
            .checkpoint_store(store.clone())
            .resume(session_id).await.unwrap()
            .build()
            .unwrap();

        println!("Resumed state: {}", agent.current_state());
        assert_eq!(agent.memory.task, "Task 1");
        assert_eq!(agent.memory.history.len(), 1);
        assert!(agent.memory.history[0].observation.contains("result 1"));

        let answer = agent.run().await.unwrap();
        assert_eq!(answer, "resumed answer that is long enough");
    }
}

#[tokio::test]
async fn test_persistence_file_store() {
    let temp_dir = TempDir::new().unwrap();
    let store = Arc::new(FileCheckpointStore::new(temp_dir.path()));
    let session_id = "test_session_file";

    // 1. First run
    {
        let mock_llm = vec![LlmResponse::FinalAnswer { content: "ok enough length".to_string(), usage: None }];
        let mut agent = AgentBuilder::new("Task File")
            .llm(Arc::new(MockLlmCaller::new(mock_llm)))
            .checkpoint_store(store.clone())
            .session_id(session_id)
            .build()
            .unwrap();
        agent.run().await.unwrap();
    }

    // 2. Verify we can load it
    let checkpoint: AgentCheckpoint = store.load_latest(session_id).await.unwrap().unwrap();
    assert_eq!(checkpoint.memory.task, "Task File");
    assert_eq!(checkpoint.state.as_str(), "Done");
}

#[tokio::test]
async fn test_persistence_sqlite_store() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let store = Arc::new(SqliteCheckpointStore::new(db_path).unwrap());
    let session_id = "test_session_sqlite";

    // 1. First run
    {
        let mock_llm = vec![LlmResponse::FinalAnswer { content: "ok enough length".to_string(), usage: None }];
        let mut agent = AgentBuilder::new("Task Sqlite")
            .llm(Arc::new(MockLlmCaller::new(mock_llm)))
            .checkpoint_store(store.clone())
            .session_id(session_id)
            .build()
            .unwrap();
        agent.run().await.unwrap();
    }

    // 2. Verify we can load it
    let checkpoint: AgentCheckpoint = store.load_latest(session_id).await.unwrap().unwrap();
    assert_eq!(checkpoint.memory.task, "Task Sqlite");
    assert_eq!(checkpoint.state.as_str(), "Done");
}
