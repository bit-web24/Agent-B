use serde::{Serialize, Deserialize};
use crate::memory::AgentMemory;
use crate::types::State;
use async_trait::async_trait;
use std::collections::HashMap;

/// A point-in-time snapshot of the agent's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    pub checkpoint_id: String,
    pub session_id:    String,
    pub state:          State,
    pub memory:         AgentMemory, // Memory includes trace, history, and config
    pub timestamp:      chrono::DateTime<chrono::Utc>,
}

#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Save a checkpoint to the store.
    async fn save(&self, checkpoint: AgentCheckpoint) -> Result<(), String>;

    /// Load the latest checkpoint for a given session.
    async fn load_latest(&self, session_id: &str) -> Result<Option<AgentCheckpoint>, String>;

    /// Load a specific checkpoint by ID.
    async fn load_by_id(&self, checkpoint_id: &str) -> Result<Option<AgentCheckpoint>, String>;

    /// List all checkpoints for a session.
    async fn list_sessions(&self) -> Result<Vec<String>, String>;
}

/// A simple in-memory store for testing and short-lived sessions.
pub struct MemoryCheckpointStore {
    checkpoints: std::sync::Mutex<HashMap<String, Vec<AgentCheckpoint>>>, // session_id -> checkpoints
}

impl MemoryCheckpointStore {
    pub fn new() -> Self {
        Self {
            checkpoints: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl CheckpointStore for MemoryCheckpointStore {
    async fn save(&self, checkpoint: AgentCheckpoint) -> Result<(), String> {
        let mut store = self.checkpoints.lock().unwrap();
        store.entry(checkpoint.session_id.clone())
            .or_default()
            .push(checkpoint);
        Ok(())
    }

    async fn load_latest(&self, session_id: &str) -> Result<Option<AgentCheckpoint>, String> {
        let store = self.checkpoints.lock().unwrap();
        Ok(store.get(session_id).and_then(|v| v.last().cloned()))
    }

    async fn load_by_id(&self, checkpoint_id: &str) -> Result<Option<AgentCheckpoint>, String> {
        let store = self.checkpoints.lock().unwrap();
        for session_checkpoints in store.values() {
            if let Some(cp) = session_checkpoints.iter().find(|c| c.checkpoint_id == checkpoint_id) {
                return Ok(Some(cp.clone()));
            }
        }
        Ok(None)
    }

    async fn list_sessions(&self) -> Result<Vec<String>, String> {
        let store = self.checkpoints.lock().unwrap();
        Ok(store.keys().cloned().collect())
    }
}

/// A checkpoint store that saves each session to a separate JSON file in a directory.
pub struct FileCheckpointStore {
    base_path: std::path::PathBuf,
}

impl FileCheckpointStore {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        let path = path.into();
        let _ = std::fs::create_dir_all(&path);
        Self { base_path: path }
    }

    fn session_path(&self, session_id: &str) -> std::path::PathBuf {
        self.base_path.join(format!("{}.json", session_id))
    }
}

#[async_trait]
impl CheckpointStore for FileCheckpointStore {
    async fn save(&self, checkpoint: AgentCheckpoint) -> Result<(), String> {
        let path = self.session_path(&checkpoint.session_id);
        let mut checkpoints: Vec<AgentCheckpoint> = if path.exists() {
            let data = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            serde_json::from_str(&data).map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };
        checkpoints.push(checkpoint);
        let data = serde_json::to_string_pretty(&checkpoints).map_err(|e| e.to_string())?;
        std::fs::write(&path, data).map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn load_latest(&self, session_id: &str) -> Result<Option<AgentCheckpoint>, String> {
        let path = self.session_path(session_id);
        if !path.exists() { return Ok(None); }
        let data = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let checkpoints: Vec<AgentCheckpoint> = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        Ok(checkpoints.last().cloned())
    }

    async fn load_by_id(&self, checkpoint_id: &str) -> Result<Option<AgentCheckpoint>, String> {
        // This is inefficient for FileStore but satisfies the trait
        for entry in std::fs::read_dir(&self.base_path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let data = std::fs::read_to_string(entry.path()).map_err(|e| e.to_string())?;
            let checkpoints: Vec<AgentCheckpoint> = serde_json::from_str(&data).map_err(|e| e.to_string())?;
            if let Some(cp) = checkpoints.iter().find(|c| c.checkpoint_id == checkpoint_id) {
                return Ok(Some(cp.clone()));
            }
        }
        Ok(None)
    }

    async fn list_sessions(&self) -> Result<Vec<String>, String> {
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(&self.base_path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if let Some(stem) = entry.path().file_stem() {
                sessions.push(stem.to_string_lossy().to_string());
            }
        }
        Ok(sessions)
    }
}

/// A checkpoint store that uses a SQLite database.
pub struct SqliteCheckpointStore {
    path: std::path::PathBuf,
}

impl SqliteCheckpointStore {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Result<Self, String> {
        let path = path.into();
        let conn = rusqlite::Connection::open(&path).map_err(|e| e.to_string())?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                state         TEXT NOT NULL,
                memory        TEXT NOT NULL,
                timestamp     TEXT NOT NULL
            )",
            [],
        ).map_err(|e| e.to_string())?;
        Ok(Self { path })
    }

    fn get_conn(&self) -> Result<rusqlite::Connection, String> {
        rusqlite::Connection::open(&self.path).map_err(|e| e.to_string())
    }
}

#[async_trait]
impl CheckpointStore for SqliteCheckpointStore {
    async fn save(&self, checkpoint: AgentCheckpoint) -> Result<(), String> {
        let conn = self.get_conn()?;
        let memory_json = serde_json::to_string(&checkpoint.memory).map_err(|e| e.to_string())?;
        let state_json = serde_json::to_string(&checkpoint.state).map_err(|e| e.to_string())?;
        
        conn.execute(
            "INSERT INTO checkpoints (checkpoint_id, session_id, state, memory, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                checkpoint.checkpoint_id,
                checkpoint.session_id,
                state_json,
                memory_json,
                checkpoint.timestamp.to_rfc3339()
            ],
        ).map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn load_latest(&self, session_id: &str) -> Result<Option<AgentCheckpoint>, String> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT checkpoint_id, session_id, state, memory, timestamp 
             FROM checkpoints WHERE session_id = ?1 ORDER BY timestamp DESC LIMIT 1"
        ).map_err(|e| e.to_string())?;
        
        let mut rows = stmt.query(rusqlite::params![session_id]).map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            let memory_json: String = row.get(3).map_err(|e| e.to_string())?;
            let state_json: String = row.get(2).map_err(|e| e.to_string())?;
            let timestamp_str: String = row.get(4).map_err(|e| e.to_string())?;
            
            Ok(Some(AgentCheckpoint {
                checkpoint_id: row.get(0).map_err(|e| e.to_string())?,
                session_id:    row.get(1).map_err(|e| e.to_string())?,
                state:          serde_json::from_str(&state_json).map_err(|e| e.to_string())?,
                memory:         serde_json::from_str(&memory_json).map_err(|e| e.to_string())?,
                timestamp:      chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                                    .map_err(|e| e.to_string())?.with_timezone(&chrono::Utc),
            }))
        } else {
            Ok(None)
        }
    }

    async fn load_by_id(&self, checkpoint_id: &str) -> Result<Option<AgentCheckpoint>, String> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT checkpoint_id, session_id, state, memory, timestamp 
             FROM checkpoints WHERE checkpoint_id = ?1"
        ).map_err(|e| e.to_string())?;
        
        let mut rows = stmt.query(rusqlite::params![checkpoint_id]).map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            let memory_json: String = row.get(3).map_err(|e| e.to_string())?;
            let state_json: String = row.get(2).map_err(|e| e.to_string())?;
            let timestamp_str: String = row.get(4).map_err(|e| e.to_string())?;
            
            Ok(Some(AgentCheckpoint {
                checkpoint_id: row.get(0).map_err(|e| e.to_string())?,
                session_id:    row.get(1).map_err(|e| e.to_string())?,
                state:          serde_json::from_str(&state_json).map_err(|e| e.to_string())?,
                memory:         serde_json::from_str(&memory_json).map_err(|e| e.to_string())?,
                timestamp:      chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                                    .map_err(|e| e.to_string())?.with_timezone(&chrono::Utc),
            }))
        } else {
            Ok(None)
        }
    }

    async fn list_sessions(&self) -> Result<Vec<String>, String> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare("SELECT DISTINCT session_id FROM checkpoints").map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |row| row.get(0)).map_err(|e| e.to_string())?;
        let mut sessions = Vec::new();
        for session in rows {
            sessions.push(session.map_err(|e| e.to_string())?);
        }
        Ok(sessions)
    }
}
