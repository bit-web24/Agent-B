use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub step:      usize,
    pub state:     String,
    pub event:     String,
    pub data:      String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    entries: Vec<TraceEntry>,
}

impl Trace {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn record(&mut self, entry: TraceEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[TraceEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all entries for a given state name
    pub fn for_state(&self, state: &str) -> Vec<&TraceEntry> {
        self.entries.iter().filter(|e| e.state == state).collect()
    }

    /// Serializes the trace to a pretty-printed JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.entries)
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// Prints a human-readable trace table to stdout
    pub fn print(&self) {
        println!("\n{:<6} {:<14} {:<28} {}", "step", "state", "event", "data");
        println!("{}", "─".repeat(80));
        for e in &self.entries {
            println!("{:<6} {:<14} {:<28} {}", e.step, e.state, e.event, &e.data.chars().take(30).collect::<String>());
        }
    }
}
