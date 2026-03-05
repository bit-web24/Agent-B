//! Deterministic Replay — time-travel debugging with patches.
//!
//! Record every LLM call, tool call, and state transition. Replay from
//! any step with optional patches to explore alternative outcomes.

use crate::types::LlmResponse;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// ReplayEntry
// ─────────────────────────────────────────────────────────────────────────────

/// What kind of event was recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplayEntryKind {
    /// An LLM call and its response.
    LlmCall {
        model: String,
        response: LlmResponse,
    },
    /// A tool execution and its result.
    ToolCall {
        name: String,
        args: Value,
        result: String,
        success: bool,
    },
    /// A state machine transition.
    StateTransition {
        from: String,
        event: String,
        to: String,
    },
}

/// A single recorded event with step and timing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEntry {
    pub step: usize,
    pub state: String,
    pub kind: ReplayEntryKind,
    /// SHA-256 hash of inputs for determinism validation.
    pub input_hash: String,
    pub timestamp: DateTime<Utc>,
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplayRecorder
// ─────────────────────────────────────────────────────────────────────────────

/// Records events during an agent run for later replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRecorder {
    pub session_id: String,
    pub entries: Vec<ReplayEntry>,
    pub recording: bool,
}

impl ReplayRecorder {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            entries: Vec::new(),
            recording: true,
        }
    }

    /// Create a disabled recorder (no-op).
    pub fn disabled() -> Self {
        Self {
            session_id: String::new(),
            entries: Vec::new(),
            recording: false,
        }
    }

    /// Record an LLM call.
    pub fn record_llm_call(
        &mut self,
        step: usize,
        state: &str,
        model: &str,
        response: &LlmResponse,
    ) {
        if !self.recording {
            return;
        }
        let input_hash = Self::hash_inputs(&format!("llm:{}:{}:{}", step, state, model));
        self.entries.push(ReplayEntry {
            step,
            state: state.to_string(),
            kind: ReplayEntryKind::LlmCall {
                model: model.to_string(),
                response: response.clone(),
            },
            input_hash,
            timestamp: Utc::now(),
        });
    }

    /// Record a tool call.
    pub fn record_tool_call(
        &mut self,
        step: usize,
        state: &str,
        name: &str,
        args: &Value,
        result: &str,
        success: bool,
    ) {
        if !self.recording {
            return;
        }
        let input_hash = Self::hash_inputs(&format!("tool:{}:{}:{}:{}", step, state, name, args));
        self.entries.push(ReplayEntry {
            step,
            state: state.to_string(),
            kind: ReplayEntryKind::ToolCall {
                name: name.to_string(),
                args: args.clone(),
                result: result.to_string(),
                success,
            },
            input_hash,
            timestamp: Utc::now(),
        });
    }

    /// Record a state transition.
    pub fn record_transition(&mut self, step: usize, from: &str, event: &str, to: &str) {
        if !self.recording {
            return;
        }
        let input_hash =
            Self::hash_inputs(&format!("transition:{}:{}:{}:{}", step, from, event, to));
        self.entries.push(ReplayEntry {
            step,
            state: from.to_string(),
            kind: ReplayEntryKind::StateTransition {
                from: from.to_string(),
                event: event.to_string(),
                to: to.to_string(),
            },
            input_hash,
            timestamp: Utc::now(),
        });
    }

    /// Get entries for a specific step.
    pub fn entries_at_step(&self, step: usize) -> Vec<&ReplayEntry> {
        self.entries.iter().filter(|e| e.step == step).collect()
    }

    /// Get the LLM response recorded at a given step (if any).
    pub fn llm_response_at(&self, step: usize) -> Option<&LlmResponse> {
        self.entries.iter().find_map(|e| {
            if e.step == step {
                if let ReplayEntryKind::LlmCall { ref response, .. } = e.kind {
                    return Some(response);
                }
            }
            None
        })
    }

    /// Get the tool result recorded at a given step (if any).
    pub fn tool_result_at(&self, step: usize) -> Option<(&str, &str, bool)> {
        self.entries.iter().find_map(|e| {
            if e.step == step {
                if let ReplayEntryKind::ToolCall {
                    ref name,
                    ref result,
                    success,
                    ..
                } = e.kind
                {
                    return Some((name.as_str(), result.as_str(), success));
                }
            }
            None
        })
    }

    /// Total number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Highest step number recorded.
    pub fn max_step(&self) -> usize {
        self.entries.iter().map(|e| e.step).max().unwrap_or(0)
    }

    /// Serialize entries to NDJSON string.
    pub fn to_ndjson(&self) -> String {
        self.entries
            .iter()
            .filter_map(|e| serde_json::to_string(e).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Deserialize entries from NDJSON string.
    pub fn from_ndjson(session_id: &str, ndjson: &str) -> Self {
        let entries = ndjson
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        Self {
            session_id: session_id.to_string(),
            entries,
            recording: false,
        }
    }

    fn hash_inputs(input: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Patch
// ─────────────────────────────────────────────────────────────────────────────

/// A patch that overrides a recorded event at a specific step.
#[derive(Debug, Clone)]
pub enum Patch {
    /// Override a tool result at this step.
    ToolResult {
        tool_name: String,
        new_result: String,
        success: bool,
    },
    /// Override the LLM response at this step.
    LlmResponse(LlmResponse),
    /// Skip this step entirely.
    Skip,
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplayEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Replays a recorded session with optional patches.
#[derive(Debug, Clone)]
pub struct ReplayEngine {
    pub recorder: ReplayRecorder,
    pub patches: HashMap<usize, Patch>,
}

impl ReplayEngine {
    /// Create from an existing recording.
    pub fn from_recorder(recorder: ReplayRecorder) -> Self {
        Self {
            recorder,
            patches: HashMap::new(),
        }
    }

    /// Add a patch at a specific step.
    pub fn patch_step(mut self, step: usize, patch: Patch) -> Self {
        self.patches.insert(step, patch);
        self
    }

    /// Check if a step has a patch.
    pub fn has_patch(&self, step: usize) -> bool {
        self.patches.contains_key(&step)
    }

    /// Get the patch for a step (if any).
    pub fn get_patch(&self, step: usize) -> Option<&Patch> {
        self.patches.get(&step)
    }

    /// Get the patched or original LLM response at a step.
    pub fn resolve_llm_response(&self, step: usize) -> Option<LlmResponse> {
        // Check patches first
        if let Some(Patch::LlmResponse(resp)) = self.patches.get(&step) {
            return Some(resp.clone());
        }
        // Fall back to recorded
        self.recorder.llm_response_at(step).cloned()
    }

    /// Get the patched or original tool result at a step.
    pub fn resolve_tool_result(&self, step: usize) -> Option<(String, String, bool)> {
        // Check patches first
        if let Some(Patch::ToolResult {
            tool_name,
            new_result,
            success,
        }) = self.patches.get(&step)
        {
            return Some((tool_name.clone(), new_result.clone(), *success));
        }
        // Fall back to recorded
        self.recorder
            .tool_result_at(step)
            .map(|(n, r, s)| (n.to_string(), r.to_string(), s))
    }

    /// Compare two replay engines (diff).
    pub fn diff(&self, other: &ReplayEngine) -> Vec<ReplayDiffEntry> {
        let max_step = self.recorder.max_step().max(other.recorder.max_step());
        let mut diffs = Vec::new();

        for step in 0..=max_step {
            let a_entries = self.recorder.entries_at_step(step);
            let b_entries = other.recorder.entries_at_step(step);

            if a_entries.len() != b_entries.len() {
                diffs.push(ReplayDiffEntry {
                    step,
                    kind: DiffKind::EntryCountMismatch {
                        left: a_entries.len(),
                        right: b_entries.len(),
                    },
                });
            } else {
                for (a, b) in a_entries.iter().zip(b_entries.iter()) {
                    if a.input_hash != b.input_hash {
                        diffs.push(ReplayDiffEntry {
                            step,
                            kind: DiffKind::InputHashMismatch {
                                left: a.input_hash.clone(),
                                right: b.input_hash.clone(),
                            },
                        });
                    }
                }
            }
        }
        diffs
    }
}

/// A single difference between two replay sessions.
#[derive(Debug, Clone)]
pub struct ReplayDiffEntry {
    pub step: usize,
    pub kind: DiffKind,
}

/// The kind of difference detected.
#[derive(Debug, Clone)]
pub enum DiffKind {
    EntryCountMismatch { left: usize, right: usize },
    InputHashMismatch { left: String, right: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplayRecording mode
// ─────────────────────────────────────────────────────────────────────────────

/// Controls what the recorder captures.
#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum ReplayRecording {
    /// No recording.
    #[default]
    Off,
    /// Record all LLM + tool I/O.
    Full,
    /// Record only state transitions (lightweight).
    TransitionsOnly,
}


// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LlmResponse;

    #[test]
    fn test_recorder_creation() {
        let rec = ReplayRecorder::new("session-1");
        assert_eq!(rec.session_id, "session-1");
        assert!(rec.is_empty());
        assert!(rec.recording);
    }

    #[test]
    fn test_recorder_disabled() {
        let mut rec = ReplayRecorder::disabled();
        rec.record_llm_call(
            1,
            "Planning",
            "gpt-4o",
            &LlmResponse::FinalAnswer {
                content: "test".into(),
                usage: None,
            },
        );
        assert!(rec.is_empty()); // Nothing recorded
    }

    #[test]
    fn test_record_llm_call() {
        let mut rec = ReplayRecorder::new("s1");
        let resp = LlmResponse::FinalAnswer {
            content: "hello".into(),
            usage: None,
        };
        rec.record_llm_call(1, "Planning", "gpt-4o", &resp);
        assert_eq!(rec.len(), 1);
        assert!(rec.llm_response_at(1).is_some());
        assert!(rec.llm_response_at(2).is_none());
    }

    #[test]
    fn test_record_tool_call() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_tool_call(
            2,
            "Acting",
            "search",
            &serde_json::json!({"q": "test"}),
            "found it",
            true,
        );
        assert_eq!(rec.len(), 1);
        let (name, result, success) = rec.tool_result_at(2).unwrap();
        assert_eq!(name, "search");
        assert_eq!(result, "found it");
        assert!(success);
    }

    #[test]
    fn test_record_transition() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_transition(1, "Planning", "llm_tool_call", "Acting");
        assert_eq!(rec.len(), 1);
        let entries = rec.entries_at_step(1);
        assert_eq!(entries.len(), 1);
        assert!(matches!(
            entries[0].kind,
            ReplayEntryKind::StateTransition { .. }
        ));
    }

    #[test]
    fn test_max_step() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_transition(3, "A", "e", "B");
        rec.record_transition(7, "B", "e", "C");
        rec.record_transition(1, "X", "e", "Y");
        assert_eq!(rec.max_step(), 7);
    }

    #[test]
    fn test_ndjson_roundtrip() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_transition(1, "Idle", "start", "Planning");
        rec.record_llm_call(
            2,
            "Planning",
            "gpt-4o",
            &LlmResponse::FinalAnswer {
                content: "done".into(),
                usage: None,
            },
        );

        let ndjson = rec.to_ndjson();
        let loaded = ReplayRecorder::from_ndjson("s1", &ndjson);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.max_step(), 2);
    }

    #[test]
    fn test_replay_engine_patch_llm() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_llm_call(
            1,
            "Planning",
            "gpt-4o",
            &LlmResponse::FinalAnswer {
                content: "original".into(),
                usage: None,
            },
        );

        let engine = ReplayEngine::from_recorder(rec).patch_step(
            1,
            Patch::LlmResponse(LlmResponse::FinalAnswer {
                content: "patched".into(),
                usage: None,
            }),
        );

        let resp = engine.resolve_llm_response(1).unwrap();
        match resp {
            LlmResponse::FinalAnswer { content, .. } => assert_eq!(content, "patched"),
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn test_replay_engine_patch_tool() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_tool_call(
            2,
            "Acting",
            "search",
            &serde_json::json!({}),
            "original result",
            true,
        );

        let engine = ReplayEngine::from_recorder(rec).patch_step(
            2,
            Patch::ToolResult {
                tool_name: "search".into(),
                new_result: "patched result".into(),
                success: false,
            },
        );

        let (name, result, success) = engine.resolve_tool_result(2).unwrap();
        assert_eq!(name, "search");
        assert_eq!(result, "patched result");
        assert!(!success);
    }

    #[test]
    fn test_replay_engine_no_patch_fallback() {
        let mut rec = ReplayRecorder::new("s1");
        rec.record_llm_call(
            1,
            "Planning",
            "gpt-4o",
            &LlmResponse::FinalAnswer {
                content: "original".into(),
                usage: None,
            },
        );

        let engine = ReplayEngine::from_recorder(rec);
        let resp = engine.resolve_llm_response(1).unwrap();
        match resp {
            LlmResponse::FinalAnswer { content, .. } => assert_eq!(content, "original"),
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn test_replay_diff() {
        let mut rec1 = ReplayRecorder::new("s1");
        rec1.record_transition(1, "A", "e", "B");

        let mut rec2 = ReplayRecorder::new("s2");
        rec2.record_transition(1, "A", "e", "B");
        rec2.record_transition(2, "B", "e", "C"); // Extra entry not in rec1

        let engine1 = ReplayEngine::from_recorder(rec1);
        let engine2 = ReplayEngine::from_recorder(rec2);

        let diffs = engine1.diff(&engine2);
        assert!(!diffs.is_empty());
        // Step 2 has count mismatch
        assert!(diffs
            .iter()
            .any(|d| d.step == 2
                && matches!(d.kind, DiffKind::EntryCountMismatch { left: 0, right: 1 })));
    }
}
