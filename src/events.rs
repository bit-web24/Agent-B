use serde::{Deserialize, Serialize};

/// A named event emitted by a state handler to drive transitions.
///
/// Events are identified by their string name. The library ships with
/// well-known constants for all built-in events, but users can define
/// custom events for their own state graphs.
///
/// # Defining a Custom Event
///
/// ```
/// use agentsm::Event;
/// let evt = Event::new("NeedsResearch");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Event(pub String);

impl Event {
    /// Create a new event with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the string name of this event.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    // ── Well-known built-in event constructors ──────────────────────────

    // Lifecycle
    pub fn start()           -> Self { Self::new("Start") }

    // Planning outcomes
    pub fn llm_tool_call()   -> Self { Self::new("LlmToolCall") }
    pub fn llm_final_answer()-> Self { Self::new("LlmFinalAnswer") }
    pub fn max_steps()       -> Self { Self::new("MaxSteps") }
    pub fn low_confidence()  -> Self { Self::new("LowConfidence") }
    pub fn answer_too_short()-> Self { Self::new("AnswerTooShort") }
    pub fn tool_blacklisted()-> Self { Self::new("ToolBlacklisted") }
    pub fn fatal_error()     -> Self { Self::new("FatalError") }

    // Acting outcomes
    pub fn tool_success()    -> Self { Self::new("ToolSuccess") }
    pub fn tool_failure()    -> Self { Self::new("ToolFailure") }

    // Observing outcomes
    pub fn r#continue()      -> Self { Self::new("Continue") }
    pub fn needs_reflection()-> Self { Self::new("NeedsReflection") }

    // Reflecting outcomes
    pub fn reflect_done()    -> Self { Self::new("ReflectDone") }
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
