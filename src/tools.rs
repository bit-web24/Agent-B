use std::collections::HashMap;
use serde_json::Value;

/// A tool function: takes JSON args, returns string result or error string.
/// Box<dyn Fn> — heap-allocated, Send + Sync for thread safety.
pub type ToolFn = Box<dyn Fn(&HashMap<String, Value>) -> Result<String, String> + Send + Sync>;

/// Tool schema for sending to LLM (OpenAI / Anthropic tool format)
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolSchema {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,   // JSON Schema object
}

/// Registered tool entry
struct ToolEntry {
    schema: ToolSchema,
    func:   ToolFn,
}

pub struct ToolRegistry {
    tools: HashMap<String, ToolEntry>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    /// Register a tool with its schema and implementation.
    ///
    /// # Arguments
    /// * `name`        - Unique tool name (must match schema name)
    /// * `description` - Clear description of what this tool does and when to use it
    /// * `schema`      - JSON Schema for the input parameters
    /// * `func`        - The actual implementation
    pub fn register(
        &mut self,
        name:        impl Into<String>,
        description: impl Into<String>,
        schema:      Value,
        func:        ToolFn,
    ) {
        let name = name.into();
        self.tools.insert(name.clone(), ToolEntry {
            schema: ToolSchema {
                name:         name.clone(),
                description:  description.into(),
                input_schema: schema,
            },
            func,
        });
    }

    /// Execute a named tool with given arguments.
    /// Returns Ok(result_string) or Err(error_string).
    /// Never panics — all errors are captured as Err variants.
    pub fn execute(&self, name: &str, args: &HashMap<String, Value>) -> Result<String, String> {
        match self.tools.get(name) {
            Some(entry) => (entry.func)(args),
            None        => Err(format!("Tool '{}' not found in registry", name)),
        }
    }

    /// Returns true if a tool with this name is registered.
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns all tool schemas — used to build the tools array for LLM calls.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|e| e.schema.clone()).collect()
    }

    /// Returns the count of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}
