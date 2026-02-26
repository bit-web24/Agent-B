use std::collections::HashMap;
use serde_json::Value;

use std::sync::Arc;

/// A tool function: takes JSON args, returns string result or error string.
/// Arc<dyn Fn> — shareable, Send + Sync for thread safety.
pub type ToolFn = Arc<dyn Fn(&HashMap<String, Value>) -> Result<String, String> + Send + Sync>;

/// Tool schema for sending to LLM (OpenAI / Anthropic tool format)
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolSchema {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,   // JSON Schema object
}

/// Registered tool entry
#[derive(Clone)]
struct ToolEntry {
    schema: ToolSchema,
    func:   ToolFn,
}

#[derive(Clone, Default)]
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

    /// Register a `Tool` built with the `Tool` builder — ergonomic shorthand.
    pub fn register_tool(&mut self, tool: Tool) {
        let (schema, func) = tool.into_parts();
        self.register(schema.name.clone(), schema.description.clone(), schema.input_schema, func);
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



// ─────────────────────────────────────────────────────────────────────────────
// Tool Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Parameter definition used by [`Tool`].
#[derive(Clone)]
struct ToolParam {
    name:        String,
    param_type:  String,
    description: String,
    required:    bool,
}

/// Ergonomic builder for constructing a tool definition.
///
/// Parameters are **required by default** — use `.param_opt()` for optional ones.
/// Call `.call()` last; it consumes the builder and produces the parts needed
/// by [`ToolRegistry::register`] or [`AgentBuilder::add_tool`].
///
/// # Example
/// ```no_run
/// # use agentsm::Tool;
/// # use std::collections::HashMap;
/// let tool = Tool::new("search", "Search the web for current information")
///     .param("query", "string", "The search query")
///     .param_opt("limit", "integer", "Maximum number of results (default: 5)")
///     .call(|args| {
///         let q = args["query"].as_str().unwrap_or("");
///         Ok(format!("Results for '{}'", q))
///     });
/// ```
pub struct Tool {
    name:        String,
    description: String,
    params:      Vec<ToolParam>,
    func:        Option<ToolFn>,
}

impl Tool {
    /// Start building a new tool.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name:        name.into(),
            description: description.into(),
            params:      Vec::new(),
            func:        None,
        }
    }

    /// Add a **required** parameter to this tool.
    ///
    /// `param_type` is a JSON Schema type string: `"string"`, `"integer"`,
    /// `"number"`, `"boolean"`, `"array"`, `"object"`.
    pub fn param(
        mut self,
        name:        impl Into<String>,
        param_type:  impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.params.push(ToolParam {
            name:        name.into(),
            param_type:  param_type.into(),
            description: description.into(),
            required:    true,
        });
        self
    }

    /// Add an **optional** parameter to this tool.
    pub fn param_opt(
        mut self,
        name:        impl Into<String>,
        param_type:  impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.params.push(ToolParam {
            name:        name.into(),
            param_type:  param_type.into(),
            description: description.into(),
            required:    false,
        });
        self
    }

    /// Attach the implementation function to this tool.
    ///
    /// This is the final step — it consumes the builder.
    pub fn call<F>(mut self, f: F) -> Self
    where
        F: Fn(&HashMap<String, Value>) -> Result<String, String> + Send + Sync + 'static,
    {
        self.func = Some(Arc::new(f));
        self
    }

    /// Build the JSON Schema and extract the (schema, fn) pair for registration.
    ///
    /// Panics if `.call()` was not invoked before this.
    pub(crate) fn into_parts(self) -> (ToolSchema, ToolFn) {
        let func = self.func
            .expect("Tool::call() must be called before registering the tool");

        let mut properties: HashMap<String, Value> = HashMap::new();
        let mut required:   Vec<Value>             = Vec::new();

        for p in &self.params {
            properties.insert(p.name.clone(), serde_json::json!({
                "type":        p.param_type,
                "description": p.description,
            }));
            if p.required {
                required.push(Value::String(p.name.clone()));
            }
        }

        let input_schema = serde_json::json!({
            "type":       "object",
            "properties": properties,
            "required":   required,
        });

        let schema = ToolSchema {
            name:         self.name,
            description:  self.description,
            input_schema,
        };

        (schema, func)
    }
}
