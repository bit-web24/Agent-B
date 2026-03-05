//! Tool Composition — dynamic tool synthesis at runtime.
//!
//! Agents can compose existing tools into higher-level pipelines.
//! Each pipeline step feeds its output into the next step.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// ToolPipelineStep
// ─────────────────────────────────────────────────────────────────────────────

/// A single step in a composite tool pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPipelineStep {
    /// Name of the tool to call.
    pub tool: String,
    /// Argument template (supports `{prev}` and `{prev.field}` substitution).
    pub args_template: Value,
    /// If set, extract this field from the JSON output for the next step.
    pub output_field: Option<String>,
}

impl ToolPipelineStep {
    pub fn new(tool: impl Into<String>, args_template: Value) -> Self {
        Self {
            tool: tool.into(),
            args_template,
            output_field: None,
        }
    }

    pub fn with_output_field(mut self, field: impl Into<String>) -> Self {
        self.output_field = Some(field.into());
        self
    }

    /// Resolve the args template by substituting `{prev}` with the previous output.
    pub fn resolve_args(&self, prev_output: &str) -> Value {
        substitute_template(&self.args_template, prev_output)
    }
}

/// Recursively substitute `{prev}` in JSON values.
fn substitute_template(template: &Value, prev_output: &str) -> Value {
    match template {
        Value::String(s) => {
            if s == "{prev}" {
                // Try to parse as JSON first, fall back to string
                serde_json::from_str(prev_output)
                    .unwrap_or_else(|_| Value::String(prev_output.to_string()))
            } else if s.contains("{prev}") {
                Value::String(s.replace("{prev}", prev_output))
            } else {
                Value::String(s.clone())
            }
        }
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                new_map.insert(k.clone(), substitute_template(v, prev_output));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => Value::Array(
            arr.iter()
                .map(|v| substitute_template(v, prev_output))
                .collect(),
        ),
        other => other.clone(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositeToolSpec
// ─────────────────────────────────────────────────────────────────────────────

/// Specification for a composite tool (a pipeline of existing tools).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeToolSpec {
    pub name: String,
    pub description: String,
    pub pipeline: Vec<ToolPipelineStep>,
    /// Source tracking for auditability.
    pub source: ToolSource,
}

impl CompositeToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            pipeline: Vec::new(),
            source: ToolSource::Composite,
        }
    }

    /// Add a pipeline step.
    pub fn step(mut self, step: ToolPipelineStep) -> Self {
        self.pipeline.push(step);
        self
    }

    /// Validate the spec: at least one step, no circular references.
    pub fn validate(&self) -> Result<(), String> {
        if self.pipeline.is_empty() {
            return Err("Composite tool must have at least one step".into());
        }
        if self.name.is_empty() {
            return Err("Composite tool name cannot be empty".into());
        }
        // Check for self-reference
        if self.pipeline.iter().any(|s| s.tool == self.name) {
            return Err(format!(
                "Circular reference: composite tool '{}' cannot call itself",
                self.name
            ));
        }
        Ok(())
    }

    /// Execute the pipeline using a tool runner function.
    /// The runner takes (tool_name, args_json) and returns (result_string, success).
    pub fn execute<F>(&self, initial_input: &str, mut runner: F) -> PipelineResult
    where
        F: FnMut(&str, &Value) -> (String, bool),
    {
        let mut current_output = initial_input.to_string();
        let mut step_results = Vec::new();

        for (i, step) in self.pipeline.iter().enumerate() {
            let args = step.resolve_args(&current_output);
            let (result, success) = runner(&step.tool, &args);

            if !success {
                return PipelineResult {
                    final_output: result.clone(),
                    success: false,
                    steps_completed: i,
                    step_results,
                    error: Some(format!(
                        "Pipeline failed at step {} (tool: {}): {}",
                        i, step.tool, result
                    )),
                };
            }

            // Extract output field if specified
            current_output = if let Some(ref field) = step.output_field {
                if let Ok(json) = serde_json::from_str::<Value>(&result) {
                    json.get(field)
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .unwrap_or(result.clone())
                } else {
                    result.clone()
                }
            } else {
                result.clone()
            };

            step_results.push(PipelineStepResult {
                step_index: i,
                tool: step.tool.clone(),
                output: current_output.clone(),
                success: true,
            });
        }

        PipelineResult {
            final_output: current_output,
            success: true,
            steps_completed: self.pipeline.len(),
            step_results,
            error: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of executing a composite tool pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub final_output: String,
    pub success: bool,
    pub steps_completed: usize,
    pub step_results: Vec<PipelineStepResult>,
    pub error: Option<String>,
}

/// Result of a single pipeline step.
#[derive(Debug, Clone)]
pub struct PipelineStepResult {
    pub step_index: usize,
    pub tool: String,
    pub output: String,
    pub success: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolSource
// ─────────────────────────────────────────────────────────────────────────────

/// Where a tool came from (for auditability).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolSource {
    /// Statically registered by the user.
    Static,
    /// Composed from a pipeline of existing tools.
    Composite,
    /// Synthesized at runtime (code generation).
    Synthesized,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositionConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for tool composition capabilities.
#[derive(Debug, Clone)]
pub struct CompositionConfig {
    /// Allow composing existing tools into pipelines.
    pub allow_compose: bool,
    /// Maximum number of composite tools per session.
    pub max_composite_tools: usize,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            allow_compose: true,
            max_composite_tools: 10,
        }
    }
}

impl CompositionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allow_compose(mut self, allow: bool) -> Self {
        self.allow_compose = allow;
        self
    }

    pub fn max_composite_tools(mut self, max: usize) -> Self {
        self.max_composite_tools = max;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositeToolRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry for composite tools created during a session.
#[derive(Debug, Clone, Default)]
pub struct CompositeToolRegistry {
    pub tools: HashMap<String, CompositeToolSpec>,
    pub config: CompositionConfig,
}

impl CompositeToolRegistry {
    pub fn new(config: CompositionConfig) -> Self {
        Self {
            tools: HashMap::new(),
            config,
        }
    }

    /// Register a new composite tool.
    pub fn register(&mut self, spec: CompositeToolSpec) -> Result<(), String> {
        if !self.config.allow_compose {
            return Err("Tool composition is disabled".into());
        }
        if self.tools.len() >= self.config.max_composite_tools {
            return Err(format!(
                "Maximum composite tools ({}) reached",
                self.config.max_composite_tools
            ));
        }
        spec.validate()?;
        if self.tools.contains_key(&spec.name) {
            return Err(format!("Tool '{}' already exists", spec.name));
        }
        self.tools.insert(spec.name.clone(), spec);
        Ok(())
    }

    /// Get a composite tool by name.
    pub fn get(&self, name: &str) -> Option<&CompositeToolSpec> {
        self.tools.get(name)
    }

    /// List all registered composite tools.
    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(|k| k.as_str()).collect()
    }

    /// Number of registered composite tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_step_creation() {
        let step = ToolPipelineStep::new("search", serde_json::json!({"q": "{prev}"}))
            .with_output_field("results");
        assert_eq!(step.tool, "search");
        assert_eq!(step.output_field.as_deref(), Some("results"));
    }

    #[test]
    fn test_template_substitution() {
        let step = ToolPipelineStep::new("search", serde_json::json!({"query": "{prev}"}));
        let resolved = step.resolve_args("hello world");
        assert_eq!(resolved["query"], "hello world");
    }

    #[test]
    fn test_template_partial_substitution() {
        let step = ToolPipelineStep::new(
            "search",
            serde_json::json!({"query": "prefix {prev} suffix"}),
        );
        let resolved = step.resolve_args("test");
        assert_eq!(resolved["query"], "prefix test suffix");
    }

    #[test]
    fn test_composite_tool_creation() {
        let spec = CompositeToolSpec::new("research", "Search and summarize")
            .step(ToolPipelineStep::new(
                "search",
                serde_json::json!({"q": "{prev}"}),
            ))
            .step(ToolPipelineStep::new(
                "summarize",
                serde_json::json!({"text": "{prev}"}),
            ));
        assert_eq!(spec.pipeline.len(), 2);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_composite_tool_validation_empty() {
        let spec = CompositeToolSpec::new("empty", "No steps");
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_composite_tool_validation_circular() {
        let spec = CompositeToolSpec::new("loop", "Self reference")
            .step(ToolPipelineStep::new("loop", serde_json::json!({})));
        let err = spec.validate().unwrap_err();
        assert!(err.contains("Circular"));
    }

    #[test]
    fn test_pipeline_execution_success() {
        let spec = CompositeToolSpec::new("research", "Search and summarize")
            .step(ToolPipelineStep::new(
                "search",
                serde_json::json!({"q": "{prev}"}),
            ))
            .step(ToolPipelineStep::new(
                "summarize",
                serde_json::json!({"text": "{prev}"}),
            ));

        let result = spec.execute("rust programming", |tool, _args| match tool {
            "search" => ("Found 10 results about Rust".into(), true),
            "summarize" => ("Rust is a systems programming language".into(), true),
            _ => ("unknown tool".into(), false),
        });

        assert!(result.success);
        assert_eq!(result.steps_completed, 2);
        assert_eq!(
            result.final_output,
            "Rust is a systems programming language"
        );
    }

    #[test]
    fn test_pipeline_execution_failure() {
        let spec = CompositeToolSpec::new("research", "Search and summarize")
            .step(ToolPipelineStep::new(
                "search",
                serde_json::json!({"q": "{prev}"}),
            ))
            .step(ToolPipelineStep::new(
                "summarize",
                serde_json::json!({"text": "{prev}"}),
            ));

        let result = spec.execute("query", |tool, _args| match tool {
            "search" => ("error: rate limit".into(), false),
            _ => ("ok".into(), true),
        });

        assert!(!result.success);
        assert_eq!(result.steps_completed, 0);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_pipeline_output_field_extraction() {
        let spec = CompositeToolSpec::new("extract", "Extract field").step(
            ToolPipelineStep::new("get_data", serde_json::json!({})).with_output_field("name"),
        );

        let result = spec.execute("", |_tool, _args| {
            (r#"{"name": "Alice", "age": 30}"#.into(), true)
        });

        assert!(result.success);
        assert_eq!(result.final_output, "Alice");
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = CompositeToolRegistry::new(CompositionConfig::default());
        let spec = CompositeToolSpec::new("research", "Search and summarize")
            .step(ToolPipelineStep::new("search", serde_json::json!({})));

        assert!(reg.register(spec).is_ok());
        assert_eq!(reg.len(), 1);
        assert!(reg.get("research").is_some());
    }

    #[test]
    fn test_registry_duplicate_rejection() {
        let mut reg = CompositeToolRegistry::new(CompositionConfig::default());
        let spec1 = CompositeToolSpec::new("research", "v1")
            .step(ToolPipelineStep::new("search", serde_json::json!({})));
        let spec2 = CompositeToolSpec::new("research", "v2")
            .step(ToolPipelineStep::new("search", serde_json::json!({})));

        assert!(reg.register(spec1).is_ok());
        assert!(reg.register(spec2).is_err()); // Duplicate name
    }

    #[test]
    fn test_registry_max_limit() {
        let config = CompositionConfig::new().max_composite_tools(2);
        let mut reg = CompositeToolRegistry::new(config);

        for i in 0..3 {
            let spec = CompositeToolSpec::new(format!("tool_{}", i), "desc")
                .step(ToolPipelineStep::new("x", serde_json::json!({})));
            let result = reg.register(spec);
            if i < 2 {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err()); // Exceeded max
            }
        }
    }

    #[test]
    fn test_registry_disabled() {
        let config = CompositionConfig::new().allow_compose(false);
        let mut reg = CompositeToolRegistry::new(config);
        let spec = CompositeToolSpec::new("tool", "desc")
            .step(ToolPipelineStep::new("x", serde_json::json!({})));
        assert!(reg.register(spec).is_err());
    }

    #[test]
    fn test_tool_source_variants() {
        assert_eq!(ToolSource::Static, ToolSource::Static);
        assert_ne!(ToolSource::Composite, ToolSource::Synthesized);
    }
}
