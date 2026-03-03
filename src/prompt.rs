//! Reusable prompt templates with variable substitution.
//!
//! Templates use `{variable}` syntax for interpolation.  Use `{{` and `}}`
//! for literal braces.

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Error)]
pub enum PromptError {
    #[error("Unresolved variable: {0}")]
    UnresolvedVariable(String),

    #[error("Invalid template syntax near position {0}: {1}")]
    InvalidSyntax(usize, String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Template
// ─────────────────────────────────────────────────────────────────────────────

/// A reusable system-prompt template with `{variable}` substitution.
///
/// # Syntax
/// - `{name}` — replaced by the value bound to `name`
/// - `{{`     — renders as a literal `{`
/// - `}}`     — renders as a literal `}`
///
/// # Modes
/// - **Strict** (`strict = true`): unresolved variables produce an error.
/// - **Lenient** (`strict = false`, default): unresolved variables are kept
///   as-is (e.g. `{missing}` stays `{missing}` in output).
///
/// # Example
/// ```
/// use agentsm::prompt::PromptTemplate;
///
/// let tpl = PromptTemplate::new("You are a {role}. Focus on {topic}.")
///     .var("role", "Rust expert")
///     .default_var("topic", "performance");
///
/// assert_eq!(
///     tpl.render().unwrap(),
///     "You are a Rust expert. Focus on performance."
/// );
/// ```
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: HashMap<String, String>,
    defaults: HashMap<String, String>,
    strict: bool,
}

impl PromptTemplate {
    /// Create a new template.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            variables: HashMap::new(),
            defaults: HashMap::new(),
            strict: false,
        }
    }

    /// Bind a variable value.  Last call wins for duplicate keys.
    pub fn var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(key.into(), value.into());
        self
    }

    /// Set a default that is used only when no explicit value is bound.
    pub fn default_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.defaults.insert(key.into(), value.into());
        self
    }

    /// When true, unresolved variables produce `Err(UnresolvedVariable)`.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Render the template using bound variables and defaults.
    pub fn render(&self) -> Result<String, PromptError> {
        self.render_with(&HashMap::new())
    }

    /// Render with additional runtime overrides (highest priority).
    ///
    /// Resolution order: `extra` → `variables` → `defaults` → error/keep.
    pub fn render_with(&self, extra: &HashMap<String, String>) -> Result<String, PromptError> {
        let src = self.template.as_bytes();
        let len = src.len();
        let mut out = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            match src[i] {
                b'{' => {
                    // Escaped brace: {{
                    if i + 1 < len && src[i + 1] == b'{' {
                        out.push('{');
                        i += 2;
                        continue;
                    }

                    // Find closing brace
                    let start = i + 1;
                    let mut end = start;
                    while end < len && src[end] != b'}' {
                        end += 1;
                    }
                    if end >= len {
                        return Err(PromptError::InvalidSyntax(
                            i,
                            "unclosed '{' — missing '}'".into(),
                        ));
                    }

                    // Extract and trim key
                    let key = std::str::from_utf8(&src[start..end]).unwrap_or("").trim();

                    if key.is_empty() {
                        return Err(PromptError::InvalidSyntax(
                            i,
                            "empty variable name in '{}'".into(),
                        ));
                    }

                    // Resolve: extra → variables → defaults
                    if let Some(val) = extra.get(key) {
                        out.push_str(val);
                    } else if let Some(val) = self.variables.get(key) {
                        out.push_str(val);
                    } else if let Some(val) = self.defaults.get(key) {
                        out.push_str(val);
                    } else if self.strict {
                        return Err(PromptError::UnresolvedVariable(key.to_string()));
                    } else {
                        // Lenient: keep as-is
                        out.push('{');
                        out.push_str(key);
                        out.push('}');
                    }

                    i = end + 1;
                }
                b'}' => {
                    // Escaped closing brace: }}
                    if i + 1 < len && src[i + 1] == b'}' {
                        out.push('}');
                        i += 2;
                    } else {
                        out.push('}');
                        i += 1;
                    }
                }
                _ => {
                    out.push(src[i] as char);
                    i += 1;
                }
            }
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_substitution() {
        let t = PromptTemplate::new("Hello {name}").var("name", "World");
        assert_eq!(t.render().unwrap(), "Hello World");
    }

    #[test]
    fn test_multiple_vars() {
        let t = PromptTemplate::new("You are a {role}. Focus on {topic}.")
            .var("role", "Rust expert")
            .var("topic", "memory safety");
        assert_eq!(
            t.render().unwrap(),
            "You are a Rust expert. Focus on memory safety."
        );
    }

    #[test]
    fn test_defaults() {
        let t = PromptTemplate::new("{role} working on {topic}")
            .var("role", "Engineer")
            .default_var("topic", "systems");
        assert_eq!(t.render().unwrap(), "Engineer working on systems");
    }

    #[test]
    fn test_extra_overrides() {
        let t = PromptTemplate::new("{role} does {task}")
            .var("role", "Alice")
            .var("task", "coding");
        let mut extra = HashMap::new();
        extra.insert("task".to_string(), "testing".to_string());
        assert_eq!(t.render_with(&extra).unwrap(), "Alice does testing");
    }

    #[test]
    fn test_escaped_braces() {
        let t = PromptTemplate::new("JSON: {{\"key\": \"{val}\"}}");
        let t = t.var("val", "42");
        assert_eq!(t.render().unwrap(), "JSON: {\"key\": \"42\"}");
    }

    #[test]
    fn test_lenient_unresolved() {
        let t = PromptTemplate::new("Hello {unknown}");
        assert_eq!(t.render().unwrap(), "Hello {unknown}");
    }

    #[test]
    fn test_strict_unresolved() {
        let t = PromptTemplate::new("Hello {missing}").strict(true);
        let err = t.render().unwrap_err();
        assert!(matches!(err, PromptError::UnresolvedVariable(ref s) if s == "missing"));
    }

    #[test]
    fn test_trimmed_keys() {
        let t = PromptTemplate::new("Hello { name }").var("name", "World");
        assert_eq!(t.render().unwrap(), "Hello World");
    }

    #[test]
    fn test_empty_value_allowed() {
        let t = PromptTemplate::new("prefix{val}suffix").var("val", "");
        assert_eq!(t.render().unwrap(), "prefixsuffix");
    }

    #[test]
    fn test_unclosed_brace_error() {
        let t = PromptTemplate::new("Hello {name");
        let err = t.render().unwrap_err();
        assert!(matches!(err, PromptError::InvalidSyntax(..)));
    }

    #[test]
    fn test_empty_variable_name_error() {
        let t = PromptTemplate::new("Hello {}");
        let err = t.render().unwrap_err();
        assert!(matches!(err, PromptError::InvalidSyntax(..)));
    }
}
