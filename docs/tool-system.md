# Tool System

Tools are the agent's interface to the external world. Every capability the agent has beyond language — web search, math, database queries, API calls — must be expressed as a registered tool.

---

## Tool Builder (Recommended)

The `Tool` builder provides a clean, schema-free API:

```rust
use agentsm::{AgentBuilder, Tool};

AgentBuilder::new("task")
    .openai("sk-...")
    .model("gpt-4o")
    .add_tool(
        Tool::new("search", "Search the web for current information.")
            .param("query", "string", "The search query")           // required
            .param_opt("limit", "integer", "Max results (default 5)") // optional
            .call(|args| {
                let q = args["query"].as_str().unwrap_or("");
                Ok(format!("Results for '{}': ...", q))
            })
    )
    .add_tool(
        Tool::new("calculator", "Evaluate mathematical expressions.")
            .param("expression", "string", "Math expression like '137 * 48'")
            .call(|args| {
                let expr = args["expression"].as_str().unwrap_or("0");
                Ok(format!("Result: {}", expr))
            })
    )
    .build()?
```

**Key points:**
- `.param()` → required parameter (included in JSON Schema `required` array)
- `.param_opt()` → optional parameter
- `.call()` → attaches the function, must be called last
- `param_type` is a JSON Schema type string: `"string"`, `"integer"`, `"number"`, `"boolean"`, `"array"`, `"object"`

---

## Raw Tool Registration (Advanced)

For full control over the JSON Schema, use the original `.tool()` method:

Use `AgentBuilder::tool()` to register tools during construction:

```rust
use serde_json::json;
use std::collections::HashMap;

let mut engine = AgentBuilder::new("task")
    .llm(llm)
    .tool(
        "search",                                     // (1) unique name
        "Search the web for current information.",    // (2) description for the LLM
        json!({                                       // (3) JSON Schema for parameters
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }),
        Box::new(|args: &HashMap<String, serde_json::Value>| {  // (4) the function
            let query = args["query"].as_str().unwrap_or("");
            // call your search API here
            Ok(format!("Results for '{}': ...", query))
        }),
    )
    .build()?;
```

### The Four Arguments

| # | Argument | Type | Purpose |
|---|---|---|---|
| 1 | `name` | `impl Into<String>` | Unique identifier — the LLM uses this name when requesting the tool |
| 2 | `description` | `impl Into<String>` | Shown to the LLM in every call — be clear and specific about when to use it |
| 3 | `schema` | `serde_json::Value` | JSON Schema object describing the required parameters |
| 4 | `func` | `Box<dyn Fn(...)>` | The actual implementation — receives args, returns `Result<String, String>` |

---

## Writing Good Tool Descriptions

The description is what the LLM reads to decide *when* to use the tool. Be explicit:

```rust
// ❌ Vague — LLM may misuse this
"Search for information."

// ✅ Clear purpose and usage guidance
"Search the web for current factual information. Use this for questions about \
 recent events, statistics, company data, or any information that requires \
 up-to-date knowledge. Do NOT use for mathematical calculations."
```

---

## Writing Good JSON Schemas

Be specific about each parameter. The LLM generates argument values based on the schema:

```rust
json!({
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City and country, e.g. 'London, UK' or 'New York, USA'"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit. Default to 'celsius' unless user specifies otherwise."
        }
    },
    "required": ["location"]  // units is optional
})
```

### Tip: Use `enum` for constrained choices

```rust
"format": {
    "type": "string",
    "enum": ["json", "csv", "markdown"],
    "description": "Output format"
}
```

---

## The Tool Function Signature

```rust
Arc<dyn Fn(&HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync>
```

- Input: `&HashMap<String, serde_json::Value>` — the parsed arguments from the LLM
- Output: `Ok(String)` on success, `Err(String)` on failure
- **Must never panic** — panics inside tools are NOT caught and will crash the agent

### Safe Argument Access

```rust
Box::new(|args: &HashMap<String, serde_json::Value>| {
    // Always use .get() + type-safe accessor — never index directly on production tools
    let query = args.get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing required argument: query".to_string())?;

    let limit = args.get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10);  // Optional with default

    Ok(format!("Search '{}' (limit {}): results here", query, limit))
})
```

---

## Tool Error Handling

**Tool errors are not crashes — they are data.**

When your tool returns `Err(...)`, `ActingState` stores `"ERROR: <your message>"` in `memory.last_observation`, and the agent proceeds normally. On the next Planning cycle, the LLM sees the error in its history and can try a different approach.

```rust
Box::new(|args: &HashMap<String, serde_json::Value>| {
    let url = args.get("url").and_then(|v| v.as_str())
        .ok_or("Missing url argument")?;

    if url.starts_with("file://") {
        // Return a descriptive error — the LLM will see this and can try a different URL
        return Err(format!("Cannot fetch local file URLs for security reasons: {}", url));
    }

    // ... fetch the URL ...
    Ok("Page content here".to_string())
})
```

---

## Multiple Tools

Register as many tools as needed. The LLM will select the most appropriate one based on your descriptions:

```rust
AgentBuilder::new("Research and calculate compound interest for $10,000 at 7% for 10 years")
    .llm(llm)
    .tool("search",     "Search for financial data...", search_schema,     search_fn)
    .tool("calculator", "Evaluate mathematical expressions...", calc_schema, calc_fn)
    .parallel_tools(true) // Enable parallel execution of independent tools
    .build()?
```

---

## Parallel Tool Execution

If `parallel_tools` is enabled (default: true) and the LLM produces a multi-tool-call response, those tools are executed simultaneously in a thread pool.

- Results are merged and presented to the LLM in the next turn.
- If one tool fails, others continue.
- Useful for speeding up independent operations (e.g., searching 3 websites at once).

---

## Sub-Agents as Tools

`agentsm-rs` allows you to treat an agent as a regular tool. This enables recursive delegation and modular multi-agent systems.

### Using `AgentBuilder::as_tool`

```rust
let calculator_agent = AgentBuilder::new("math")
    .openai(key)
    .add_tool(calc_tool)
    .as_tool("math_specialist", "A specialist for complex mathematical proofs.");

AgentBuilder::new("Prove the Riemann Hypothesis")
    .openai(key)
    .add_subagent(calculator_agent) // Register it like any other tool
    .build()?
```

**How it works:**
- The sub-agent is cloned into the tool registry (via `Arc`).
- When the parent calls the sub-agent tool, the sub-agent runs to completion (synchronously from the perspective of the tool call).
- The sub-agent's final answer becomes the tool observation for the parent.

---

## Blacklisting Tools

Prevent the agent from using specific tools, even if they are registered:

```rust
AgentBuilder::new("task")
    .llm(llm)
    .tool("search", ...)
    .tool("delete_files", ...)   // registered but forbidden
    .blacklist_tool("delete_files")
    .build()?
```

When the LLM requests a blacklisted tool, `PlanningState` logs the attempt and returns `Event::ToolBlacklisted`, which transitions back to `Planning` — the LLM gets another chance without executing the forbidden tool.

Blacklisting is also available after construction:

```rust
engine.memory.blacklist_tool("dangerous_tool");
```

---

## Accessing the ToolRegistry Directly

For programmatic registration outside of the builder:

```rust
use agentsm::ToolRegistry;
use serde_json::json;

let mut registry = ToolRegistry::new();

registry.register(
    "ping",
    "Check if a host is reachable",
    json!({ "type": "object", "properties": { "host": { "type": "string" } }, "required": ["host"] }),
    Box::new(|args| {
        let host = args["host"].as_str().unwrap_or("");
        Ok(format!("{} is reachable", host))
    }),
);

println!("Registered {} tools", registry.len());
println!("Has 'ping'? {}", registry.has("ping"));

let result = registry.execute("ping", &[("host".to_string(), serde_json::json!("8.8.8.8"))].into());
```

---

## Tool Schema Reference

`ToolRegistry::schemas()` returns `Vec<ToolSchema>`, which is what LLM callers send to the API:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ToolSchema {
    pub name:         String,
    pub description:  String,
    pub input_schema: serde_json::Value,  // The JSON Schema object
}
```

For OpenAI, this maps to the `tools` array in the chat completion request.  
For Anthropic, this maps to the `tools` array in the messages request.
