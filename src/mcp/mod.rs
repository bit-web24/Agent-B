pub mod types;
pub mod transport;
pub mod client;

pub use client::McpClient;
pub use types::{McpTool, CallToolResult, McpContent};

use std::sync::Arc;
use std::collections::HashMap;
use crate::tools::ToolFn;
use serde_json::Value;

/// Bridges an MCP tool into an Agent-B ToolFn.
pub fn bridge_mcp_tool(client: Arc<McpClient>, tool_name: String) -> ToolFn {
    Box::new(move |args: &HashMap<String, Value>| {
        let client = Arc::clone(&client);
        let name = tool_name.clone();
        let args_clone = args.clone();

        // Use tokio::task::block_in_place to bridge async to sync.
        // This is safe because we are called from the AgentEngine which is potentially
        // running inside a tokio runtime (via SyncWrapper).
        tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            let result = handle.block_on(client.call_tool(&name, args_clone));
            
            match result {
                Ok(res) => {
                    let mut output = String::new();
                    for content in res.content {
                        if let McpContent::Text { text } = content {
                            output.push_str(&text);
                            output.push('\n');
                        }
                    }
                    if res.is_error {
                        Err(output.trim().to_string())
                    } else {
                        Ok(output.trim().to_string())
                    }
                }
                Err(e) => Err(format!("MCP tool error: {}", e)),
            }
        })
    })
}
