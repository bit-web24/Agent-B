use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use tokio::sync::{oneshot, Mutex};
use tokio::io::{BufReader, BufWriter};
use anyhow::{Result, Context};
use crate::mcp::transport::{StdioTransport, McpMessage, send_request, send_notification, read_message};
use crate::mcp::types::*;
use serde_json::json;

pub struct McpClient {
    writer:    Mutex<BufWriter<tokio::process::ChildStdin>>,
    next_id:   AtomicU64,
    pending:   Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
}

impl McpClient {
    pub async fn new(command: &str, args: &[String]) -> Result<Arc<Self>> {
        let transport = StdioTransport::spawn(command, args)?;
        let StdioTransport { writer, mut reader, .. } = transport;

        let client = Arc::new(Self {
            writer:    Mutex::new(writer),
            next_id:   AtomicU64::new(1),
            pending:   Arc::new(Mutex::new(HashMap::new())),
        });

        // Start background reader loop
        let pending_clone = Arc::clone(&client.pending);
        tokio::spawn(async move {
            if let Err(e) = Self::run_reader_loop(&mut reader, pending_clone).await {
                tracing::error!("MCP reader loop failed: {}", e);
            }
        });

        // Initialize handshake
        client.initialize().await?;

        Ok(client)
    }

    async fn run_reader_loop(
        reader: &mut BufReader<tokio::process::ChildStdout>,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    ) -> Result<()> {
        loop {
            let msg = read_message(reader).await?;

            match msg {
                McpMessage::Response(resp) => {
                    if let Some(id_val) = resp.id.as_u64() {
                        let mut pending_guard = pending.lock().await;
                        if let Some(tx) = pending_guard.remove(&id_val) {
                            let _ = tx.send(resp);
                        }
                    }
                }
                McpMessage::Request(req) => {
                    tracing::debug!("Received MCP request from server: {:?}", req);
                    // TODO: Handle server-to-client requests if needed
                }
                McpMessage::Notification(notif) => {
                    tracing::debug!("Received MCP notification from server: {:?}", notif);
                }
            }
        }
    }

    async fn send_request_internal(&self, method: &str, params: Option<serde_json::Value>) -> Result<JsonRpcResponse> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method:  method.to_string(),
            params,
            id:      json!(id),
        };

        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        {
            let mut writer = self.writer.lock().await;
            send_request(&mut writer, &request).await?;
        }

        rx.await.context("MCP response channel closed")
    }

    async fn initialize(&self) -> Result<()> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "agent-b",
                "version": "0.1.0"
            }
        });

        let resp = self.send_request_internal("initialize", Some(params)).await?;
        if let Some(err) = resp.error {
            return Err(anyhow::anyhow!("MCP initialization failed: {}", err.message));
        }

        // Send initialized notification
        let notif = JsonRpcNotification {
            jsonrpc: "2.0".to_string(),
            method:  "notifications/initialized".to_string(),
            params:  Some(json!({})),
        };
        
        {
            let mut writer = self.writer.lock().await;
            send_notification(&mut writer, &notif).await?;
        }

        Ok(())
    }

    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let resp = self.send_request_internal("tools/list", Some(json!({}))).await?;
        if let Some(err) = resp.error {
            return Err(anyhow::anyhow!("Failed to list tools: {}", err.message));
        }

        let result: ListToolsResult = serde_json::from_value(resp.result.clone().unwrap_or_default())?;
        Ok(result.tools)
    }

    pub async fn call_tool(&self, name: &str, arguments: HashMap<String, serde_json::Value>) -> Result<CallToolResult> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });

        let resp = self.send_request_internal("tools/call", Some(params)).await?;
        if let Some(err) = resp.error {
            return Err(anyhow::anyhow!("Tool call failed: {}", err.message));
        }

        let result: CallToolResult = serde_json::from_value(resp.result.clone().unwrap_or_default())?;
        Ok(result)
    }
}
