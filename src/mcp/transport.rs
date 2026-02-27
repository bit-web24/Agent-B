use tokio::process::{Child, Command};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use std::process::Stdio;
use anyhow::{Result, Context};
use crate::mcp::types::{JsonRpcRequest, JsonRpcResponse, JsonRpcNotification};
use serde_json::Value;

pub struct StdioTransport {
    pub child: Child,
    pub writer: BufWriter<tokio::process::ChildStdin>,
    pub reader: BufReader<tokio::process::ChildStdout>,
}

impl StdioTransport {
    pub fn spawn(command: &str, args: &[String]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn MCP server process")?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("Failed to open stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("Failed to open stdout"))?;

        Ok(Self {
            child,
            writer: BufWriter::new(stdin),
            reader: BufReader::new(stdout),
        })
    }
}

pub async fn send_request(writer: &mut BufWriter<tokio::process::ChildStdin>, request: &JsonRpcRequest) -> Result<()> {
    let json = serde_json::to_string(request)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

pub async fn send_notification(writer: &mut BufWriter<tokio::process::ChildStdin>, notif: &JsonRpcNotification) -> Result<()> {
    let json = serde_json::to_string(notif)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_message(reader: &mut BufReader<tokio::process::ChildStdout>) -> Result<McpMessage> {
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        if line.is_empty() {
             return Err(anyhow::anyhow!("Connection closed"));
        }

        let val: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                tracing::debug!("Skipping non-JSON line from MCP server: {}", line.trim());
                continue;
            }
        };
        
        if val.get("id").is_some() && !val["id"].is_null() {
            if val.get("method").is_some() {
                return Ok(McpMessage::Request(serde_json::from_value(val)?));
            } else {
                return Ok(McpMessage::Response(serde_json::from_value(val)?));
            }
        } else {
            return Ok(McpMessage::Notification(serde_json::from_value(val)?));
        }
    }
}

pub enum McpMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}
