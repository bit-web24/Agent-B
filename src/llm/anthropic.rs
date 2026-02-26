use async_trait::async_trait;
use crate::llm::AsyncLlmCaller;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::{LlmResponse, ToolCall};

// ── Anthropic request types ──────────────────────────────

#[derive(serde::Serialize)]
struct AnthropicRequest {
    model:      String,
    max_tokens: u32,
    system:     Option<String>,
    tools:      Vec<AnthropicToolDef>,
    messages:   Vec<AnthropicMessage>,
    stream:     bool,
}

#[derive(serde::Serialize)]
struct AnthropicToolDef {
    name:         String,
    description:  String,
    input_schema: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct AnthropicMessage {
    role:    String,
    content: serde_json::Value,   // string or array of content blocks
}

// ── Anthropic response types ─────────────────────────────

#[derive(serde::Deserialize, Debug)]
struct AnthropicResponse {
    content:     Vec<AnthropicContentBlock>,
    stop_reason: String,
    usage:       AnthropicUsage,
}

#[derive(serde::Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens:  u32,
    output_tokens: u32,
}

#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id:    String,
        name:  String,
        input: serde_json::Value,
    },
}

#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: usize, content_block: AnthropicContentBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: AnthropicDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: AnthropicMessageDelta, usage: AnthropicUsage },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

#[derive(serde::Deserialize, Debug)]
#[serde(tag = "type")]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(serde::Deserialize, Debug)]
struct AnthropicMessageDelta {
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
}

// ── Caller ───────────────────────────────────────────────

pub struct AnthropicCaller {
    client:  reqwest::Client,
    api_key: String,
    api_base: String,
}

impl AnthropicCaller {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client:   reqwest::Client::new(),
            api_key:  api_key.into(),
            api_base: "https://api.anthropic.com".to_string(),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;
        Ok(Self::new(key))
    }

    fn build_tool_defs(tools: &ToolRegistry) -> Vec<AnthropicToolDef> {
        tools.schemas().into_iter().map(|s| AnthropicToolDef {
            name:         s.name,
            description:  s.description,
            input_schema: s.input_schema,
        }).collect()
    }

    fn build_messages(memory: &AgentMemory) -> Vec<AnthropicMessage> {
        // Convert memory.build_messages() (serde_json::Value array)
        // into Vec<AnthropicMessage>
        // Filter out "system" role (sent separately in AnthropicRequest.system)
        memory.build_messages()
            .into_iter()
            .filter(|m| m["role"] != "system")
            .map(|m| AnthropicMessage {
                role:    m["role"].as_str().unwrap_or("user").to_string(),
                content: m["content"].clone(),
            })
            .collect()
    }
}

#[async_trait]
impl AsyncLlmCaller for AnthropicCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        let system = if memory.system_prompt.is_empty() {
            None
        } else {
            Some(memory.system_prompt.clone())
        };

        let body = AnthropicRequest {
            model:      model.to_string(),
            max_tokens: 4096,
            system,
            tools:      Self::build_tool_defs(tools),
            messages:   Self::build_messages(memory),
            stream:     false,
        };

        let response = self.client
            .post(format!("{}/v1/messages", self.api_base))
            .header("x-api-key",         &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type",      "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Network error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body   = response.text().await.unwrap_or_default();
            return Err(format!("Anthropic API error {}: {}", status, body));
        }

        let parsed: AnthropicResponse = response.json()
            .await
            .map_err(|e| format!("Failed to parse Anthropic response: {}", e))?;

        // Tool use takes priority
        for block in parsed.content {
            match block {
                AnthropicContentBlock::ToolUse { id, name, input, .. } => {
                    let args = serde_json::from_value(input)
                        .map_err(|e| format!("Invalid tool args: {}", e))?;
                    return Ok(LlmResponse::ToolCall {
                        tool: ToolCall { name, args, id: Some(id) },
                        confidence: 1.0,
                    });
                }
                AnthropicContentBlock::Text { text } => {
                    return Ok(LlmResponse::FinalAnswer { content: text });
                }
            }
        }

        Err("Anthropic returned empty content".to_string())
    }

    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
    ) -> futures::stream::BoxStream<'a, Result<crate::types::LlmStreamChunk, String>> {
        use futures::{StreamExt, stream};
        
        let system = if memory.system_prompt.is_empty() {
            None
        } else {
            Some(memory.system_prompt.clone())
        };

        let body = AnthropicRequest {
            model:      model.to_string(),
            max_tokens: 4096,
            system,
            tools:      Self::build_tool_defs(tools),
            messages:   Self::build_messages(memory),
            stream:     true,
        };

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();

        let s = stream::once(async move {
            client
                .post(format!("{}/v1/messages", api_base))
                .header("x-api-key",         &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type",      "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| format!("Network error: {}", e))
        })
        .flat_map(|res| {
            match res {
                Ok(resp) if resp.status().is_success() => {
                    let mut accumulated_content = String::new();
                    let mut accumulated_tool_id = String::new();
                    let mut accumulated_tool_name = String::new();
                    let mut accumulated_tool_args = String::new();
                    
                    resp.bytes_stream()
                        .map(|b| b.map_err(|e| format!("Stream error: {}", e)))
                        .map(move |res| {
                            let bytes = res?;
                            let s = String::from_utf8_lossy(&bytes);
                            let mut chunks = Vec::new();
                            
                            for line in s.lines() {
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                                        match event {
                                            AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                                                if let AnthropicContentBlock::ToolUse { id, name, .. } = content_block {
                                                    accumulated_tool_id = id;
                                                    accumulated_tool_name = name;
                                                }
                                            }
                                            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                                                match delta {
                                                    AnthropicDelta::TextDelta { text } => {
                                                        accumulated_content.push_str(&text);
                                                        chunks.push(Ok(crate::types::LlmStreamChunk::Content(text)));
                                                    }
                                                    AnthropicDelta::InputJsonDelta { partial_json } => {
                                                        accumulated_tool_args.push_str(&partial_json);
                                                        chunks.push(Ok(crate::types::LlmStreamChunk::ToolCallDelta {
                                                            name: Some(accumulated_tool_name.clone()),
                                                            args_json: accumulated_tool_args.clone(),
                                                        }));
                                                    }
                                                }
                                            }
                                            AnthropicStreamEvent::MessageDelta { delta, .. } => {
                                                if delta.stop_reason.is_some() {
                                                    if !accumulated_tool_args.is_empty() {
                                                        let args: std::collections::HashMap<String, serde_json::Value> = 
                                                            serde_json::from_str(&accumulated_tool_args)
                                                                .map_err(|e| format!("Failed to parse Anthropic tool args: {}", e))?;
                                                        chunks.push(Ok(crate::types::LlmStreamChunk::Done(LlmResponse::ToolCall {
                                                            tool: ToolCall { name: accumulated_tool_name.clone(), args, id: Some(accumulated_tool_id.clone()) },
                                                            confidence: 1.0,
                                                        })));
                                                    } else if !accumulated_content.is_empty() {
                                                        chunks.push(Ok(crate::types::LlmStreamChunk::Done(LlmResponse::FinalAnswer {
                                                            content: accumulated_content.clone(),
                                                        })));
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            Ok(chunks)
                        })
                        .flat_map(|res| {
                            match res {
                                Ok(chunks) => stream::iter(chunks),
                                Err(e) => stream::iter(vec![Err(e)]),
                            }
                        })
                        .boxed()
                }
                Ok(resp) => {
                    stream::once(async move {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        Err(format!("Anthropic API error {}: {}", status, body))
                    }).boxed()
                }
                Err(e) => stream::once(async move { Err(e) }).boxed(),
            }
        });

        s.boxed()
    }
}
