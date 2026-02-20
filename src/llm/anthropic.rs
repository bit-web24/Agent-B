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
                AnthropicContentBlock::ToolUse { name, input, .. } => {
                    let args = serde_json::from_value(input)
                        .map_err(|e| format!("Invalid tool args: {}", e))?;
                    return Ok(LlmResponse::ToolCall {
                        tool: ToolCall { name, args },
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
}
