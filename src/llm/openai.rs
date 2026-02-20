use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage,
        ChatCompletionTool,
        ChatCompletionToolType,
        CreateChatCompletionRequestArgs,
        FunctionObject,
        ChatCompletionMessageToolCall,
    },
    Client,
};
use async_trait::async_trait;
use crate::llm::AsyncLlmCaller;
use crate::memory::AgentMemory;
use crate::tools::ToolRegistry;
use crate::types::{LlmResponse, ToolCall};
use std::collections::HashMap;

pub struct OpenAiCaller {
    client: Client<OpenAIConfig>,
}

impl OpenAiCaller {
    /// Standard OpenAI client using OPENAI_API_KEY env var
    pub fn new() -> Self {
        Self { client: Client::new() }
    }

    /// Custom base URL — for Groq, Together, Ollama, Fireworks, etc.
    /// api_base example: "https://api.groq.com/openai/v1"
    pub fn with_base_url(api_base: impl Into<String>, api_key: impl Into<String>) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(api_base)
            .with_api_key(api_key);
        Self { client: Client::with_config(config) }
    }

    /// Convert our ToolSchema into async-openai's ChatCompletionTool type
    fn build_tools(tools: &ToolRegistry) -> Vec<ChatCompletionTool> {
        tools.schemas().into_iter().map(|schema| {
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name:        schema.name,
                    description: Some(schema.description),
                    parameters:  Some(schema.input_schema),
                },
            }
        }).collect()
    }

    /// Parse the first tool call from an OpenAI response into our ToolCall type
    fn parse_tool_call(tc: &ChatCompletionMessageToolCall) -> Result<ToolCall, String> {
        let args: HashMap<String, serde_json::Value> =
            serde_json::from_str(&tc.function.arguments)
                .map_err(|e| format!("Failed to parse tool args: {}", e))?;
        Ok(ToolCall {
            name: tc.function.name.clone(),
            args,
        })
    }
}

#[async_trait]
impl AsyncLlmCaller for OpenAiCaller {
    async fn call_async(
        &self,
        memory: &AgentMemory,
        tools:  &ToolRegistry,
        model:  &str,
    ) -> Result<LlmResponse, String> {
        let messages_json = memory.build_messages();

        // Convert serde_json::Value messages to async-openai types
        // Use serde round-trip: serialize to string, deserialize as typed
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::Value::Array(messages_json))
                .map_err(|e| format!("Failed to build messages: {}", e))?;

        let oai_tools = Self::build_tools(tools);

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(model).messages(messages);

        if !oai_tools.is_empty() {
            request_builder.tools(oai_tools);
        }

        let request = request_builder.build()
            .map_err(|e| format!("Failed to build request: {}", e))?;

        let response = self.client.chat()
            .create(request)
            .await
            .map_err(|e| format!("OpenAI API error: {}", e))?;

        let choice = response.choices.into_iter().next()
            .ok_or("Empty response from OpenAI")?;

        let message = choice.message;

        // Tool call takes priority over text content
        if let Some(tool_calls) = message.tool_calls {
            if let Some(tc) = tool_calls.into_iter().next() {
                let tool = Self::parse_tool_call(&tc)?;
                return Ok(LlmResponse::ToolCall { tool, confidence: 1.0 });
            }
        }

        let content = message.content
            .ok_or("No content in OpenAI response")?;

        Ok(LlmResponse::FinalAnswer { content })
    }
}
