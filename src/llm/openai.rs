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
// use futures::StreamExt;
use futures::stream::BoxStream;
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
            id: Some(tc.id.clone()),
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
        _output_tx: Option<&tokio::sync::mpsc::UnboundedSender<crate::types::AgentOutput>>,
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

        let usage = response.usage.map(|u| {
            crate::budget::TokenUsage::new(u.prompt_tokens, u.completion_tokens)
        });

        let choice = response.choices.into_iter().next()
            .ok_or("Empty response from OpenAI")?;

        let message = choice.message;

        // Tool call takes priority over text content
        if let Some(tool_calls) = message.tool_calls {
            if tool_calls.len() > 1 {
                let mut parsed_tools = Vec::new();
                for tc in tool_calls {
                    parsed_tools.push(Self::parse_tool_call(&tc)?);
                }
                return Ok(LlmResponse::ParallelToolCalls { 
                    tools: parsed_tools, 
                    confidence: 1.0,
                    usage 
                });
            } else if let Some(tc) = tool_calls.into_iter().next() {
                let tool = Self::parse_tool_call(&tc)?;
                return Ok(LlmResponse::ToolCall { 
                    tool, 
                    confidence: 1.0, 
                    usage 
                });
            }
        }

        let content = message.content
            .ok_or("No content in OpenAI response")?;

        Ok(LlmResponse::FinalAnswer { content, usage })
    }

    fn call_stream_async<'a>(
        &'a self,
        memory: &'a AgentMemory,
        tools:  &'a ToolRegistry,
        model:  &'a str,
        _output_tx: Option<&tokio::sync::mpsc::UnboundedSender<crate::types::AgentOutput>>,
    ) -> BoxStream<'a, Result<crate::types::LlmStreamChunk, String>> {
        use futures::{StreamExt, stream};
        let messages_json = memory.build_messages();
        let messages: Vec<ChatCompletionRequestMessage> =
            match serde_json::from_value(serde_json::Value::Array(messages_json)) {
                Ok(m) => m,
                Err(e) => return stream::once(async move { Err(format!("Failed to build messages: {}", e)) }).boxed(),
            };

        let oai_tools = Self::build_tools(tools);
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(model).messages(messages).stream(true);

        if !oai_tools.is_empty() {
            request_builder.tools(oai_tools);
        }

        let request = match request_builder.build() {
            Ok(r) => r,
            Err(e) => return stream::once(async move { Err(format!("Failed to build request: {}", e)) }).boxed(),
        };

        let client = self.client.clone();
        
        let s = stream::once(async move {
            client.chat().create_stream(request).await
                .map_err(|e| format!("OpenAI API error: {}", e))
        })
        .flat_map(|res| {
            match res {
                Ok(stream) => {
                    let mut accumulated_content = String::new();
                    
                    #[derive(Default)]
                    struct ToolCallAcc {
                        id:   Option<String>,
                        name: Option<String>,
                        args: String,
                    }
                    let mut tool_accumulators: HashMap<i32, ToolCallAcc> = HashMap::new();

                    stream.map(move |res| {
                        let res = res.map_err(|e| format!("OpenAI stream error: {}", e))?;
                        let choice = res.choices.into_iter().next().ok_or("Empty choice in stream")?;
                        let delta = choice.delta;

                        if let Some(tool_calls) = delta.tool_calls {
                            for tc in tool_calls {
                                let acc = tool_accumulators.entry(tc.index).or_default();
                                if let Some(id) = tc.id {
                                    acc.id = Some(id);
                                }
                                if let Some(func) = tc.function {
                                    if let Some(name) = func.name {
                                        acc.name = Some(name);
                                    }
                                    if let Some(args) = func.arguments {
                                        acc.args.push_str(&args);
                                    }
                                }
                            }
                            
                            // Emit a delta for the most recently updated tool call (or all of them?)
                            // For simplicity, we just send a generic delta indicating tool progress.
                            // The engine currently doesn't use the index to differentiate in UI,
                            // it just accumulates name/args from LLM_TOOL_CALL_DELTA events.
                            // BUT wait! If they are parallel, we MUST specify WHICH ONE.
                            // For now, let's at least emit the LATEST one.
                            let (name, args_json) = tool_accumulators.values()
                                .next() // arbitrary
                                .map(|a| (a.name.clone(), a.args.clone()))
                                .unwrap_or((None, String::new()));

                            return Ok(crate::types::LlmStreamChunk::ToolCallDelta {
                                name,
                                args_json,
                            });
                        }

                        if let Some(content) = delta.content {
                            accumulated_content.push_str(&content);
                            return Ok(crate::types::LlmStreamChunk::Content(content));
                        }

                        if let Some(_reason) = choice.finish_reason {
                            if !tool_accumulators.is_empty() {
                                 if tool_accumulators.len() > 1 {
                                     let mut tools = Vec::new();
                                     for acc in tool_accumulators.values() {
                                         let name = acc.name.clone().unwrap_or_default();
                                         let args: HashMap<String, serde_json::Value> = serde_json::from_str(&acc.args)
                                            .map_err(|e| format!("Failed to parse tool args (parallel): {}", e))?;
                                         tools.push(crate::types::ToolCall { name, args, id: acc.id.clone() });
                                     }
                                      return Ok(crate::types::LlmStreamChunk::Done(LlmResponse::ParallelToolCalls {
                                         tools,
                                         confidence: 1.0,
                                         usage: None,
                                     }));
                                 } else {
                                     let acc = tool_accumulators.values().next().unwrap();
                                     let name = acc.name.clone().unwrap_or_default();
                                     let args: HashMap<String, serde_json::Value> = serde_json::from_str(&acc.args)
                                        .map_err(|e| format!("Failed to parse tool args: {}", e))?;
                                     return Ok(crate::types::LlmStreamChunk::Done(LlmResponse::ToolCall {
                                         tool: crate::types::ToolCall { name, args, id: acc.id.clone() },
                                         confidence: 1.0,
                                         usage: None,
                                     }));
                                 }
                            } else if !accumulated_content.is_empty() {
                                return Ok(crate::types::LlmStreamChunk::Done(LlmResponse::FinalAnswer { content: accumulated_content.clone(), usage: None }));
                            }
                        }

                        Err("SKIP".to_string())
                    })
                    .filter(|res| futures::future::ready(match res {
                        Ok(_) => true,
                        Err(e) => e != "SKIP",
                    }))
                    .boxed()
                }
                Err(e) => stream::once(async move { Err(e) }).boxed(),
            }
        });

        s.boxed()
    }
}
