use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanApprovalRequest {
    pub tool_name: String,
    pub tool_args: HashMap<String, serde_json::Value>,
    pub risk_level: RiskLevel,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HumanDecision {
    Approved,
    Rejected(String), // Reason for rejection
    Modified {
        tool_name: String,
        tool_args: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone)]
pub enum ApprovalPolicy {
    AlwaysAsk,
    NeverAsk,
    AskAbove(RiskLevel),
    ToolBased(HashMap<String, RiskLevel>),
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        Self::AskAbove(RiskLevel::High)
    }
}

impl ApprovalPolicy {
    pub fn needs_approval(&self, tool_name: &str, _args: &HashMap<String, serde_json::Value>) -> bool {
        match self {
            Self::AlwaysAsk => true,
            Self::NeverAsk => false,
            Self::AskAbove(threshold) => {
                // Default risk for unknown tools is Medium
                RiskLevel::Medium >= *threshold
            }
            Self::ToolBased(map) => {
                let risk = map.get(tool_name).copied().unwrap_or(RiskLevel::Low);
                risk >= RiskLevel::High // Hardcoded default threshold for tool-based if not specified? 
                // Better to make ToolBased include the threshold.
            }
        }
    }
}
