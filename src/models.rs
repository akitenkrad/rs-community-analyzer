use serde::{Deserialize, Serialize};

/// Organizational role of a user．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Exec,
    Manager,
    Lead,
    Staff,
    Unknown,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Exec => "exec",
            Role::Manager => "manager",
            Role::Lead => "lead",
            Role::Staff => "staff",
            Role::Unknown => "unknown",
        }
    }

    /// Parse a role string leniently, falling back to `Unknown`．
    pub fn from_str_lenient(s: &str) -> Role {
        match s.trim().to_ascii_lowercase().as_str() {
            "exec" => Role::Exec,
            "manager" => Role::Manager,
            "lead" => Role::Lead,
            "staff" => Role::Staff,
            _ => Role::Unknown,
        }
    }
}

/// Category of a channel．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelCategory {
    Official,
    Tech,
    Casual,
    Leadership,
    PrivateGroup,
    Unknown,
}

impl ChannelCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChannelCategory::Official => "official",
            ChannelCategory::Tech => "tech",
            ChannelCategory::Casual => "casual",
            ChannelCategory::Leadership => "leadership",
            ChannelCategory::PrivateGroup => "private_group",
            ChannelCategory::Unknown => "unknown",
        }
    }

    /// Parse a category string leniently, falling back to `Unknown`．
    pub fn from_str_lenient(s: &str) -> ChannelCategory {
        match s.trim().to_ascii_lowercase().as_str() {
            "official" => ChannelCategory::Official,
            "tech" => ChannelCategory::Tech,
            "casual" => ChannelCategory::Casual,
            "leadership" => ChannelCategory::Leadership,
            "private_group" => ChannelCategory::PrivateGroup,
            _ => ChannelCategory::Unknown,
        }
    }
}

fn default_fiscal_start() -> u32 {
    4
}

/// Fiscal-year configuration．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiscalSpec {
    #[serde(default = "default_fiscal_start")]
    pub fiscal_year_start_month: u32,
}

impl Default for FiscalSpec {
    fn default() -> Self {
        Self {
            fiscal_year_start_month: default_fiscal_start(),
        }
    }
}

/// Analysis period specification．
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeriodSpec {
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(default)]
    pub fiscal: FiscalSpec,
}

/// Top-level communication-analysis report．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommReport {
    pub run_id: String,
    pub config_hash: String,
    pub period: PeriodSpec,
    pub h1: Option<H1Metrics>,
    pub h2: Option<H2Metrics>,
    pub h3: Option<H3Metrics>,
    pub h4: Option<H4Metrics>,
    pub h5: Option<H5Metrics>,
    pub summary: ReportSummary,
}

/// H1: decision-making dynamics metrics．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H1Metrics {
    pub decision_change_rate: f64,
    pub initial_proposal_regression: f64,
    pub silence_after_manager: SilenceRatio,
    pub dissent_convergence_speed_minutes: f64,
    pub thread_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilenceRatio {
    pub manager_post_silence_rate: f64,
    pub staff_post_silence_rate: f64,
    pub ratio: f64,
}

/// H2: psychological-safety metrics．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H2Metrics {
    pub hedging_rate: f64,
    pub self_defense_rate: f64,
    pub incomplete_utterance_rate: f64,
    pub escalation_delay_minutes: f64,
    pub dm_dependency_ratio: Option<f64>,
    pub sample_size: usize,
}

/// H3: power-concentration metrics．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H3Metrics {
    pub utterance_entropy_by_channel: Vec<(String, f64)>,
    pub normalized_concentration: Vec<(String, f64)>,
    pub reply_concentration_gini: f64,
    pub manager_reaction_rate: f64,
    pub dissent_target_bias: DissentBias,
    pub hierarchy_score: f64,
    pub pagerank_topk_manager_share: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissentBias {
    pub to_manager: f64,
    pub to_staff: f64,
    pub ratio_staff_to_manager: f64,
}

/// H4: surface-agreement / hidden-dissent metrics．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H4Metrics {
    pub surface_agreement_rate: f64,
    pub public_private_sentiment_delta: Option<f64>,
    pub post_meeting_dissent_rate: f64,
    pub execution_delay_hours: f64,
    pub reaction_text_disagreement: Option<f64>,
}

/// H5: idea-diversity / exploration metrics．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H5Metrics {
    pub proposal_semantic_diversity: Option<f64>,
    pub hypothesis_retention_period_hours: f64,
    pub unresolved_thread_ratio: f64,
    pub novel_vocabulary_rate_monthly: Vec<(String, f64)>,
    pub topic_cluster_count_monthly: Option<Vec<(String, u32)>>,
}

/// Aggregated report summary with health flags．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub overall_health_score: f64,
    pub red_flags: Vec<String>,
    pub yellow_flags: Vec<String>,
    pub green_signals: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serde_roundtrip() {
        let json = serde_json::to_string(&Role::Exec).unwrap();
        assert_eq!(json, "\"exec\"");
        let parsed: Role = serde_json::from_str("\"exec\"").unwrap();
        assert_eq!(parsed, Role::Exec);

        let json = serde_json::to_string(&Role::Unknown).unwrap();
        assert_eq!(json, "\"unknown\"");
        let parsed: Role = serde_json::from_str("\"manager\"").unwrap();
        assert_eq!(parsed, Role::Manager);
    }

    #[test]
    fn test_channel_category_serde_roundtrip() {
        let json = serde_json::to_string(&ChannelCategory::PrivateGroup).unwrap();
        assert_eq!(json, "\"private_group\"");
        let parsed: ChannelCategory = serde_json::from_str("\"private_group\"").unwrap();
        assert_eq!(parsed, ChannelCategory::PrivateGroup);

        let parsed: ChannelCategory = serde_json::from_str("\"official\"").unwrap();
        assert_eq!(parsed, ChannelCategory::Official);
    }

    #[test]
    fn test_role_from_str_lenient() {
        assert_eq!(Role::from_str_lenient("EXEC"), Role::Exec);
        assert_eq!(Role::from_str_lenient(" manager "), Role::Manager);
        assert_eq!(Role::from_str_lenient("nope"), Role::Unknown);
    }

    #[test]
    fn test_channel_category_from_str_lenient() {
        assert_eq!(
            ChannelCategory::from_str_lenient("private_group"),
            ChannelCategory::PrivateGroup
        );
        assert_eq!(
            ChannelCategory::from_str_lenient("garbage"),
            ChannelCategory::Unknown
        );
    }

    #[test]
    fn test_period_spec_default() {
        let p = PeriodSpec::default();
        assert!(p.from.is_none());
        assert_eq!(p.fiscal.fiscal_year_start_month, 4);
    }
}
