use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{CommError, Result};
use crate::models::{ChannelCategory, PeriodSpec, Role};

/// Top-level communication-analysis configuration．
///
/// Deserialized from a TOML file．Unknown top-level sections (e.g. `[storage]`,
/// `[private_channel]` added by callers) are tolerated and ignored．
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommConfig {
    #[serde(default)]
    pub period: PeriodSpec,
    #[serde(rename = "roles", default)]
    pub roles: Vec<RoleEntry>,
    #[serde(rename = "channels", default)]
    pub channels: Vec<ChannelEntry>,
    #[serde(default)]
    pub patterns: PatternDict,
    #[serde(default)]
    pub nlp_sidecar: NlpSidecarConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

/// A single role-mapping entry．`user_id` is an opaque platform-specific
/// identifier string (the library does not parse its shape)．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleEntry {
    pub user_id: String,
    pub role: Role,
    #[serde(default)]
    pub seniority_level: i32,
    #[serde(default)]
    pub team: Option<String>,
}

/// A single channel-category mapping entry (regex pattern -> category)．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelEntry {
    pub pattern: String,
    pub category: ChannelCategory,
    #[serde(default)]
    pub is_decision_channel: bool,
}

fn default_hedging() -> Vec<String> {
    vec![
        "仮説ですが".to_string(),
        "未整理ですが".to_string(),
        "とりあえず".to_string(),
        "雑に".to_string(),
        "間違ってたら".to_string(),
        "違ったらすみません".to_string(),
        "FYI".to_string(),
        "ご参考まで".to_string(),
        "メモ程度".to_string(),
    ]
}

fn default_self_defense() -> Vec<String> {
    vec![
        "間違っていたらすみません".to_string(),
        "違ったらすみません".to_string(),
        "詳しくないですが".to_string(),
        "詳しくないんですけど".to_string(),
        "経験浅いですが".to_string(),
        "勘違いかもしれませんが".to_string(),
    ]
}

fn default_surface_agreement() -> Vec<String> {
    vec![
        "承知しました".to_string(),
        "了解です".to_string(),
        "わかりました".to_string(),
        "賛成です".to_string(),
        "問題ないです".to_string(),
        "👍".to_string(),
        ":+1:".to_string(),
        ":ok:".to_string(),
    ]
}

fn default_incomplete() -> Vec<String> {
    vec![
        "えーと".to_string(),
        "うーん".to_string(),
        "いや，".to_string(),
        "あの…".to_string(),
        "その…".to_string(),
    ]
}

fn default_unresolved_marker() -> Vec<String> {
    vec![
        "❓".to_string(),
        "未確定".to_string(),
        "保留".to_string(),
        "未整理".to_string(),
    ]
}

fn default_conclusion_marker() -> Vec<String> {
    vec![
        "結論".to_string(),
        "採用".to_string(),
        "決定".to_string(),
        "FIX".to_string(),
        "確定".to_string(),
    ]
}

/// Detection pattern dictionary．Defaults are built-in Japanese phrases．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDict {
    #[serde(default = "default_hedging")]
    pub hedging: Vec<String>,
    #[serde(default = "default_self_defense")]
    pub self_defense: Vec<String>,
    #[serde(default = "default_surface_agreement")]
    pub surface_agreement: Vec<String>,
    #[serde(default = "default_incomplete")]
    pub incomplete: Vec<String>,
    #[serde(default = "default_unresolved_marker")]
    pub unresolved_marker: Vec<String>,
    #[serde(default = "default_conclusion_marker")]
    pub conclusion_marker: Vec<String>,
}

impl Default for PatternDict {
    fn default() -> Self {
        Self {
            hedging: default_hedging(),
            self_defense: default_self_defense(),
            surface_agreement: default_surface_agreement(),
            incomplete: default_incomplete(),
            unresolved_marker: default_unresolved_marker(),
            conclusion_marker: default_conclusion_marker(),
        }
    }
}

fn d_true() -> bool {
    true
}

fn d_py() -> String {
    "uv run python".to_string()
}

fn d_balanced() -> String {
    "balanced".to_string()
}

fn d_32() -> usize {
    32
}

/// Python NLP sidecar configuration．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlpSidecarConfig {
    #[serde(default = "d_true")]
    pub enabled: bool,
    #[serde(default = "d_py")]
    pub python: String,
    #[serde(default)]
    pub script: String,
    #[serde(default = "d_balanced")]
    pub profile: String,
    #[serde(default)]
    pub device: String,
    #[serde(default = "d_32")]
    pub embed_batch_size: usize,
    #[serde(default)]
    pub ruri_prefix: RuriPrefix,
    #[serde(default)]
    pub models: NlpModels,
}

impl Default for NlpSidecarConfig {
    fn default() -> Self {
        Self {
            enabled: d_true(),
            python: d_py(),
            script: String::new(),
            profile: d_balanced(),
            device: String::new(),
            embed_batch_size: d_32(),
            ruri_prefix: RuriPrefix::default(),
            models: NlpModels::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuriPrefix {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub passage: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NlpModels {
    #[serde(default)]
    pub fast: Option<ModelSet>,
    #[serde(default)]
    pub balanced: Option<ModelSet>,
    #[serde(default)]
    pub quality: Option<ModelSet>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSet {
    #[serde(default)]
    pub embedding: String,
    #[serde(default)]
    pub sentiment: String,
    #[serde(default)]
    pub stance: String,
    #[serde(default)]
    pub stance_mode: String,
    #[serde(default)]
    pub stance_llm: String,
}

fn d_outdir() -> String {
    "output/comm".to_string()
}

fn d_20() -> usize {
    20
}

/// Output configuration．
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "d_outdir")]
    pub dir: String,
    #[serde(default = "d_true")]
    pub anonymize_user_ids: bool,
    #[serde(default = "d_20")]
    pub top_n: usize,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: d_outdir(),
            anonymize_user_ids: d_true(),
            top_n: d_20(),
        }
    }
}

impl CommConfig {
    /// Load configuration from a TOML file．
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use figment::providers::Format;
        figment::Figment::new()
            .merge(figment::providers::Toml::file(path.as_ref()))
            .extract()
            .map_err(|e| CommError::Figment(e.to_string()))
    }

    /// Deterministic SHA256 hex digest of the (serialized) configuration．
    ///
    /// Serde serializes struct fields in declaration order, so JSON
    /// serialization of `self` is stable across runs of the same binary．
    pub fn config_hash(&self) -> String {
        let json = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&json);
        let digest = hasher.finalize();
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            out.push_str(&format!("{:02x}", byte));
        }
        out
    }

    /// Classify a channel name into a category by the first matching entry．
    /// Invalid regex patterns are skipped with a warning．
    pub fn classify_channel(&self, name: &str) -> ChannelCategory {
        self.channel_entry_for(name)
            .map(|e| e.category)
            .unwrap_or(ChannelCategory::Unknown)
    }

    /// Return the first `ChannelEntry` whose regex pattern matches `name`．
    pub fn channel_entry_for(&self, name: &str) -> Option<&ChannelEntry> {
        for entry in &self.channels {
            match regex::Regex::new(&entry.pattern) {
                Ok(re) => {
                    if re.is_match(name) {
                        return Some(entry);
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        pattern = %entry.pattern,
                        error = %err,
                        "skipping channel entry with invalid regex"
                    );
                }
            }
        }
        None
    }

    /// Return the role for a given user id, or `Role::Unknown` if unmapped．
    pub fn role_for(&self, user_id: &str) -> Role {
        self.roles
            .iter()
            .find(|r| r.user_id == user_id)
            .map(|r| r.role)
            .unwrap_or(Role::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const EXAMPLE_TOML: &str = r#"
[period]
from = "2025-04-01"
to   = "2026-03-31"

[period.fiscal]
fiscal_year_start_month = 4

[[roles]]
user_id = "USER_ALICE"
role    = "exec"
seniority_level = 5
team    = "leadership"

[[roles]]
user_id = "USER_BOB"
role    = "manager"
seniority_level = 4
team    = "ai-team"

[[channels]]
pattern  = "^proj-.*"
category = "official"
is_decision_channel = true

[[channels]]
pattern  = "^team-leads$"
category = "leadership"
is_decision_channel = true

[[channels]]
pattern  = "^(times|random|zatsudan)-.*"
category = "casual"

[[channels]]
pattern  = "^(dev|ai|ds|sec)-.*"
category = "tech"

[patterns]
hedging = ["仮説ですが", "とりあえず"]
self_defense = ["違ったらすみません"]
surface_agreement = ["承知しました", "👍"]

[nlp_sidecar]
enabled = true
python  = "uv run python"
profile = "balanced"
device  = "mps"
embed_batch_size = 32

[nlp_sidecar.ruri_prefix]
query   = "クエリ: "
passage = "文章: "

[nlp_sidecar.models.balanced]
embedding = "cl-nagoya/ruri-v3-130m"

[output]
dir = "output/comm"
anonymize_user_ids = true
top_n = 20
"#;

    fn write_tmp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("create temp file");
        f.write_all(content.as_bytes()).expect("write temp file");
        f.flush().expect("flush temp file");
        f
    }

    #[test]
    fn test_load_and_classify_channels() {
        let f = write_tmp(EXAMPLE_TOML);
        let cfg = CommConfig::load(f.path()).expect("load config");

        assert_eq!(cfg.classify_channel("proj-x"), ChannelCategory::Official);
        assert_eq!(
            cfg.classify_channel("team-leads"),
            ChannelCategory::Leadership
        );
        assert_eq!(cfg.classify_channel("times-foo"), ChannelCategory::Casual);
        assert_eq!(cfg.classify_channel("dev-bar"), ChannelCategory::Tech);
        assert_eq!(cfg.classify_channel("nomatch"), ChannelCategory::Unknown);
    }

    #[test]
    fn test_channel_entry_decision_flag() {
        let f = write_tmp(EXAMPLE_TOML);
        let cfg = CommConfig::load(f.path()).expect("load config");

        assert!(cfg.channel_entry_for("proj-x").unwrap().is_decision_channel);
        assert!(
            !cfg.channel_entry_for("times-foo")
                .unwrap()
                .is_decision_channel
        );
        assert!(cfg.channel_entry_for("nomatch").is_none());
    }

    #[test]
    fn test_role_for() {
        let f = write_tmp(EXAMPLE_TOML);
        let cfg = CommConfig::load(f.path()).expect("load config");

        assert_eq!(cfg.role_for("USER_ALICE"), Role::Exec);
        assert_eq!(cfg.role_for("USER_BOB"), Role::Manager);
        assert_eq!(cfg.role_for("ZZZ"), Role::Unknown);
    }

    #[test]
    fn test_config_hash_is_deterministic_hex() {
        let f = write_tmp(EXAMPLE_TOML);
        let cfg1 = CommConfig::load(f.path()).expect("load config");
        let cfg2 = CommConfig::load(f.path()).expect("load config");

        let h1 = cfg1.config_hash();
        let h2 = cfg2.config_hash();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_load_with_missing_sections_uses_defaults() {
        let minimal = r#"
[[roles]]
user_id = "U1"
role    = "staff"
"#;
        let f = write_tmp(minimal);
        let cfg = CommConfig::load(f.path()).expect("load minimal config");

        assert!(!cfg.patterns.hedging.is_empty());
        assert!(cfg.patterns.incomplete.contains(&"えーと".to_string()));
        assert!(cfg.patterns.conclusion_marker.contains(&"結論".to_string()));
        assert!(cfg.nlp_sidecar.enabled);
        assert_eq!(cfg.nlp_sidecar.python, "uv run python");
        assert_eq!(cfg.nlp_sidecar.embed_batch_size, 32);
        assert_eq!(cfg.output.dir, "output/comm");
        assert_eq!(cfg.output.top_n, 20);
        assert_eq!(cfg.period.fiscal.fiscal_year_start_month, 4);
        assert_eq!(cfg.role_for("U1"), Role::Staff);
    }

    #[test]
    fn test_unknown_sections_are_tolerated() {
        let with_extra = r#"
[storage]
path = "comm.db"

[private_channel]
exclude_patterns = ["^hr-.*"]

[[roles]]
user_id = "U1"
role    = "lead"
"#;
        let f = write_tmp(with_extra);
        let cfg = CommConfig::load(f.path()).expect("load config with extra sections");
        assert_eq!(cfg.role_for("U1"), Role::Lead);
    }
}
