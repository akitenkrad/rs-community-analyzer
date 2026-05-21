//! Model registry: device selection, HuggingFace asset resolution, and lazy
//! per-model caches.
//!
//! All weights are loaded lazily (on first use) and memory-mapped via
//! [`candle_nn::VarBuilder::from_mmaped_safetensors`]. Assets are resolved
//! through `hf-hub`, which downloads to `~/.cache/huggingface` on first use
//! and honours `HF_HOME` / `HF_HUB_OFFLINE`. No network access happens at
//! construction time — only when a model is first exercised.

use std::path::PathBuf;

use candle_core::Device;
use hf_hub::api::sync::Api;

use crate::config::{NlpConfig, Profile};
use crate::error::{CommError, Result};

/// Select the candle device from a config hint.
///
/// This session compiles only the CPU backend (`metal` / `cuda` candle
/// features are intentionally off), so any GPU hint degrades to CPU with a
/// warning. The features can be enabled later for real GPU inference.
pub fn select_device(hint: &str) -> Device {
    match hint.trim().to_ascii_lowercase().as_str() {
        "" | "cpu" => Device::Cpu,
        other => {
            tracing::warn!(
                device = other,
                "candle GPU backends (metal/cuda) are not compiled in this build; using CPU"
            );
            Device::Cpu
        }
    }
}

fn err(e: impl std::fmt::Display) -> CommError {
    CommError::Nlp(e.to_string())
}

/// A resolved set of local file paths for one HuggingFace model repo.
pub struct ModelAssets {
    /// `model.safetensors` (single-file weights).
    pub weights: PathBuf,
    /// `config.json`.
    pub config: PathBuf,
    /// `tokenizer.json` if present, else `None` (caller falls back to `vocab.txt`).
    pub tokenizer_json: Option<PathBuf>,
    /// `vocab.txt` if present (WordPiece vocab for cl-tohoku BERT).
    pub vocab_txt: Option<PathBuf>,
}

/// Resolve (download if missing) the standard asset files for `repo_id`.
///
/// `model_id` may carry a `@revision` suffix to pin a commit (design §6.6.5).
pub fn resolve_assets(model_id: &str) -> Result<ModelAssets> {
    let (repo_id, revision) = match model_id.split_once('@') {
        Some((id, rev)) => (id.to_string(), Some(rev.to_string())),
        None => (model_id.to_string(), None),
    };
    let api = Api::new().map_err(err)?;
    let repo = match revision {
        Some(rev) => api.repo(hf_hub::Repo::with_revision(
            repo_id,
            hf_hub::RepoType::Model,
            rev,
        )),
        None => api.model(repo_id),
    };

    let weights = repo.get("model.safetensors").map_err(err)?;
    let config = repo.get("config.json").map_err(err)?;
    // Optional files: a missing one is not fatal.
    let tokenizer_json = repo.get("tokenizer.json").ok();
    let vocab_txt = repo.get("vocab.txt").ok();

    Ok(ModelAssets {
        weights,
        config,
        tokenizer_json,
        vocab_txt,
    })
}

/// The active profile's resolved configuration view.
pub struct ResolvedConfig {
    /// Active profile (informational; the per-model ids below are already resolved).
    #[allow(dead_code)]
    pub profile: Profile,
    pub device: Device,
    pub embedding_id: String,
    pub sentiment_id: String,
    pub stance_id: String,
    pub stance_mode: String,
    pub stance_llm_id: String,
    pub passage_prefix: String,
    pub embed_batch_size: usize,
}

impl ResolvedConfig {
    pub fn from(cfg: &NlpConfig) -> Self {
        let ms = cfg.model_set();
        // The library compares symmetric messages, so it always uses the
        // "passage" prefix (design §6.5). Default to "文章: " when unset.
        let passage_prefix = if cfg.ruri_prefix.passage.is_empty() {
            "文章: ".to_string()
        } else {
            cfg.ruri_prefix.passage.clone()
        };
        Self {
            profile: cfg.profile(),
            device: select_device(&cfg.device),
            embedding_id: ms.embedding,
            sentiment_id: ms.sentiment,
            stance_id: ms.stance,
            stance_mode: if ms.stance_mode.is_empty() {
                "nli".to_string()
            } else {
                ms.stance_mode
            },
            stance_llm_id: ms.stance_llm,
            passage_prefix,
            embed_batch_size: cfg.embed_batch_size.max(1),
        }
    }
}
