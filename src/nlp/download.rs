//! Pre-download model assets via `hf-hub` (design §6.6).
//!
//! These functions trigger HuggingFace downloads into `~/.cache/huggingface`
//! (honouring `HF_HOME` / `HF_HUB_OFFLINE`). They are **not** called by the
//! test suite — invoking them performs network I/O. A host CLI (e.g.
//! `persona-from-slack comm-init --download-models`) should call them.

use crate::config::{ModelSet, Profile};
use crate::error::Result;
use crate::nlp::registry::resolve_assets;

/// Download all assets required for a single `profile`.
pub fn download_models(profile: Profile) -> Result<()> {
    let ms = ModelSet::default_for(profile);
    fetch(&ms.embedding)?;
    fetch(&ms.sentiment)?;
    fetch(&ms.stance)?;
    if profile == Profile::Quality && !ms.stance_llm.is_empty() {
        fetch(&ms.stance_llm)?;
    }
    Ok(())
}

/// Download assets for every profile (`fast` + `balanced` + `quality`).
pub fn download_all() -> Result<()> {
    download_models(Profile::Fast)?;
    download_models(Profile::Balanced)?;
    download_models(Profile::Quality)?;
    Ok(())
}

fn fetch(model_id: &str) -> Result<()> {
    if model_id.is_empty() {
        return Ok(());
    }
    tracing::info!(model = model_id, "downloading model assets");
    let _ = resolve_assets(model_id)?;
    Ok(())
}
