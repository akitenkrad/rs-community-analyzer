//! Ruri v3 (ModernBERT-Ja) sentence embedding.
//!
//! Ruri v3 is built on ModernBERT-Ja, which candle-transformers 0.10 exposes
//! as [`candle_transformers::models::modernbert`]. We mean-pool the last hidden
//! state (attention-mask weighted), prepend the `文章: ` passage prefix (the
//! library compares symmetric messages, design §6.5), and L2-normalize.
//!
//! Tokenizer: Ruri v3 ships a HuggingFace `tokenizer.json` (fast tokenizer,
//! no MeCab pre-tokenization needed for ModernBERT-Ja), so [`tokenizers`]
//! handles it directly.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config, ModernBert};
use tokenizers::Tokenizer;

use crate::error::{CommError, Result};
use crate::nlp::registry::{resolve_assets, ResolvedConfig};

fn err(e: impl std::fmt::Display) -> CommError {
    CommError::Nlp(e.to_string())
}

/// A loaded Ruri v3 embedding model.
pub struct EmbeddingModel {
    model: ModernBert,
    tokenizer: Tokenizer,
    device: Device,
    passage_prefix: String,
    batch_size: usize,
}

impl EmbeddingModel {
    /// Load weights, config, and tokenizer for the configured embedding model.
    pub fn load(rc: &ResolvedConfig) -> Result<Self> {
        let assets = resolve_assets(&rc.embedding_id)?;
        let tokenizer_path = assets.tokenizer_json.ok_or_else(|| {
            CommError::Nlp(format!(
                "embedding model {} has no tokenizer.json (Ruri v3 expects a fast tokenizer)",
                rc.embedding_id
            ))
        })?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(err)?;

        let config_bytes = std::fs::read(&assets.config)?;
        let config: Config = serde_json::from_slice(&config_bytes)
            .map_err(|e| CommError::Nlp(format!("parse modernbert config: {e}")))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[assets.weights], DType::F32, &rc.device)
                .map_err(err)?
        };
        let model = ModernBert::load(vb, &config).map_err(err)?;

        Ok(Self {
            model,
            tokenizer,
            device: rc.device.clone(),
            passage_prefix: rc.passage_prefix.clone(),
            batch_size: rc.embed_batch_size,
        })
    }

    /// Embed `texts`, batching by the configured `embed_batch_size`.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            out.extend(self.embed_batch(chunk)?);
        }
        Ok(out)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", self.passage_prefix, t))
            .collect();

        let encodings = self.tokenizer.encode_batch(prefixed, true).map_err(err)?;
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0).max(1);

        let mut ids_rows: Vec<u32> = Vec::with_capacity(texts.len() * max_len);
        let mut mask_rows: Vec<u32> = Vec::with_capacity(texts.len() * max_len);
        for e in &encodings {
            let ids = e.get_ids();
            let att = e.get_attention_mask();
            for j in 0..max_len {
                ids_rows.push(ids.get(j).copied().unwrap_or(0));
                mask_rows.push(att.get(j).copied().unwrap_or(0));
            }
        }

        let n = texts.len();
        let input_ids = Tensor::from_vec(ids_rows, (n, max_len), &self.device).map_err(err)?;
        let attn_u32 = Tensor::from_vec(mask_rows, (n, max_len), &self.device).map_err(err)?;

        // last_hidden_state: [n, seq, hidden]
        let hidden = self.model.forward(&input_ids, &attn_u32).map_err(err)?;

        // Mean-pool over the sequence dimension, weighted by the attention mask.
        let mask_f = attn_u32.to_dtype(DType::F32).map_err(err)?; // [n, seq]
        let mask_3d = mask_f.unsqueeze(2).map_err(err)?; // [n, seq, 1]
        let masked = hidden.broadcast_mul(&mask_3d).map_err(err)?;
        let summed = masked.sum(1).map_err(err)?; // [n, hidden]
        let counts = mask_f.sum(1).map_err(err)?.unsqueeze(1).map_err(err)?; // [n, 1]
        let counts = counts.clamp(1e-9f32, f32::MAX).map_err(err)?;
        let pooled = summed.broadcast_div(&counts).map_err(err)?; // [n, hidden]

        let rows: Vec<Vec<f32>> = pooled.to_vec2().map_err(err)?;
        Ok(rows.into_iter().map(l2_normalize).collect())
    }
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}
