//! [`CandleNlp`]: in-process candle inference backend (feature `nlp`).
//!
//! Each model is loaded lazily on first use via [`OnceCell`] and memory-mapped
//! (no eager 6 GB load). Construction (`CandleNlp::new`) does **no** network I/O
//! and **no** model loading — it only resolves config. The first call to each
//! method triggers download (if absent) + load of that model.
//!
//! `&self` methods are safe to call concurrently: candle tensor ops execute
//! sequentially under the hood, so no external locking is needed (design §6.3).
//! Long LLM inference (the `quality` profile) should be wrapped in
//! `tokio::task::spawn_blocking` by an async caller.

use once_cell::sync::OnceCell;

use crate::config::NlpConfig;
use crate::error::Result;
use crate::nlp::clustering;
use crate::nlp::embedding::EmbeddingModel;
use crate::nlp::registry::ResolvedConfig;
use crate::nlp::sentiment::SentimentModel;
use crate::nlp::stance::StanceModel;
use crate::nlp::{ClusterResult, Nlp, Sentiment, Stance};

/// Candle-backed NLP engine. Construct with [`CandleNlp::new`].
pub struct CandleNlp {
    rc: ResolvedConfig,
    embedder: OnceCell<EmbeddingModel>,
    sentiment: OnceCell<SentimentModel>,
    stance: OnceCell<StanceModel>,
}

impl CandleNlp {
    /// Resolve configuration for the active profile. Models load lazily on
    /// first use; this does no network or disk I/O for weights.
    pub fn new(cfg: &NlpConfig) -> Result<Self> {
        Ok(Self {
            rc: ResolvedConfig::from(cfg),
            embedder: OnceCell::new(),
            sentiment: OnceCell::new(),
            stance: OnceCell::new(),
        })
    }

    fn embedder(&self) -> Result<&EmbeddingModel> {
        self.embedder
            .get_or_try_init(|| EmbeddingModel::load(&self.rc))
    }

    fn sentiment_model(&self) -> Result<&SentimentModel> {
        self.sentiment
            .get_or_try_init(|| SentimentModel::load(&self.rc))
    }

    fn stance_model(&self) -> Result<&StanceModel> {
        self.stance.get_or_try_init(|| StanceModel::load(&self.rc))
    }
}

impl Nlp for CandleNlp {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.embedder()?.embed(texts)
    }

    fn sentiment(&self, text: &str) -> Result<Sentiment> {
        self.sentiment_model()?.sentiment(text)
    }

    fn stance(&self, text: &str, context: Option<&str>) -> Result<Stance> {
        self.stance_model()?.stance(text, context)
    }

    fn cluster(&self, embeddings: &[Vec<f32>], min_cluster_size: usize) -> Result<ClusterResult> {
        clustering::cluster(embeddings, min_cluster_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_does_no_io_and_clusters_without_models() {
        // Construction must not load weights or touch the network.
        let cfg = NlpConfig::default();
        let engine = CandleNlp::new(&cfg).expect("construct CandleNlp");

        // `cluster` is model-free, so it works with synthetic embeddings even
        // though no weights are present.
        let embs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i % 3) as f32]).collect();
        let r = engine.cluster(&embs, 3).expect("cluster");
        assert_eq!(r.labels.len(), 10);
        assert!(r.num_clusters >= 1);

        // Empty embed input short-circuits without loading the embedder.
        assert!(engine.embed(&[]).unwrap().is_empty());
    }

    /// Real-model smoke test. Requires HuggingFace weights (network on first
    /// run); never exercised by CI. Run manually with:
    /// `cargo test --all-features -- --ignored real_model_embed`.
    #[test]
    #[ignore = "requires model weights / network (hf-hub download)"]
    fn real_model_embed() {
        let engine = CandleNlp::new(&NlpConfig::default()).unwrap();
        let v = engine.embed(&["これはテストです".to_string()]).unwrap();
        assert_eq!(v.len(), 1);
        assert!(!v[0].is_empty());
    }
}
