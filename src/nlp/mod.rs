//! Platform-agnostic NLP backend abstraction.
//!
//! The [`Nlp`] trait decouples the H1/H4/H5 enrichment layer
//! ([`crate::metrics::nlp_enrich`]) from any particular inference engine.
//! Two implementations ship:
//!
//! - [`MockNlp`] — deterministic, dependency-free, always available. It
//!   reproduces byte-for-byte the contract of the historical Python `--mock`
//!   sidecar, so tests/CI need neither model weights nor network access.
//! - [`CandleNlp`] (feature `nlp`) — real in-process inference via candle
//!   (Ruri v3 embedding, BERT-WRIME sentiment, JSNLI-BERT / Sarashina2.2
//!   stance) and HDBSCAN clustering. Models are lazily loaded and downloaded
//!   from HuggingFace on first use; the test suite never exercises them.
//!
//! All methods are **synchronous**. The previous async sidecar is gone; long
//! LLM inference (the `quality` profile) should be wrapped in
//! `tokio::task::spawn_blocking` by the caller if it runs inside an async
//! runtime.

use sha2::{Digest, Sha256};

#[cfg(feature = "nlp")]
mod bert_classifier;
#[cfg(feature = "nlp")]
mod clustering;
#[cfg(feature = "nlp")]
pub mod download;
#[cfg(feature = "nlp")]
mod embedding;
#[cfg(feature = "nlp")]
mod registry;
#[cfg(feature = "nlp")]
mod sentiment;
#[cfg(feature = "nlp")]
mod stance;

#[cfg(feature = "nlp")]
mod candle_engine;
#[cfg(feature = "nlp")]
pub use candle_engine::CandleNlp;

/// Sentiment scores for one piece of text.
///
/// `polarity` is in `[-1, 1]` (negative = unhappy, positive = happy);
/// `magnitude` is in `[0, 1]` (emotional intensity, sign-agnostic).
#[derive(Debug, Clone, Copy)]
pub struct Sentiment {
    pub polarity: f32,
    pub magnitude: f32,
}

/// Discrete stance of a message relative to its thread context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StanceLabel {
    Support,
    Neutral,
    Disagree,
    Clarify,
}

/// A stance classification result with a confidence `score` in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct Stance {
    pub label: StanceLabel,
    pub score: f32,
}

/// The outcome of clustering a set of embeddings.
///
/// `labels[i]` is the cluster id of `embeddings[i]` (a negative value denotes
/// noise / no cluster); `num_clusters` is the count of distinct non-noise
/// clusters.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Vec<i32>,
    pub num_clusters: u32,
}

/// Platform-agnostic NLP backend.
///
/// Implemented by [`MockNlp`] (tests) and [`CandleNlp`] (feature `nlp`).
pub trait Nlp {
    /// Embed each text into a fixed-length vector (one row per input).
    fn embed(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>>;
    /// Score the sentiment of a single text.
    fn sentiment(&self, text: &str) -> crate::error::Result<Sentiment>;
    /// Classify the stance of `text`, optionally given thread `context`.
    fn stance(&self, text: &str, context: Option<&str>) -> crate::error::Result<Stance>;
    /// Cluster `embeddings`, honoring a minimum cluster size of `min_cluster_size`.
    fn cluster(
        &self,
        embeddings: &[Vec<f32>],
        min_cluster_size: usize,
    ) -> crate::error::Result<ClusterResult>;
}

/// Deterministic stub backend (no models, no network).
///
/// Preserves the exact contract of the historical Python `--mock` sidecar so
/// that enrichment behaviour is reproducible in tests/CI.
#[derive(Debug, Default, Clone, Copy)]
pub struct MockNlp;

/// Dimensionality of [`MockNlp`] embeddings (matches the legacy sidecar).
const MOCK_EMBED_DIM: usize = 8;

impl Nlp for MockNlp {
    fn embed(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| mock_embed_one(t)).collect())
    }

    fn sentiment(&self, text: &str) -> crate::error::Result<Sentiment> {
        let h: u32 = text.chars().map(|c| c as u32).sum();
        let polarity = ((h % 201) as i64 - 100) as f32 / 100.0;
        let magnitude = (h % 100) as f32 / 100.0;
        Ok(Sentiment {
            polarity,
            magnitude,
        })
    }

    fn stance(&self, _text: &str, _context: Option<&str>) -> crate::error::Result<Stance> {
        Ok(Stance {
            label: StanceLabel::Neutral,
            score: 0.5,
        })
    }

    fn cluster(
        &self,
        embeddings: &[Vec<f32>],
        min_cluster_size: usize,
    ) -> crate::error::Result<ClusterResult> {
        let k = (embeddings.len() / min_cluster_size.max(1)).max(1);
        let labels: Vec<i32> = (0..embeddings.len()).map(|i| (i % k) as i32).collect();
        Ok(ClusterResult {
            labels,
            num_clusters: k as u32,
        })
    }
}

/// 8-float embedding from the first 8 SHA256 bytes of `text`, L2-normalized.
/// A zero vector degenerates to the uniform unit vector `1/sqrt(8)`.
fn mock_embed_one(text: &str) -> Vec<f32> {
    let digest = Sha256::digest(text.as_bytes());
    let mut v: Vec<f32> = digest
        .iter()
        .take(MOCK_EMBED_DIM)
        .map(|&b| b as f32)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        let u = 1.0 / (MOCK_EMBED_DIM as f32).sqrt();
        v.iter_mut().for_each(|x| *x = u);
    } else {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_sentiment_matches_python_contract() {
        // h = sum of code points; P=((h%201)-100)/100; M=(h%100)/100.
        let nlp = MockNlp;
        let text = "これはテスト";
        let h: u32 = text.chars().map(|c| c as u32).sum();
        let s = nlp.sentiment(text).unwrap();
        assert!((s.polarity - (((h % 201) as i64 - 100) as f32 / 100.0)).abs() < 1e-9);
        assert!((s.magnitude - ((h % 100) as f32 / 100.0)).abs() < 1e-9);
        assert!((-1.0..=1.0).contains(&s.polarity));
        assert!((0.0..=1.0).contains(&s.magnitude));
    }

    #[test]
    fn mock_sentiment_is_deterministic() {
        let nlp = MockNlp;
        let a = nlp.sentiment("hello").unwrap();
        let b = nlp.sentiment("hello").unwrap();
        assert_eq!(a.polarity, b.polarity);
        assert_eq!(a.magnitude, b.magnitude);
    }

    #[test]
    fn mock_embed_is_unit_8d() {
        let nlp = MockNlp;
        let vs = nlp
            .embed(&["x".to_string(), "y".to_string(), "".to_string()])
            .unwrap();
        assert_eq!(vs.len(), 3);
        for v in &vs {
            assert_eq!(v.len(), MOCK_EMBED_DIM);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "embedding must be L2-normalized");
        }
        // Empty string still yields a unit vector (uniform fallback path is only
        // hit on an all-zero digest, which never happens for SHA256; assert unit).
        let empty = &vs[2];
        let norm = empty.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mock_stance_is_neutral_half() {
        let nlp = MockNlp;
        let s = nlp.stance("そうですね", Some("提案です")).unwrap();
        assert_eq!(s.label, StanceLabel::Neutral);
        assert!((s.score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mock_cluster_contract() {
        let nlp = MockNlp;
        let embeddings: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32]).collect();
        let r = nlp.cluster(&embeddings, 2).unwrap();
        // k = max(1, 6 / 2) = 3; labels[i] = i % 3.
        assert_eq!(r.num_clusters, 3);
        assert_eq!(r.labels, vec![0, 1, 2, 0, 1, 2]);

        // min_cluster_size 0 must not divide-by-zero (treated as 1).
        let r0 = nlp.cluster(&embeddings, 0).unwrap();
        assert_eq!(r0.num_clusters, 6);

        // Empty input => k=max(1,0)=1, no labels.
        let re = nlp.cluster(&[], 4).unwrap();
        assert_eq!(re.num_clusters, 1);
        assert!(re.labels.is_empty());
    }
}
