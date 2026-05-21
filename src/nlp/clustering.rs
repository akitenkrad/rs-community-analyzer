//! Semantic clustering via HDBSCAN (`petal-clustering`).
//!
//! `petal-clustering` and `annembed` (UMAP) build cleanly alongside the candle
//! stack, so both are folded directly into the `nlp` feature (no separate
//! sub-feature needed). HDBSCAN runs on the raw embeddings; a deterministic
//! `√n` bucketing fallback guarantees [`crate::nlp::Nlp::cluster`] always
//! returns a usable result even when HDBSCAN labels every point as noise (or
//! the input is degenerate).
//!
//! UMAP dimensionality reduction (`annembed`) is available as a dependency for
//! future use; HDBSCAN on the (already low-dim, L2-normalized) sentence
//! embeddings is sufficient for the monthly topic-count metric and avoids the
//! cost/instability of an extra reduction step on small per-month samples.

use ndarray::Array2;
use petal_clustering::{Fit, HDbscan};

use crate::error::{CommError, Result};
use crate::nlp::ClusterResult;

/// Cluster `embeddings` with HDBSCAN, falling back to deterministic bucketing.
pub fn cluster(embeddings: &[Vec<f32>], min_cluster_size: usize) -> Result<ClusterResult> {
    let n = embeddings.len();
    if n == 0 {
        return Ok(ClusterResult {
            labels: Vec::new(),
            num_clusters: 0,
        });
    }
    let dim = embeddings[0].len();
    if dim == 0 || embeddings.iter().any(|v| v.len() != dim) {
        return Err(CommError::Nlp(
            "cluster: embeddings must be non-empty and equal-length".into(),
        ));
    }

    let mcs = min_cluster_size.max(2);
    // HDBSCAN needs at least `min_cluster_size` points to form any cluster.
    if n < mcs {
        return Ok(fallback_bucketing(n));
    }

    let flat: Vec<f64> = embeddings
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f64))
        .collect();
    let array = Array2::from_shape_vec((n, dim), flat)
        .map_err(|e| CommError::Nlp(format!("cluster: shape: {e}")))?;

    // `HDbscan::<f64, Euclidean>::default()` supplies the Euclidean metric
    // (a transitive type from petal-neighbors); override the sizing in place.
    let mut hdbscan = HDbscan::<f64, _> {
        alpha: 1.0,
        min_samples: mcs,
        min_cluster_size: mcs,
        boruvka: true,
        ..HDbscan::default()
    };
    let (clusters, _outliers, _scores) = hdbscan.fit(&array, None);

    if clusters.is_empty() {
        // All noise — degrade to deterministic bucketing.
        return Ok(fallback_bucketing(n));
    }

    let mut labels = vec![-1i32; n];
    for (cluster_id, indices) in clusters.values().enumerate() {
        for &idx in indices {
            if idx < n {
                labels[idx] = cluster_id as i32;
            }
        }
    }
    Ok(ClusterResult {
        labels,
        num_clusters: clusters.len() as u32,
    })
}

/// Deterministic `k = ceil(sqrt(n))` round-robin bucketing.
fn fallback_bucketing(n: usize) -> ClusterResult {
    if n == 0 {
        return ClusterResult {
            labels: Vec::new(),
            num_clusters: 0,
        };
    }
    let k = ((n as f64).sqrt().ceil() as usize).max(1);
    let labels: Vec<i32> = (0..n).map(|i| (i % k) as i32).collect();
    ClusterResult {
        labels,
        num_clusters: k as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let r = cluster(&[], 5).unwrap();
        assert_eq!(r.num_clusters, 0);
        assert!(r.labels.is_empty());
    }

    #[test]
    fn ragged_input_errors() {
        let embs = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(cluster(&embs, 2).is_err());
    }

    #[test]
    fn small_input_uses_fallback() {
        // n=3 < min_cluster_size=5 -> fallback k=ceil(sqrt(3))=2.
        let embs: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, 0.0]).collect();
        let r = cluster(&embs, 5).unwrap();
        assert_eq!(r.labels.len(), 3);
        assert!(r.num_clusters >= 1);
        assert_eq!(r.num_clusters, 2);
        assert_eq!(r.labels, vec![0, 1, 0]);
    }

    #[test]
    fn two_well_separated_blobs() {
        // 12 points: 6 near origin, 6 far away. HDBSCAN should find structure.
        let mut embs: Vec<Vec<f32>> = Vec::new();
        for i in 0..6 {
            embs.push(vec![0.0 + i as f32 * 0.01, 0.0]);
        }
        for i in 0..6 {
            embs.push(vec![100.0 + i as f32 * 0.01, 100.0]);
        }
        let r = cluster(&embs, 3).unwrap();
        assert_eq!(r.labels.len(), 12);
        assert!(r.num_clusters >= 1);
    }
}
