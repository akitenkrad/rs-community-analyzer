//! H5: exploration-ability / idea-diversity metrics (Rust-only)．
//!
//! Embedding / clustering dependent fields are left at `None` and filled by
//! [`crate::metrics::enrich_h5`] when the Python NLP sidecar is available．

use std::collections::BTreeMap;
use std::collections::HashSet;

use chrono::{DateTime, Utc};

use crate::analysis;
use crate::error::Result;
use crate::models::H5Metrics;
use crate::morphology::Morphology;
use crate::text::PatternMatcher;
use crate::types::AnalysisInput;

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn month_bucket(t: DateTime<Utc>) -> String {
    t.format("%Y-%m").to_string()
}

/// Compute H5 (exploration ability) metrics．
///
/// `morph` extracts content nouns for the monthly novel-vocabulary rate;
/// callers can use [`crate::WhitespaceMorphology`] for tests or
/// [`crate::LinderaMorphology`] (default feature) for Japanese-aware analysis．
pub fn compute_h5(input: &AnalysisInput<'_>, morph: &dyn Morphology) -> Result<H5Metrics> {
    let pm = PatternMatcher::build(&input.config.patterns)
        .map_err(|e| crate::error::CommError::Config(e.to_string()))?;

    let messages = input.messages;
    let groups = analysis::group_threads(messages);
    let threads: Vec<&analysis::ThreadGroup> =
        groups.iter().filter(|g| g.messages.len() >= 2).collect();

    let (unresolved_thread_ratio, hypothesis_retention_period_hours) = if threads.is_empty() {
        (0.0, 0.0)
    } else {
        let unresolved = threads
            .iter()
            .filter(|g| {
                !g.messages
                    .iter()
                    .any(|m| pm.contains_conclusion_marker(&m.text))
            })
            .count();
        let ratio = unresolved as f64 / threads.len() as f64;

        let mut durations: Vec<f64> = Vec::new();
        for g in &threads {
            let has_unresolved = g
                .messages
                .iter()
                .any(|m| pm.contains_unresolved_marker(&m.text));
            if !has_unresolved {
                continue;
            }
            let times: Vec<f64> = g
                .messages
                .iter()
                .map(|m| analysis::timestamp_secs_f64(m.timestamp))
                .collect();
            let (min, max) = times
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)));
            durations.push((max - min) / 3600.0);
        }
        (ratio, median(&mut durations))
    };

    // --- Novel-vocabulary rate, monthly ---
    let mut by_month: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for m in messages {
        let mk = month_bucket(m.timestamp);
        by_month.entry(mk).or_default().push(m.text.as_str());
    }

    let mut novel_vocabulary_rate_monthly: Vec<(String, f64)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (month, texts) in &by_month {
        let mut vm: HashSet<String> = HashSet::new();
        for t in texts {
            for n in morph.extract_nouns(t) {
                vm.insert(n);
            }
        }
        let rate = if vm.is_empty() {
            0.0
        } else {
            let novel = vm.iter().filter(|w| !seen.contains(*w)).count();
            novel as f64 / vm.len() as f64
        };
        novel_vocabulary_rate_monthly.push((month.clone(), rate));
        for w in vm {
            seen.insert(w);
        }
    }

    Ok(H5Metrics {
        proposal_semantic_diversity: None,
        hypothesis_retention_period_hours,
        unresolved_thread_ratio,
        novel_vocabulary_rate_monthly,
        topic_cluster_count_monthly: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CommConfig;
    use crate::morphology::WhitespaceMorphology;
    use crate::types::Message;
    use chrono::TimeZone;

    fn ts(s: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(s, 0).single().unwrap()
    }

    fn msg(
        ch: &str,
        id: &str,
        u: &str,
        text: &str,
        t: DateTime<Utc>,
        root: Option<&str>,
    ) -> Message {
        Message {
            id: id.to_string(),
            channel_id: ch.to_string(),
            author_id: u.to_string(),
            text: text.to_string(),
            timestamp: t,
            thread_root_id: root.map(String::from),
            reaction_count: 0,
        }
    }

    #[test]
    fn test_empty_messages_zero() {
        let cfg = CommConfig::default();
        let input = AnalysisInput {
            messages: &[],
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h5(&input, &WhitespaceMorphology).unwrap();
        assert_eq!(m.unresolved_thread_ratio, 0.0);
        assert!(m.novel_vocabulary_rate_monthly.is_empty());
        assert!(m.proposal_semantic_diversity.is_none());
    }

    #[test]
    fn test_unresolved_ratio_and_retention() {
        let cfg = CommConfig::default();
        let msgs = vec![
            msg("C1", "M1", "U1", "提案Aです", ts(1000), None),
            msg("C1", "M2", "U2", "結論はAで決定", ts(1500), Some("M1")),
            msg("C1", "M3", "U1", "提案Bです 保留", ts(5000), None),
            msg("C1", "M4", "U2", "まだ❓のままです", ts(8600), Some("M3")),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h5(&input, &WhitespaceMorphology).unwrap();
        assert!((m.unresolved_thread_ratio - 0.5).abs() < 1e-9);
        assert!((m.hypothesis_retention_period_hours - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_novel_vocabulary_two_months() {
        let cfg = CommConfig::default();
        let msgs = vec![
            msg(
                "C1",
                "M1",
                "U1",
                "database design meeting january",
                ts(1_735_732_800),
                None,
            ),
            msg(
                "C1",
                "M2",
                "U2",
                "database design new architecture",
                ts(1_738_454_400),
                None,
            ),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h5(&input, &WhitespaceMorphology).unwrap();
        assert_eq!(m.novel_vocabulary_rate_monthly.len(), 2);
        assert!(m.novel_vocabulary_rate_monthly[0].0 < m.novel_vocabulary_rate_monthly[1].0);
        // First month is all-novel.
        assert!((m.novel_vocabulary_rate_monthly[0].1 - 1.0).abs() < 1e-9);
        assert!((0.0..=1.0).contains(&m.novel_vocabulary_rate_monthly[1].1));
    }
}
