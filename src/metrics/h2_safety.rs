//! H2: psychological-safety metrics (pattern-based, Rust-only)．

use std::collections::HashMap;

use crate::analysis::timestamp_secs_f64;
use crate::error::Result;
use crate::models::H2Metrics;
use crate::text::PatternMatcher;
use crate::types::{AnalysisInput, Message};

const INCIDENT_KEYWORDS: &[&str] = &[
    "障害",
    "インシデント",
    "エラー",
    "ダウン",
    "落ち",
    "不具合",
    "停止",
    "アラート",
    "timeout",
    "500",
];

/// Compute H2 (psychological safety) metrics over `input.messages`．
pub fn compute_h2(input: &AnalysisInput<'_>) -> Result<H2Metrics> {
    let matcher = PatternMatcher::build(&input.config.patterns)
        .map_err(|e| crate::error::CommError::Config(e.to_string()))?;

    let messages = input.messages;
    let total = messages.len();
    if total == 0 {
        return Ok(H2Metrics {
            hedging_rate: 0.0,
            self_defense_rate: 0.0,
            incomplete_utterance_rate: 0.0,
            escalation_delay_minutes: 0.0,
            dm_dependency_ratio: None,
            sample_size: 0,
        });
    }
    let total_f = total as f64;

    let mut hedging = 0usize;
    let mut self_defense = 0usize;
    let mut incomplete = 0usize;
    for m in messages {
        if matcher.count_hedging(&m.text) > 0 {
            hedging += 1;
        }
        if matcher.count_self_defense(&m.text) > 0 {
            self_defense += 1;
        }
        if matcher.count_incomplete(&m.text) > 0 {
            incomplete += 1;
        }
    }

    // --- Escalation delay (simplified heuristic) ---
    let mut threads: HashMap<(String, String), Vec<&Message>> = HashMap::new();
    for m in messages {
        let key = (
            m.channel_id.clone(),
            m.thread_root_id.clone().unwrap_or_else(|| m.id.clone()),
        );
        threads.entry(key).or_default().push(m);
    }

    let mut delays: Vec<f64> = Vec::new();
    for thread in threads.values_mut() {
        thread.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        let problem = thread.iter().find(|m| {
            let lt = m.text.to_ascii_lowercase();
            INCIDENT_KEYWORDS
                .iter()
                .any(|k| lt.contains(&k.to_ascii_lowercase()))
        });
        let Some(problem) = problem else {
            continue;
        };
        let t_problem = timestamp_secs_f64(problem.timestamp);
        let share = thread.iter().find(|m| {
            timestamp_secs_f64(m.timestamp) > t_problem && m.author_id != problem.author_id
        });
        if let Some(share) = share {
            let delay_min = (timestamp_secs_f64(share.timestamp) - t_problem) / 60.0;
            if delay_min >= 0.0 {
                delays.push(delay_min);
            }
        }
    }

    let escalation_delay_minutes = median(&mut delays);

    Ok(H2Metrics {
        hedging_rate: hedging as f64 / total_f,
        self_defense_rate: self_defense as f64 / total_f,
        incomplete_utterance_rate: incomplete as f64 / total_f,
        escalation_delay_minutes,
        dm_dependency_ratio: None,
        sample_size: total,
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CommConfig;
    use chrono::{DateTime, TimeZone, Utc};

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).single().unwrap()
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
        let m = compute_h2(&input).unwrap();
        assert_eq!(m.sample_size, 0);
        assert_eq!(m.hedging_rate, 0.0);
        assert!(m.dm_dependency_ratio.is_none());
    }

    #[test]
    fn test_rates_and_escalation() {
        let cfg = CommConfig::default();
        let msgs = vec![
            msg("C1", "M1", "U1", "とりあえずFYIで雑にメモ", ts(100), None),
            msg("C1", "M2", "U2", "障害が発生しました", ts(200), None),
            msg("C1", "M3", "U3", "確認します", ts(260), Some("M2")),
            msg("C1", "M4", "U4", "通常の発言です", ts(300), None),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h2(&input).unwrap();
        assert_eq!(m.sample_size, 4);
        assert!(m.hedging_rate > 0.0);
        // 200 -> 260 = 60s = 1 min.
        assert!((m.escalation_delay_minutes - 1.0).abs() < 1e-6);
    }
}
