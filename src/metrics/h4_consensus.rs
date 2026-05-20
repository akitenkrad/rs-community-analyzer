//! H4: pseudo-consensus / hidden-dissent metrics (pattern-based, Rust-only)．

use std::collections::HashMap;

use crate::analysis::timestamp_secs_f64;
use crate::error::Result;
use crate::models::{ChannelCategory, H4Metrics};
use crate::text::PatternMatcher;
use crate::types::{AnalysisInput, Message};

const WINDOW_HOURS: f64 = 48.0;

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

/// Compute H4 (pseudo-consensus) metrics over `input.messages`．
pub fn compute_h4(input: &AnalysisInput<'_>) -> Result<H4Metrics> {
    let pm = PatternMatcher::build(&input.config.patterns)
        .map_err(|e| crate::error::CommError::Config(e.to_string()))?;

    let messages = input.messages;
    let total = messages.len() as f64;
    if total == 0.0 {
        return Ok(H4Metrics {
            surface_agreement_rate: 0.0,
            public_private_sentiment_delta: None,
            post_meeting_dissent_rate: 0.0,
            execution_delay_hours: 0.0,
            reaction_text_disagreement: None,
        });
    }

    // --- Surface-agreement rate ---
    let surface_count = messages
        .iter()
        .filter(|m| pm.contains_surface_agreement(&m.text))
        .count();
    let surface_agreement_rate = surface_count as f64 / total;

    // --- Execution delay ---
    let mut by_user: HashMap<&str, Vec<&Message>> = HashMap::new();
    for m in messages {
        by_user.entry(m.author_id.as_str()).or_default().push(m);
    }
    let mut delays: Vec<f64> = Vec::new();
    for msgs in by_user.values_mut() {
        msgs.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        for (i, m) in msgs.iter().enumerate() {
            if !pm.contains_commitment(&m.text) {
                continue;
            }
            let commit_secs = timestamp_secs_f64(m.timestamp);
            if let Some(action) = msgs[i + 1..].iter().find(|n| {
                timestamp_secs_f64(n.timestamp) > commit_secs && pm.contains_action_mention(&n.text)
            }) {
                let delay_h = (timestamp_secs_f64(action.timestamp) - commit_secs) / 3600.0;
                if delay_h >= 0.0 {
                    delays.push(delay_h);
                }
            }
        }
    }
    let execution_delay_hours = median(&mut delays);

    // --- Post-meeting dissent rate (Rust-only pattern fallback) ---
    let id_to_name: HashMap<&str, &str> = input
        .channels
        .iter()
        .map(|c| (c.id.as_str(), c.name.as_str()))
        .collect();
    let channel_by_id: HashMap<&str, &crate::types::Channel> =
        input.channels.iter().map(|c| (c.id.as_str(), c)).collect();
    let cfg = input.config;
    let is_decision = |cid: &str| -> bool {
        // Caller-provided override on Channel struct takes precedence; otherwise
        // fall back to (a) config channel_entry_for(name) and (b) category.
        if let Some(c) = channel_by_id.get(cid) {
            if c.is_decision_channel {
                return true;
            }
            if matches!(
                c.category,
                Some(ChannelCategory::Leadership) | Some(ChannelCategory::Official)
            ) {
                return true;
            }
        }
        match id_to_name.get(cid) {
            Some(name) => {
                let flagged = cfg
                    .channel_entry_for(name)
                    .map(|e| e.is_decision_channel)
                    .unwrap_or(false);
                let cat = cfg.classify_channel(name);
                flagged || matches!(cat, ChannelCategory::Leadership | ChannelCategory::Official)
            }
            None => false,
        }
    };

    let mut agreement_events = 0u64;
    let mut dissent_positive = 0u64;
    for m in messages {
        if !is_decision(&m.channel_id) || !pm.contains_surface_agreement(&m.text) {
            continue;
        }
        agreement_events += 1;
        let t = timestamp_secs_f64(m.timestamp);
        let positive = messages.iter().any(|n| {
            n.author_id == m.author_id
                && n.channel_id != m.channel_id
                && {
                    let dt = timestamp_secs_f64(n.timestamp) - t;
                    dt > 0.0 && dt <= WINDOW_HOURS * 3600.0
                }
                && pm.count_dissent(&n.text) > 0
        });
        if positive {
            dissent_positive += 1;
        }
    }
    let post_meeting_dissent_rate = if agreement_events == 0 {
        0.0
    } else {
        dissent_positive as f64 / agreement_events as f64
    };

    Ok(H4Metrics {
        surface_agreement_rate,
        public_private_sentiment_delta: None,
        post_meeting_dissent_rate,
        execution_delay_hours,
        reaction_text_disagreement: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChannelEntry, CommConfig};
    use crate::types::Channel;
    use chrono::{DateTime, TimeZone, Utc};

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

    fn channel(id: &str, name: &str) -> Channel {
        Channel {
            id: id.to_string(),
            name: name.to_string(),
            category: None,
            is_decision_channel: false,
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
        let m = compute_h4(&input).unwrap();
        assert_eq!(m.surface_agreement_rate, 0.0);
        assert_eq!(m.execution_delay_hours, 0.0);
        assert_eq!(m.post_meeting_dissent_rate, 0.0);
        assert!(m.public_private_sentiment_delta.is_none());
        assert!(m.reaction_text_disagreement.is_none());
    }

    #[test]
    fn test_execution_delay_and_surface_rate() {
        let cfg = CommConfig::default();
        let msgs = vec![
            msg("C1", "M1", "U1", "対応します", ts(1000), None),
            msg("C1", "M2", "U2", "承知しました", ts(1100), None),
            msg("C1", "M3", "U1", "修正しました", ts(8200), None),
            msg("C1", "M4", "U3", "ただの雑談", ts(1200), None),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h4(&input).unwrap();
        assert!((m.execution_delay_hours - 2.0).abs() < 1e-6);
        assert!((m.surface_agreement_rate - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_post_meeting_dissent_cross_channel() {
        let mut cfg = CommConfig::default();
        cfg.channels.push(ChannelEntry {
            pattern: "^decisions$".to_string(),
            category: ChannelCategory::Official,
            is_decision_channel: true,
        });
        let channels = vec![channel("C_DEC", "decisions"), channel("C_DM", "random")];
        let msgs = vec![
            msg("C_DEC", "M1", "U1", "承知しました", ts(10000), None),
            msg(
                "C_DM",
                "M2",
                "U1",
                "ただ、しかし懸念があります",
                ts(13600),
                None,
            ),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &channels,
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h4(&input).unwrap();
        assert!((m.post_meeting_dissent_rate - 1.0).abs() < 1e-9);
    }
}
