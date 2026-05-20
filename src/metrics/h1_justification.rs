//! H1: exploration-vs-justification metrics (pattern + time based, Rust-only)．
//!
//! Two embedding-dependent fields (`decision_change_rate`,
//! `initial_proposal_regression`) are left at `0.0` here and filled later by
//! [`crate::metrics::enrich_h1`] when the Python NLP sidecar is available．
//!
//! ## Role resolution
//!
//! For a user id `uid` the organizational role is resolved from
//! `input.users` (the caller's pre-resolved record) first, falling back to
//! `cfg.role_for(uid)`．"managerial" = role ∈ {Manager, Exec, Lead};
//! "staff" = role == Staff．`Unknown` enters neither denominator．

use crate::analysis::{group_threads, timestamp_secs_f64};
use crate::error::Result;
use crate::models::{H1Metrics, Role, SilenceRatio};
use crate::text::PatternMatcher;
use crate::types::AnalysisInput;

/// Window (seconds) within which a same-thread later message counts as a
/// "reply" for the silence-rate computation. 24h．
const SILENCE_WINDOW_SECS: f64 = 86_400.0;

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

fn resolve_role(input: &AnalysisInput<'_>, uid: &str) -> Role {
    if let Some(u) = input.users.iter().find(|u| u.id == uid) {
        if u.role != Role::Unknown {
            return u.role;
        }
    }
    input.config.role_for(uid)
}

fn is_managerial(r: Role) -> bool {
    matches!(r, Role::Manager | Role::Exec | Role::Lead)
}

fn is_staff(r: Role) -> bool {
    matches!(r, Role::Staff)
}

/// Compute H1 (exploration-vs-justification) metrics over `input.messages`．
pub fn compute_h1(input: &AnalysisInput<'_>) -> Result<H1Metrics> {
    let pm = PatternMatcher::build(&input.config.patterns)
        .map_err(|e| crate::error::CommError::Config(e.to_string()))?;

    let groups = group_threads(input.messages);

    // --- silence_after_manager ---
    let mut mgr_total = 0u64;
    let mut mgr_silent = 0u64;
    let mut staff_total = 0u64;
    let mut staff_silent = 0u64;
    for g in &groups {
        for (i, m) in g.messages.iter().enumerate() {
            let t = timestamp_secs_f64(m.timestamp);
            let has_reply = g.messages.iter().enumerate().any(|(j, n)| {
                if j == i {
                    return false;
                }
                let dt = timestamp_secs_f64(n.timestamp) - t;
                dt > 0.0 && dt <= SILENCE_WINDOW_SECS
            });
            let role = resolve_role(input, &m.author_id);
            if is_managerial(role) {
                mgr_total += 1;
                if !has_reply {
                    mgr_silent += 1;
                }
            } else if is_staff(role) {
                staff_total += 1;
                if !has_reply {
                    staff_silent += 1;
                }
            }
        }
    }
    let manager_post_silence_rate = if mgr_total == 0 {
        0.0
    } else {
        mgr_silent as f64 / mgr_total as f64
    };
    let staff_post_silence_rate = if staff_total == 0 {
        0.0
    } else {
        staff_silent as f64 / staff_total as f64
    };
    let ratio = if staff_post_silence_rate == 0.0 {
        0.0
    } else {
        manager_post_silence_rate / staff_post_silence_rate
    };

    // --- dissent_convergence_speed_minutes (pattern fallback) ---
    let mut conv: Vec<f64> = Vec::new();
    for g in &groups {
        let mut dissent_ts: Vec<f64> = g
            .messages
            .iter()
            .filter(|m| pm.count_dissent(&m.text) > 0)
            .map(|m| timestamp_secs_f64(m.timestamp))
            .collect();
        if dissent_ts.is_empty() {
            continue;
        }
        dissent_ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let first = dissent_ts[0];
        let last = *dissent_ts.last().unwrap();
        conv.push((last - first) / 60.0);
    }
    let dissent_convergence_speed_minutes = median(&mut conv);

    // --- thread_count ---
    let thread_count = groups.iter().filter(|g| g.messages.len() >= 2).count();

    Ok(H1Metrics {
        decision_change_rate: 0.0,
        initial_proposal_regression: 0.0,
        silence_after_manager: SilenceRatio {
            manager_post_silence_rate,
            staff_post_silence_rate,
            ratio,
        },
        dissent_convergence_speed_minutes,
        thread_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CommConfig;
    use crate::types::{Message, User};
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

    fn user(id: &str, role: Role) -> User {
        User {
            id: id.to_string(),
            display_name: id.to_string(),
            role,
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
        let m = compute_h1(&input).unwrap();
        assert_eq!(m.decision_change_rate, 0.0);
        assert_eq!(m.initial_proposal_regression, 0.0);
        assert_eq!(m.silence_after_manager.manager_post_silence_rate, 0.0);
        assert_eq!(m.silence_after_manager.staff_post_silence_rate, 0.0);
        assert_eq!(m.silence_after_manager.ratio, 0.0);
        assert_eq!(m.dissent_convergence_speed_minutes, 0.0);
        assert_eq!(m.thread_count, 0);
    }

    #[test]
    fn test_silence_rates_and_ratio() {
        let cfg = CommConfig::default();
        let users = vec![user("MGR1", Role::Manager), user("STF1", Role::Staff)];
        let msgs = vec![
            msg("C1", "M1", "MGR1", "提案A", ts(100), None),
            msg("C1", "M2", "STF1", "了解", ts(200), Some("M1")),
            msg("C1", "M3", "MGR1", "単独投稿", ts(5000), None),
            msg("C1", "M4", "STF1", "質問", ts(8000), None),
            msg("C1", "M5", "MGR1", "回答", ts(8100), Some("M4")),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &users,
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h1(&input).unwrap();
        assert!((m.silence_after_manager.manager_post_silence_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((m.silence_after_manager.staff_post_silence_rate - 0.5).abs() < 1e-9);
        assert!((m.silence_after_manager.ratio - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_dissent_convergence_and_thread_count() {
        let cfg = CommConfig::default();
        let msgs = vec![
            msg("C1", "M1", "U1", "提案します", ts(100), None),
            msg("C1", "M2", "U2", "ただ、反対です", ts(200), Some("M1")),
            msg(
                "C1",
                "M3",
                "U3",
                "しかし懸念があります",
                ts(2000),
                Some("M1"),
            ),
            msg("C2", "M4", "U1", "別スレ root", ts(5000), None),
            msg("C2", "M5", "U2", "別スレ reply", ts(5100), Some("M4")),
        ];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h1(&input).unwrap();
        // first dissent @200, last @2000 -> 30.0 min.
        assert!((m.dissent_convergence_speed_minutes - 30.0).abs() < 1e-6);
        assert_eq!(m.thread_count, 2);
    }
}
