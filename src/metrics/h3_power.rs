//! H3: power-concentration / hierarchy metrics (graph-based, Rust-only)．

use std::collections::HashMap;

use crate::error::Result;
use crate::graph::ReplyGraph;
use crate::models::{DissentBias, H3Metrics, Role};
use crate::text::PatternMatcher;
use crate::types::AnalysisInput;

/// Emoji names treated as "agreement" reactions (compared case-insensitively,
/// surrounding `:` stripped)．
const AGREEMENT_EMOJIS: &[&str] = &[
    "+1",
    "thumbsup",
    "ok",
    "ok_hand",
    "white_check_mark",
    "raised_hands",
    "clap",
    "pray",
    "100",
    "heavy_check_mark",
    "👍",
    "🙆",
    "👏",
];

fn is_managerial(role: Role) -> bool {
    matches!(role, Role::Manager | Role::Exec | Role::Lead)
}

fn normalize_emoji(e: &str) -> String {
    e.trim().trim_matches(':').to_ascii_lowercase()
}

fn is_agreement(emoji: &str) -> bool {
    let n = normalize_emoji(emoji);
    AGREEMENT_EMOJIS.iter().any(|a| a.to_ascii_lowercase() == n)
}

fn entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let tf = total as f64;
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / tf;
            p * p.log2()
        })
        .sum::<f64>()
}

fn gini(values: &[f64]) -> f64 {
    let n = values.len();
    let sum: f64 = values.iter().sum();
    if n == 0 || sum == 0.0 {
        return 0.0;
    }
    let mut numer = 0.0;
    for &xi in values {
        for &xj in values {
            numer += (xi - xj).abs();
        }
    }
    let g = numer / (2.0 * n as f64 * sum);
    g.clamp(0.0, 1.0)
}

/// Compute H3 (power concentration) metrics over `input.messages`．
///
/// Channel iteration uses `input.channels` (no DB)．Manager-reaction rate
/// joins `input.reactions` with the in-memory user role map．
pub fn compute_h3(input: &AnalysisInput<'_>) -> Result<H3Metrics> {
    let matcher = PatternMatcher::build(&input.config.patterns)
        .map_err(|e| crate::error::CommError::Config(e.to_string()))?;
    let graph = ReplyGraph::build(input.messages, input.config);

    // Build a (channel_id -> name) map from input.channels for output labelling.
    let id_to_name: HashMap<&str, &str> = input
        .channels
        .iter()
        .map(|c| (c.id.as_str(), c.name.as_str()))
        .collect();

    // --- Per-channel utterance entropy / normalized concentration ---
    let mut by_channel: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
    for m in input.messages {
        *by_channel
            .entry(m.channel_id.as_str())
            .or_default()
            .entry(m.author_id.as_str())
            .or_insert(0) += 1;
    }

    // Iterate channels in scope (those that appear in input.channels) plus any
    // channels referenced only by messages but not declared (fallback to id).
    let mut scope_ids: Vec<&str> = input.channels.iter().map(|c| c.id.as_str()).collect();
    // Stable order: declared channels first, then any extras encountered.
    let declared: std::collections::HashSet<&str> = scope_ids.iter().copied().collect();
    let mut extras: Vec<&str> = by_channel
        .keys()
        .copied()
        .filter(|id| !declared.contains(id))
        .collect();
    extras.sort();
    scope_ids.extend(extras);

    let mut utterance_entropy_by_channel: Vec<(String, f64)> = Vec::new();
    let mut normalized_concentration: Vec<(String, f64)> = Vec::new();
    for cid in &scope_ids {
        let Some(users) = by_channel.get(*cid) else {
            continue;
        };
        let counts: Vec<usize> = users.values().copied().collect();
        if counts.iter().sum::<usize>() == 0 {
            continue;
        }
        let h = entropy(&counts);
        let label = id_to_name
            .get(cid)
            .map(|s| (*s).to_string())
            .unwrap_or_else(|| (*cid).to_string());
        utterance_entropy_by_channel.push((label.clone(), h));

        let n = counts.len();
        let eta = if n < 2 {
            0.0
        } else {
            (1.0 - h / (n as f64).log2()).clamp(0.0, 1.0)
        };
        normalized_concentration.push((label, eta));
    }

    // --- Reply concentration Gini ---
    let received: Vec<f64> = graph
        .replies_received()
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    let reply_concentration_gini = gini(&received);

    // --- Manager reaction rate ---
    //
    // Build (message_id -> author_role) from messages + users + cfg, then
    // count agreement-emoji reactions on managerial-authored messages.
    let role_for = |uid: &str| -> Role {
        if let Some(u) = input.users.iter().find(|u| u.id == uid) {
            if u.role != Role::Unknown {
                return u.role;
            }
        }
        input.config.role_for(uid)
    };

    let mut msg_author: HashMap<&str, &str> = HashMap::new();
    for m in input.messages {
        msg_author.insert(m.id.as_str(), m.author_id.as_str());
    }

    let mut mgr_msg_total: u64 = 0;
    for m in input.messages {
        if is_managerial(role_for(&m.author_id)) {
            mgr_msg_total += 1;
        }
    }
    let mut agree_total: u64 = 0;
    for r in input.reactions {
        let Some(author) = msg_author.get(r.message_id.as_str()) else {
            continue;
        };
        if !is_managerial(role_for(author)) {
            continue;
        }
        if r.user_id == *author {
            continue; // self-reaction ignored
        }
        if is_agreement(&r.emoji_name) {
            agree_total += 1;
        }
    }
    let manager_reaction_rate = if mgr_msg_total == 0 {
        0.0
    } else {
        agree_total as f64 / mgr_msg_total as f64
    };

    // --- Dissent target bias ---
    let mut root_author: HashMap<(&str, &str), &str> = HashMap::new();
    for m in input.messages {
        root_author.insert((m.channel_id.as_str(), m.id.as_str()), m.author_id.as_str());
    }
    let mut tot_to_mgr = 0u64;
    let mut tot_to_staff = 0u64;
    let mut dissent_to_mgr = 0u64;
    let mut dissent_to_staff = 0u64;
    for m in input.messages {
        let Some(root_id) = m.thread_root_id.as_deref() else {
            continue;
        };
        if root_id == m.id {
            continue;
        }
        let Some(&root) = root_author.get(&(m.channel_id.as_str(), root_id)) else {
            continue;
        };
        let parent_role = role_for(root);
        let has_dissent = matcher.count_dissent(&m.text) > 0;
        if is_managerial(parent_role) {
            tot_to_mgr += 1;
            if has_dissent {
                dissent_to_mgr += 1;
            }
        } else if parent_role == Role::Staff {
            tot_to_staff += 1;
            if has_dissent {
                dissent_to_staff += 1;
            }
        }
    }
    let to_manager = if tot_to_mgr == 0 {
        0.0
    } else {
        dissent_to_mgr as f64 / tot_to_mgr as f64
    };
    let to_staff = if tot_to_staff == 0 {
        0.0
    } else {
        dissent_to_staff as f64 / tot_to_staff as f64
    };
    let ratio_staff_to_manager = if to_manager > 0.0 {
        to_staff / to_manager
    } else {
        0.0
    };

    // --- Hierarchy score ---
    let hierarchy_score = graph.hierarchy_score();

    // --- PageRank top-k managerial share ---
    let pr = graph.pagerank(0.85, 100);
    let pagerank_topk_manager_share = if pr.is_empty() {
        0.0
    } else {
        let k = ((pr.len() as f64) * 0.10).ceil() as usize;
        let k = k.max(1);
        let topk = &pr[..k.min(pr.len())];
        let mgr = topk
            .iter()
            .filter(|(uid, _)| graph.node_role(uid).map(is_managerial).unwrap_or(false))
            .count();
        mgr as f64 / k as f64
    };

    Ok(H3Metrics {
        utterance_entropy_by_channel,
        normalized_concentration,
        reply_concentration_gini,
        manager_reaction_rate,
        dissent_target_bias: DissentBias {
            to_manager,
            to_staff,
            ratio_staff_to_manager,
        },
        hierarchy_score,
        pagerank_topk_manager_share,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CommConfig, RoleEntry};
    use crate::types::{Channel, Message};
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
    fn test_entropy_balanced_and_skewed() {
        assert!((entropy(&[5, 5]) - 1.0).abs() < 1e-9);
        assert!((entropy(&[10]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gini_bounds() {
        assert_eq!(gini(&[]), 0.0);
        assert_eq!(gini(&[0.0, 0.0]), 0.0);
        let g = gini(&[0.0, 0.0, 10.0]);
        assert!((0.0..=1.0).contains(&g));
    }

    #[test]
    fn test_compute_h3_bounds() {
        let mut cfg = CommConfig::default();
        cfg.roles.push(RoleEntry {
            user_id: "MGR".to_string(),
            role: Role::Manager,
            seniority_level: 4,
            team: None,
        });
        let msgs = vec![
            msg("C1", "M0", "MGR", "ルートメッセージ", ts(100), Some("M0")),
            msg(
                "C1",
                "M1",
                "S1",
                "しかし別の見方があります",
                ts(110),
                Some("M0"),
            ),
            msg("C1", "M2", "S2", "確認しました", ts(120), Some("M0")),
            msg("C1", "M3", "S1", "通常発言", ts(130), None),
        ];
        let channels = vec![channel("C1", "general")];
        let input = AnalysisInput {
            messages: &msgs,
            channels: &channels,
            users: &[],
            reactions: &[],
            config: &cfg,
        };
        let m = compute_h3(&input).unwrap();
        assert!((0.0..=1.0).contains(&m.hierarchy_score));
        assert!((0.0..=1.0).contains(&m.pagerank_topk_manager_share));
        assert!((0.0..=1.0).contains(&m.reply_concentration_gini));
        assert!((0.0..=1.0).contains(&m.dissent_target_bias.to_manager));
        assert!(m.dissent_target_bias.to_manager > 0.0);
        assert!((m.hierarchy_score - 1.0).abs() < 1e-9);
    }
}
