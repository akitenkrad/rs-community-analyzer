//! NLP enrichment: fill the H1/H4/H5 metric fields that need the Python sidecar．
//!
//! `compute_h1` / `compute_h4` / `compute_h5` stay sync and Rust-only．These
//! async functions run *after* them, when a sidecar is available, and populate
//! the embedding / sentiment / clustering dependent fields．
//!
//! **Graceful degradation**: any sidecar / IO error is logged with
//! `tracing::warn!` and the affected field is left untouched．Enrichment never
//! returns an error that would abort the whole analysis run．

use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Utc};

use crate::analysis::{group_threads, timestamp_secs_f64};
use crate::error::Result;
use crate::models::{ChannelCategory, H1Metrics, H4Metrics, H5Metrics};
use crate::nlp_sidecar::{NlpRequest, NlpResponse, NlpSidecar};
use crate::types::{AnalysisInput, Message};

/// Emoji names treated as "agreement" reactions (mirrors H3's set)．
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

const PROPOSAL_CUES: &[&str] = &[
    "提案",
    "案として",
    "案1",
    "案2",
    "どうでしょう",
    "してはどう",
    "はどうか",
    "アイデア",
    "代替案",
    "別案",
];

const SENTIMENT_SAMPLE_CAP: usize = 300;
const CLUSTER_SAMPLE_CAP: usize = 500;

fn norm_emoji(e: &str) -> String {
    e.trim().trim_matches(':').to_ascii_lowercase()
}

fn is_agreement_emoji(name: &str) -> bool {
    let n = norm_emoji(name);
    AGREEMENT_EMOJIS.iter().any(|a| a.to_ascii_lowercase() == n)
}

/// Build the set of message ids that received at least one agreement-emoji
/// reaction from any user other than the message author．
fn agreement_reacted_ids(input: &AnalysisInput<'_>) -> HashSet<String> {
    use std::collections::HashMap;
    let author: HashMap<&str, &str> = input
        .messages
        .iter()
        .map(|m| (m.id.as_str(), m.author_id.as_str()))
        .collect();
    let mut out: HashSet<String> = HashSet::new();
    for r in input.reactions {
        if !is_agreement_emoji(&r.emoji_name) {
            continue;
        }
        if let Some(&a) = author.get(r.message_id.as_str()) {
            if r.user_id == a {
                continue;
            }
        }
        out.insert(r.message_id.clone());
    }
    out
}

fn capped_by_ts<'a>(msgs: &[&'a Message], cap: usize) -> Vec<&'a Message> {
    let mut v: Vec<&Message> = msgs.to_vec();
    v.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    v.truncate(cap);
    v
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

fn month_bucket(t: DateTime<Utc>) -> String {
    t.format("%Y-%m").to_string()
}

async fn polarities(sc: &mut NlpSidecar, texts: &[String]) -> Result<Vec<f64>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let reqs: Vec<NlpRequest> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| NlpRequest::Sentiment {
            id: i.to_string(),
            text: t.clone(),
        })
        .collect();
    let resps = sc.batch_request(reqs).await?;
    let mut out = Vec::with_capacity(resps.len());
    for r in resps {
        match r {
            NlpResponse::Sentiment { polarity, .. } => out.push(polarity as f64),
            NlpResponse::Error { message, .. } => {
                return Err(crate::error::CommError::Nlp(message))
            }
            other => {
                return Err(crate::error::CommError::Nlp(format!(
                    "unexpected sentiment response: {other:?}"
                )))
            }
        }
    }
    Ok(out)
}

fn category_of(input: &AnalysisInput<'_>, cid: &str) -> ChannelCategory {
    if let Some(c) = input.channels.iter().find(|c| c.id == cid) {
        if let Some(cat) = c.category {
            return cat;
        }
        return input.config.classify_channel(&c.name);
    }
    ChannelCategory::Unknown
}

/// Enrich H4 NLP-dependent fields: `public_private_sentiment_delta`,
/// `reaction_text_disagreement`．
///
/// `reaction_text_disagreement`: GENERALIZED — previously parsed Slack
/// `raw_json` for reactions; now iterates `input.reactions` to find messages
/// that received an agreement-emoji reaction, then scores their text．Rate
/// = (#scored with polarity < −0.1) / (#scored)．
pub async fn enrich_h4(
    input: &AnalysisInput<'_>,
    sc: &mut NlpSidecar,
    m: &mut H4Metrics,
) -> Result<()> {
    // --- public_private_sentiment_delta ---
    let public: Vec<&Message> = input
        .messages
        .iter()
        .filter(|msg| {
            matches!(
                category_of(input, &msg.channel_id),
                ChannelCategory::Official | ChannelCategory::Tech | ChannelCategory::Leadership
            )
        })
        .collect();
    let casual: Vec<&Message> = input
        .messages
        .iter()
        .filter(|msg| matches!(category_of(input, &msg.channel_id), ChannelCategory::Casual))
        .collect();

    if !public.is_empty() && !casual.is_empty() {
        let pub_texts: Vec<String> = capped_by_ts(&public, SENTIMENT_SAMPLE_CAP)
            .iter()
            .map(|m| m.text.clone())
            .collect();
        let cas_texts: Vec<String> = capped_by_ts(&casual, SENTIMENT_SAMPLE_CAP)
            .iter()
            .map(|m| m.text.clone())
            .collect();
        match (
            polarities(sc, &pub_texts).await,
            polarities(sc, &cas_texts).await,
        ) {
            (Ok(p), Ok(c)) => {
                if let (Some(pm), Some(cm)) = (mean(&p), mean(&c)) {
                    m.public_private_sentiment_delta = Some(pm - cm);
                }
            }
            (Err(e), _) | (_, Err(e)) => {
                tracing::warn!(error = %e, "enrich_h4: sentiment delta degraded to None");
            }
        }
    }

    // --- reaction_text_disagreement (GENERALIZED to use &[Reaction]) ---
    let reacted_ids = agreement_reacted_ids(input);
    let reacted: Vec<&Message> = input
        .messages
        .iter()
        .filter(|m| reacted_ids.contains(&m.id))
        .collect();
    if !reacted.is_empty() {
        let texts: Vec<String> = reacted.iter().map(|m| m.text.clone()).collect();
        match polarities(sc, &texts).await {
            Ok(p) if !p.is_empty() => {
                let neg = p.iter().filter(|&&x| x < -0.1).count();
                m.reaction_text_disagreement = Some(neg as f64 / p.len() as f64);
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "enrich_h4: reaction/text disagreement degraded to None");
            }
        }
    }

    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

const CHANGE_THREAD_CAP: usize = 400;
const CHANGE_SIM_THRESHOLD: f32 = 0.5;
const REGRESSION_SIM_THRESHOLD: f32 = 0.7;
const REGRESSION_DIVERGE_THRESHOLD: f32 = 0.5;
const STANCE_CALL_CAP: usize = 800;

/// Enrich H1 NLP-dependent fields using the sidecar．
pub async fn enrich_h1(
    input: &AnalysisInput<'_>,
    sc: &mut NlpSidecar,
    m: &mut H1Metrics,
) -> Result<()> {
    let groups = group_threads(input.messages);

    // --- decision_change_rate ---
    let qualifying: Vec<&crate::analysis::ThreadGroup> = groups
        .iter()
        .filter(|g| g.messages.len() >= 2)
        .take(CHANGE_THREAD_CAP)
        .collect();
    if !qualifying.is_empty() {
        let mut changed = 0usize;
        let mut counted = 0usize;
        let mut degraded = false;
        for g in &qualifying {
            let first = g.messages.first().unwrap().text.clone();
            let last = g.messages.last().unwrap().text.clone();
            match sc
                .request(NlpRequest::Embed {
                    id: g.thread_key.clone(),
                    texts: vec![first, last],
                })
                .await
            {
                Ok(NlpResponse::Embed { vectors, .. }) if vectors.len() >= 2 => {
                    counted += 1;
                    if cosine(&vectors[0], &vectors[1]) < CHANGE_SIM_THRESHOLD {
                        changed += 1;
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "enrich_h1: decision-change embed degraded");
                    degraded = true;
                    break;
                }
            }
        }
        if counted > 0 && !degraded {
            m.decision_change_rate = changed as f64 / counted as f64;
        }
    }

    // --- initial_proposal_regression ---
    let regress_threads: Vec<&crate::analysis::ThreadGroup> = groups
        .iter()
        .filter(|g| g.messages.len() >= 3)
        .take(CHANGE_THREAD_CAP)
        .collect();
    if !regress_threads.is_empty() {
        let mut regressed = 0usize;
        let mut counted = 0usize;
        let mut degraded = false;
        for g in &regress_threads {
            let n = g.messages.len();
            let texts: Vec<String> = g.messages.iter().map(|m| m.text.clone()).collect();
            let embeds = match sc
                .request(NlpRequest::Embed {
                    id: g.thread_key.clone(),
                    texts,
                })
                .await
            {
                Ok(NlpResponse::Embed { vectors, .. }) if vectors.len() == n => vectors,
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "enrich_h1: regression embed degraded");
                    degraded = true;
                    break;
                }
            };
            let first = &embeds[0];
            let last = &embeds[n - 1];
            let mut min_div = f32::INFINITY;
            for emb in embeds.iter().take(n - 1).skip(1) {
                let c = cosine(first, emb);
                if c < min_div {
                    min_div = c;
                }
            }
            counted += 1;
            let cos_fl = cosine(first, last);
            if cos_fl >= REGRESSION_SIM_THRESHOLD && min_div < REGRESSION_DIVERGE_THRESHOLD {
                regressed += 1;
            }
        }
        if counted > 0 && !degraded {
            m.initial_proposal_regression = regressed as f64 / counted as f64;
        }
    }

    // --- dissent_convergence (stance refinement, optional) ---
    let mut stance_budget = STANCE_CALL_CAP;
    let mut stance_conv: Vec<f64> = Vec::new();
    let mut stance_degraded = false;
    'threads: for g in &groups {
        if g.messages.len() < 2 {
            continue;
        }
        let root_text = g.messages.first().map(|m| m.text.clone());
        let mut dissent_ts: Vec<f64> = Vec::new();
        for msg in &g.messages {
            if stance_budget == 0 {
                break 'threads;
            }
            stance_budget -= 1;
            match sc
                .request(NlpRequest::Stance {
                    id: msg.id.clone(),
                    text: msg.text.clone(),
                    context: root_text.clone(),
                })
                .await
            {
                Ok(NlpResponse::Stance { label, .. }) => {
                    if label == "disagree" {
                        dissent_ts.push(timestamp_secs_f64(msg.timestamp));
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "enrich_h1: stance refinement degraded");
                    stance_degraded = true;
                    break 'threads;
                }
            }
        }
        if !dissent_ts.is_empty() {
            dissent_ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let first = dissent_ts[0];
            let last = *dissent_ts.last().unwrap();
            stance_conv.push((last - first) / 60.0);
        }
    }
    if !stance_degraded && !stance_conv.is_empty() {
        stance_conv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = stance_conv.len();
        m.dissent_convergence_speed_minutes = if n % 2 == 1 {
            stance_conv[n / 2]
        } else {
            (stance_conv[n / 2 - 1] + stance_conv[n / 2]) / 2.0
        };
    }

    Ok(())
}

/// Enrich H5 NLP-dependent fields: `proposal_semantic_diversity`,
/// `topic_cluster_count_monthly`．
pub async fn enrich_h5(
    input: &AnalysisInput<'_>,
    sc: &mut NlpSidecar,
    m: &mut H5Metrics,
) -> Result<()> {
    let is_proposal = |t: &str| PROPOSAL_CUES.iter().any(|c| t.contains(c));
    let groups = group_threads(input.messages);
    let mut thread_diversities: Vec<f64> = Vec::new();
    for g in &groups {
        let proposals: Vec<&Message> = g.messages.iter().filter(|m| is_proposal(&m.text)).collect();
        if proposals.len() < 2 {
            continue;
        }
        let texts: Vec<String> = proposals.iter().map(|m| m.text.clone()).collect();
        let req = NlpRequest::Embed {
            id: g.thread_key.clone(),
            texts,
        };
        match sc.request(req).await {
            Ok(NlpResponse::Embed { vectors, .. }) if vectors.len() >= 2 => {
                if let Some(div) = covariance_trace(&vectors) {
                    thread_diversities.push(div);
                }
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "enrich_h5: proposal diversity (one thread) skipped");
            }
        }
    }
    if let Some(avg) = mean(&thread_diversities) {
        m.proposal_semantic_diversity = Some(avg);
    }

    let min_cluster_size = (input.config.nlp_sidecar.embed_batch_size / 8).max(2);
    let mut by_month: BTreeMap<String, Vec<&Message>> = BTreeMap::new();
    for msg in input.messages {
        let mk = month_bucket(msg.timestamp);
        by_month.entry(mk).or_default().push(msg);
    }
    let mut monthly: Vec<(String, u32)> = Vec::new();
    for (month, msgs) in &by_month {
        if msgs.len() < min_cluster_size {
            continue;
        }
        let sample = capped_by_ts(msgs, CLUSTER_SAMPLE_CAP);
        let texts: Vec<String> = sample.iter().map(|m| m.text.clone()).collect();
        let embeds = match sc
            .request(NlpRequest::Embed {
                id: month.clone(),
                texts,
            })
            .await
        {
            Ok(NlpResponse::Embed { vectors, .. }) => vectors,
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!(error = %e, month = %month, "enrich_h5: month embed skipped");
                continue;
            }
        };
        match sc
            .request(NlpRequest::Cluster {
                embeddings: embeds,
                min_cluster_size,
            })
            .await
        {
            Ok(NlpResponse::Cluster { num_clusters, .. }) => {
                monthly.push((month.clone(), num_clusters));
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, month = %month, "enrich_h5: month cluster skipped");
            }
        }
    }
    if !monthly.is_empty() {
        m.topic_cluster_count_monthly = Some(monthly);
    }

    Ok(())
}

fn covariance_trace(vectors: &[Vec<f32>]) -> Option<f64> {
    let n = vectors.len();
    if n < 2 {
        return None;
    }
    let d = vectors[0].len();
    if d == 0 || vectors.iter().any(|v| v.len() != d) {
        return None;
    }
    let mut trace = 0.0;
    for j in 0..d {
        let col: Vec<f64> = vectors.iter().map(|v| v[j] as f64).collect();
        let mu = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n as f64;
        trace += var;
    }
    Some(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChannelEntry, CommConfig};
    use crate::models::SilenceRatio;
    use crate::types::{Channel, Reaction};
    use chrono::TimeZone;

    fn sidecar_path() -> String {
        // CARGO_MANIFEST_DIR = <repo>/  (single crate; no workspace layer).
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest)
            .join("tools/comm/src/nlp_sidecar.py")
            .to_string_lossy()
            .into_owned()
    }

    fn python3() -> Option<String> {
        let out = std::process::Command::new("which").arg("python3").output();
        match out {
            Ok(o) if o.status.success() => {
                let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
                (!p.is_empty()).then_some(p)
            }
            _ => None,
        }
    }

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

    fn react(message_id: &str, channel: &str, user: &str, emoji: &str) -> Reaction {
        Reaction {
            message_id: message_id.to_string(),
            channel_id: channel.to_string(),
            user_id: user.to_string(),
            emoji_name: emoji.to_string(),
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

    fn cfg_with_channels() -> CommConfig {
        let mut c = CommConfig::default();
        c.channels.push(ChannelEntry {
            pattern: "^proj-.*".into(),
            category: ChannelCategory::Official,
            is_decision_channel: true,
        });
        c.channels.push(ChannelEntry {
            pattern: "^times-.*".into(),
            category: ChannelCategory::Casual,
            is_decision_channel: false,
        });
        c
    }

    #[test]
    fn test_covariance_trace_basic() {
        assert!(covariance_trace(&[vec![1.0]]).is_none());
        let t = covariance_trace(&[vec![0.0, 0.0], vec![2.0, 0.0]]).unwrap();
        assert!((t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_agreement_reacted_ids() {
        let msgs = vec![
            msg("C1", "M1", "U1", "ok", ts(1), None),
            msg("C1", "M2", "U1", "no", ts(2), None),
            msg("C1", "M3", "U1", "plain", ts(3), None),
        ];
        let reactions = vec![
            react("M1", "C1", "U2", "+1"),
            react("M2", "C1", "U2", "eyes"),
        ];
        let cfg = CommConfig::default();
        let input = AnalysisInput {
            messages: &msgs,
            channels: &[],
            users: &[],
            reactions: &reactions,
            config: &cfg,
        };
        let ids = agreement_reacted_ids(&input);
        assert!(ids.contains("M1"));
        assert!(!ids.contains("M2"));
        assert!(!ids.contains("M3"));
    }

    #[tokio::test]
    async fn test_enrich_h4_h5_with_mock_sidecar() {
        let py = match python3() {
            Some(p) => p,
            None => {
                eprintln!("skip: python3 not found");
                return;
            }
        };
        let cfg = cfg_with_channels();
        let channels = vec![channel("C_OFF", "proj-x"), channel("C_CAS", "times-foo")];

        let jan = 1_735_732_800_i64;
        let feb = 1_738_454_400_i64;
        let mut messages: Vec<Message> = Vec::new();
        for i in 0..6 {
            messages.push(msg(
                "C_OFF",
                &format!("M_OFF_{i}"),
                "U1",
                &format!("公式の話題その{i}"),
                ts(jan + i),
                None,
            ));
            messages.push(msg(
                "C_CAS",
                &format!("M_CAS_{i}"),
                "U2",
                &format!("雑談その{i}"),
                ts(feb + i),
                None,
            ));
        }
        // Agreement-reacted message for reaction_text_disagreement.
        messages.push(msg(
            "C_OFF",
            "M_REACT",
            "U3",
            "本当は反対だが",
            ts(jan + 100),
            None,
        ));
        let troot = "M_PROP";
        messages.push(msg("C_OFF", troot, "U1", "提案: A 案", ts(jan + 200), None));
        messages.push(msg(
            "C_OFF",
            "M_PROP_R1",
            "U2",
            "代替案として B はどうでしょう",
            ts(jan + 260),
            Some(troot),
        ));
        messages.push(msg(
            "C_OFF",
            "M_PROP_R2",
            "U3",
            "別案 C のアイデアもあります",
            ts(jan + 320),
            Some(troot),
        ));
        let reactions = vec![react("M_REACT", "C_OFF", "U9", "+1")];

        let input = AnalysisInput {
            messages: &messages,
            channels: &channels,
            users: &[],
            reactions: &reactions,
            config: &cfg,
        };

        let args = vec![sidecar_path(), "--mock".to_string()];
        let mut sc = NlpSidecar::spawn_cmd(&py, &args).await.expect("spawn mock");

        let mut h4 = H4Metrics {
            surface_agreement_rate: 0.0,
            public_private_sentiment_delta: None,
            post_meeting_dissent_rate: 0.0,
            execution_delay_hours: 0.0,
            reaction_text_disagreement: None,
        };
        enrich_h4(&input, &mut sc, &mut h4)
            .await
            .expect("enrich_h4 ok");
        let d = h4.public_private_sentiment_delta.expect("delta Some");
        assert!((-2.0..=2.0).contains(&d));
        let r = h4
            .reaction_text_disagreement
            .expect("reaction disagreement Some");
        assert!((0.0..=1.0).contains(&r));

        let mut h5 = H5Metrics {
            proposal_semantic_diversity: None,
            hypothesis_retention_period_hours: 0.0,
            unresolved_thread_ratio: 0.0,
            novel_vocabulary_rate_monthly: Vec::new(),
            topic_cluster_count_monthly: None,
        };
        enrich_h5(&input, &mut sc, &mut h5)
            .await
            .expect("enrich_h5 ok");
        let div = h5.proposal_semantic_diversity.expect("diversity Some");
        assert!(div >= 0.0);
        let monthly = h5
            .topic_cluster_count_monthly
            .expect("monthly clusters Some");
        assert!(!monthly.is_empty());

        sc.shutdown().await.expect("shutdown ok");
    }

    #[test]
    fn test_cosine_basic() {
        assert_eq!(cosine(&[1.0, 0.0], &[0.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        let c = cosine(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_enrich_h1_with_mock_sidecar() {
        let py = match python3() {
            Some(p) => p,
            None => {
                eprintln!("skip: python3 not found");
                return;
            }
        };
        let cfg = CommConfig::default();

        let r1 = "M_R1";
        let r2 = "M_R2";
        let messages = vec![
            msg("C1", r1, "U1", "提案: 案A を採用したい", ts(1000), None),
            msg(
                "C1",
                "M_R1_1",
                "U2",
                "別案 B を検討すべき",
                ts(1100),
                Some(r1),
            ),
            msg(
                "C1",
                "M_R1_2",
                "U3",
                "やはり案A で確定です",
                ts(1200),
                Some(r1),
            ),
            msg("C1", r2, "U1", "次の議題に移ります", ts(5000), None),
            msg(
                "C1",
                "M_R2_1",
                "U2",
                "全く別のトピックです",
                ts(5100),
                Some(r2),
            ),
            msg(
                "C1",
                "M_R2_2",
                "U3",
                "結論は保留とします",
                ts(5200),
                Some(r2),
            ),
        ];
        let input = AnalysisInput {
            messages: &messages,
            channels: &[],
            users: &[],
            reactions: &[],
            config: &cfg,
        };

        let args = vec![sidecar_path(), "--mock".to_string()];
        let mut sc = NlpSidecar::spawn_cmd(&py, &args).await.expect("spawn mock");

        let mut h1 = H1Metrics {
            decision_change_rate: 0.0,
            initial_proposal_regression: 0.0,
            silence_after_manager: SilenceRatio {
                manager_post_silence_rate: 0.0,
                staff_post_silence_rate: 0.0,
                ratio: 0.0,
            },
            dissent_convergence_speed_minutes: 42.0,
            thread_count: 2,
        };
        enrich_h1(&input, &mut sc, &mut h1)
            .await
            .expect("enrich_h1 ok");

        assert!((0.0..=1.0).contains(&h1.decision_change_rate));
        assert!((0.0..=1.0).contains(&h1.initial_proposal_regression));
        // Mock stance is always "neutral" => the pre-set sentinel is retained.
        assert!((h1.dissent_convergence_speed_minutes - 42.0).abs() < 1e-9);

        sc.shutdown().await.expect("shutdown ok");
    }
}
