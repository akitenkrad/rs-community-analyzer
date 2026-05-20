//! Aggregate [`ReportSummary`] computation across the H1–H5 metrics．
//!
//! This is a **documented, tunable heuristic**．For every available indicator
//! we derive a normalized risk in `[0, 1]` and classify it red / yellow /
//! green by fixed thresholds．The overall health score is `1.0 - mean(risk)`
//! over available indicators (clamped to `[0, 1]`; `0.5` when nothing is
//! available)．
//!
//! These thresholds are first-pass calibration only and are expected to be
//! re-tuned with real organizational data; none of them constitute a
//! clinical or HR judgement (see the ethics note in every report)．

use crate::models::{H1Metrics, H2Metrics, H3Metrics, H4Metrics, H5Metrics, ReportSummary};

// ===========================================================================
// HEURISTIC THRESHOLDS (tunable)
// ===========================================================================

const H2_HEDGE_RED: f64 = 0.25;
const H2_HEDGE_YELLOW: f64 = 0.12;
const H2_DEFENSE_RED: f64 = 0.25;
const H2_DEFENSE_YELLOW: f64 = 0.12;

const H3_PR_RED: f64 = 0.6;
const H3_PR_YELLOW: f64 = 0.4;
const H3_CONC_RED: f64 = 0.6;
const H3_CONC_YELLOW: f64 = 0.4;
const H3_GINI_RED: f64 = 0.6;
const H3_GINI_YELLOW: f64 = 0.45;

const H4_SURFACE_RED: f64 = 0.30;
const H4_SURFACE_YELLOW: f64 = 0.15;

const H5_UNRESOLVED_RED: f64 = 0.6;
const H5_UNRESOLVED_YELLOW: f64 = 0.4;

const H1_SILENCE_RATIO_RED: f64 = 2.0;
const H1_SILENCE_RATIO_YELLOW: f64 = 1.3;

struct Eval {
    risk: f64,
}

#[allow(clippy::too_many_arguments)]
fn classify(
    value: f64,
    red: f64,
    yellow: f64,
    red_msg: String,
    yellow_msg: String,
    green_msg: String,
    s: &mut ReportSummary,
) -> Eval {
    let risk = value.clamp(0.0, 1.0);
    if value >= red {
        s.red_flags.push(red_msg);
    } else if value >= yellow {
        s.yellow_flags.push(yellow_msg);
    } else {
        s.green_signals.push(green_msg);
    }
    Eval { risk }
}

/// Build the aggregate report summary from whichever hypotheses were computed．
pub fn build_summary(
    h1: Option<&H1Metrics>,
    h2: Option<&H2Metrics>,
    h3: Option<&H3Metrics>,
    h4: Option<&H4Metrics>,
    h5: Option<&H5Metrics>,
) -> ReportSummary {
    let mut s = ReportSummary {
        overall_health_score: 0.0,
        red_flags: Vec::new(),
        yellow_flags: Vec::new(),
        green_signals: Vec::new(),
    };
    let mut risks: Vec<f64> = Vec::new();

    if let Some(m) = h2 {
        risks.push(
            classify(
                m.hedging_rate,
                H2_HEDGE_RED,
                H2_HEDGE_YELLOW,
                format!("ヘッジ表現率 {:.0}% (基準 < 12%)", m.hedging_rate * 100.0),
                format!("ヘッジ表現率 {:.0}% がやや高い", m.hedging_rate * 100.0),
                "ヘッジ表現率は健全な水準".to_string(),
                &mut s,
            )
            .risk,
        );
        risks.push(
            classify(
                m.self_defense_rate,
                H2_DEFENSE_RED,
                H2_DEFENSE_YELLOW,
                format!(
                    "自己防衛表現率 {:.0}% (基準 < 12%)",
                    m.self_defense_rate * 100.0
                ),
                format!(
                    "自己防衛表現率 {:.0}% がやや高い",
                    m.self_defense_rate * 100.0
                ),
                "自己防衛表現率は健全な水準".to_string(),
                &mut s,
            )
            .risk,
        );
    }

    if let Some(m) = h3 {
        risks.push(
            classify(
                m.pagerank_topk_manager_share,
                H3_PR_RED,
                H3_PR_YELLOW,
                format!(
                    "PageRank上位の管理職占有率 {:.0}% (基準 < 40%)",
                    m.pagerank_topk_manager_share * 100.0
                ),
                format!(
                    "PageRank上位の管理職占有率 {:.0}% がやや高い",
                    m.pagerank_topk_manager_share * 100.0
                ),
                "中心性は管理職に偏っていない".to_string(),
                &mut s,
            )
            .risk,
        );
        let mean_conc = if m.normalized_concentration.is_empty() {
            0.0
        } else {
            m.normalized_concentration
                .iter()
                .map(|(_, v)| *v)
                .sum::<f64>()
                / m.normalized_concentration.len() as f64
        };
        risks.push(
            classify(
                mean_conc,
                H3_CONC_RED,
                H3_CONC_YELLOW,
                format!("平均発言集中度 {:.2} (基準 < 0.40)", mean_conc),
                format!("平均発言集中度 {:.2} がやや高い", mean_conc),
                "発言は特定者に集中していない".to_string(),
                &mut s,
            )
            .risk,
        );
        risks.push(
            classify(
                m.reply_concentration_gini,
                H3_GINI_RED,
                H3_GINI_YELLOW,
                format!(
                    "返信集中Gini {:.2} (基準 < 0.45)",
                    m.reply_concentration_gini
                ),
                format!("返信集中Gini {:.2} がやや高い", m.reply_concentration_gini),
                "返信は偏っていない".to_string(),
                &mut s,
            )
            .risk,
        );
    }

    if let Some(m) = h4 {
        risks.push(
            classify(
                m.surface_agreement_rate,
                H4_SURFACE_RED,
                H4_SURFACE_YELLOW,
                format!(
                    "表層同意率 {:.0}% (基準 < 15%)",
                    m.surface_agreement_rate * 100.0
                ),
                format!(
                    "表層同意率 {:.0}% がやや高い (基準 < 15%)",
                    m.surface_agreement_rate * 100.0
                ),
                "表層同意率は健全な水準".to_string(),
                &mut s,
            )
            .risk,
        );
    }

    if let Some(m) = h5 {
        risks.push(
            classify(
                m.unresolved_thread_ratio,
                H5_UNRESOLVED_RED,
                H5_UNRESOLVED_YELLOW,
                format!(
                    "未解決スレッド比 {:.0}% (基準 < 40%)",
                    m.unresolved_thread_ratio * 100.0
                ),
                format!(
                    "未解決スレッド比 {:.0}% がやや高い",
                    m.unresolved_thread_ratio * 100.0
                ),
                "スレッドは概ね収束している".to_string(),
                &mut s,
            )
            .risk,
        );
    }

    if let Some(m) = h1 {
        let ratio = m.silence_after_manager.ratio;
        let risk = (ratio / H1_SILENCE_RATIO_RED).clamp(0.0, 1.0);
        if ratio >= H1_SILENCE_RATIO_RED {
            s.red_flags.push(format!(
                "管理職/スタッフ沈黙比 {:.2} (基準 < 1.3; 管理職発言が無視されやすい)",
                ratio
            ));
        } else if ratio >= H1_SILENCE_RATIO_YELLOW {
            s.yellow_flags
                .push(format!("管理職/スタッフ沈黙比 {:.2} がやや高い", ratio));
        } else {
            s.green_signals
                .push("管理職発言の沈黙比は健全な水準".to_string());
        }
        risks.push(risk);
    }

    s.overall_health_score = if risks.is_empty() {
        0.5
    } else {
        let mean_risk = risks.iter().sum::<f64>() / risks.len() as f64;
        (1.0 - mean_risk).clamp(0.0, 1.0)
    };
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DissentBias, SilenceRatio};

    fn h2(hedge: f64, defense: f64) -> H2Metrics {
        H2Metrics {
            hedging_rate: hedge,
            self_defense_rate: defense,
            incomplete_utterance_rate: 0.0,
            escalation_delay_minutes: 0.0,
            dm_dependency_ratio: None,
            sample_size: 10,
        }
    }

    #[test]
    fn test_empty_is_neutral() {
        let s = build_summary(None, None, None, None, None);
        assert_eq!(s.overall_health_score, 0.5);
        assert!(s.red_flags.is_empty());
    }

    #[test]
    fn test_h2_red_yellow_green_buckets() {
        let m = h2(0.30, 0.15);
        let s = build_summary(None, Some(&m), None, None, None);
        assert_eq!(s.red_flags.len(), 1);
        assert_eq!(s.yellow_flags.len(), 1);

        let m = h2(0.01, 0.01);
        let s = build_summary(None, Some(&m), None, None, None);
        assert_eq!(s.green_signals.len(), 2);
        assert!(s.overall_health_score > 0.9);
    }

    #[test]
    fn test_h1_silence_ratio_buckets() {
        let mk = |ratio: f64| H1Metrics {
            decision_change_rate: 0.0,
            initial_proposal_regression: 0.0,
            silence_after_manager: SilenceRatio {
                manager_post_silence_rate: 0.0,
                staff_post_silence_rate: 0.0,
                ratio,
            },
            dissent_convergence_speed_minutes: 0.0,
            thread_count: 0,
        };
        let s = build_summary(Some(&mk(2.5)), None, None, None, None);
        assert_eq!(s.red_flags.len(), 1);
        let s = build_summary(Some(&mk(1.5)), None, None, None, None);
        assert_eq!(s.yellow_flags.len(), 1);
        let s = build_summary(Some(&mk(0.8)), None, None, None, None);
        assert_eq!(s.green_signals.len(), 1);
    }

    #[test]
    fn test_h3_h4_h5_boundaries() {
        let h3 = H3Metrics {
            utterance_entropy_by_channel: vec![],
            normalized_concentration: vec![("a".into(), 0.7), ("b".into(), 0.7)],
            reply_concentration_gini: 0.7,
            manager_reaction_rate: 0.0,
            dissent_target_bias: DissentBias {
                to_manager: 0.0,
                to_staff: 0.0,
                ratio_staff_to_manager: 0.0,
            },
            hierarchy_score: 0.0,
            pagerank_topk_manager_share: 0.7,
        };
        let h4 = H4Metrics {
            surface_agreement_rate: 0.4,
            public_private_sentiment_delta: None,
            post_meeting_dissent_rate: 0.0,
            execution_delay_hours: 0.0,
            reaction_text_disagreement: None,
        };
        let h5 = H5Metrics {
            proposal_semantic_diversity: None,
            hypothesis_retention_period_hours: 0.0,
            unresolved_thread_ratio: 0.7,
            novel_vocabulary_rate_monthly: vec![],
            topic_cluster_count_monthly: None,
        };
        let s = build_summary(None, None, Some(&h3), Some(&h4), Some(&h5));
        assert_eq!(s.red_flags.len(), 5);
        assert!(s.overall_health_score < 0.5);
    }
}
