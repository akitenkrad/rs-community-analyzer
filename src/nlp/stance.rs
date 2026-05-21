//! Stance classification: NLI (default) or LLM (quality profile).
//!
//! ## NLI mode (`stance_mode = "nli"`)
//!
//! Zero-shot stance via the Japanese NLI model `Formzu/bert-base-japanese-jsnli`
//! (design §6.4). For each of the four stance hypotheses we run an
//! NLI pass (premise = thread context, hypothesis = template), take the
//! `entailment` probability, and pick the label with the highest entailment.
//!
//! | stance   | hypothesis template                          |
//! |----------|----------------------------------------------|
//! | support  | この発言は前の意見に賛成している             |
//! | disagree | この発言は前の意見に反対している             |
//! | neutral  | この発言は前の意見について中立的である       |
//! | clarify  | この発言は前の意見に対して質問・確認をしている |
//!
//! JSNLI label order is `[entailment, neutral, contradiction]`.
//!
//! ## LLM mode (`stance_mode = "llm"`)
//!
//! Sarashina2.2-3b via `candle-transformers::models::llama`. The model is
//! prompted for a JSON `{"label": ..., "reason": ...}` answer; on a parse
//! failure (or if the LLM cannot be loaded), we fall back to NLI.

use crate::error::{CommError, Result};
use crate::nlp::bert_classifier::{argmax, softmax, BertSequenceClassifier};
use crate::nlp::registry::ResolvedConfig;
use crate::nlp::{Stance, StanceLabel};

/// JSNLI: index of the `entailment` class in the 3-way output.
const ENTAILMENT_IDX: usize = 0;
const NUM_NLI_LABELS: usize = 3;

const TEMPLATES: [(StanceLabel, &str); 4] = [
    (StanceLabel::Support, "この発言は前の意見に賛成している"),
    (StanceLabel::Disagree, "この発言は前の意見に反対している"),
    (
        StanceLabel::Neutral,
        "この発言は前の意見について中立的である",
    ),
    (
        StanceLabel::Clarify,
        "この発言は前の意見に対して質問・確認をしている",
    ),
];

/// A loaded stance model (NLI backbone always present; LLM optional).
pub struct StanceModel {
    nli: BertSequenceClassifier,
    use_llm: bool,
    // TODO(candle): hold a loaded Sarashina2.2 (Llama) handle here once the LLM
    // path is exercised with real weights. The model is lazily resolved in
    // `stance()` to avoid pulling 6 GB of weights when only NLI is used.
    llm_id: String,
}

impl StanceModel {
    pub fn load(rc: &ResolvedConfig) -> Result<Self> {
        let nli = BertSequenceClassifier::load(rc, &rc.stance_id, NUM_NLI_LABELS)?;
        let use_llm = rc.stance_mode.eq_ignore_ascii_case("llm") && !rc.stance_llm_id.is_empty();
        Ok(Self {
            nli,
            use_llm,
            llm_id: rc.stance_llm_id.clone(),
        })
    }

    pub fn stance(&self, text: &str, context: Option<&str>) -> Result<Stance> {
        if self.use_llm {
            match self.stance_llm(text, context) {
                Ok(s) => return Ok(s),
                Err(e) => {
                    tracing::warn!(error = %e, model = %self.llm_id, "LLM stance failed; falling back to NLI");
                }
            }
        }
        self.stance_nli(text, context)
    }

    /// NLI zero-shot: premise = `context` (parent message) joined with the
    /// target; hypothesis = each stance template. Highest entailment wins.
    fn stance_nli(&self, text: &str, context: Option<&str>) -> Result<Stance> {
        // Premise carries both the thread context and the target utterance so a
        // standalone message (no context) still produces a meaningful judgement.
        let premise = match context {
            Some(c) if !c.is_empty() => format!("{c} 対象発言: {text}"),
            _ => text.to_string(),
        };
        let mut entail_scores = [0.0f32; TEMPLATES.len()];
        for (i, (_, hyp)) in TEMPLATES.iter().enumerate() {
            let logits = self.nli.logits(&premise, Some(hyp))?;
            let probs = softmax(&logits);
            entail_scores[i] = probs.get(ENTAILMENT_IDX).copied().unwrap_or(0.0);
        }
        Ok(nli_label_from_entailment(&entail_scores))
    }

    /// LLM stance via Sarashina2.2 (Llama). Honestly unimplemented for inference
    /// in this session — returns an error so the caller falls back to NLI.
    fn stance_llm(&self, _text: &str, _context: Option<&str>) -> Result<Stance> {
        // TODO(candle): load Sarashina2.2-3b via candle_transformers::models::llama,
        // render the §6.4 prompt, generate, and parse the JSON answer with
        // `parse_llm_stance`. Blocked here only by the absence of weights/network
        // in this session; the JSON parser below is implemented and tested.
        Err(CommError::Nlp(format!(
            "LLM stance ({}) not wired in this build; use NLI mode",
            self.llm_id
        )))
    }
}

/// Pick the stance label whose hypothesis had the highest entailment score.
pub fn nli_label_from_entailment(entail_scores: &[f32]) -> Stance {
    let idx = argmax(entail_scores);
    let label = TEMPLATES
        .get(idx)
        .map(|(l, _)| *l)
        .unwrap_or(StanceLabel::Neutral);
    let score = entail_scores.get(idx).copied().unwrap_or(0.0);
    Stance { label, score }
}

/// Parse an LLM JSON answer of the form `{"label": "...", "reason": "..."}`.
///
/// Tolerant of surrounding prose: extracts the first `{...}` block. Returns
/// `None` on parse failure or an unrecognized label (caller falls back to NLI).
///
// TODO(candle): this is wired into `stance_llm` once Sarashina2.2 generation is
// implemented; kept tested and ready, hence `dead_code` until then.
#[allow(dead_code)]
pub fn parse_llm_stance(raw: &str) -> Option<Stance> {
    let start = raw.find('{')?;
    let end = raw[start..].find('}').map(|e| start + e + 1)?;
    let json = &raw[start..end];
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let label_str = v.get("label")?.as_str()?.trim().to_ascii_lowercase();
    let label = match label_str.as_str() {
        "support" => StanceLabel::Support,
        "disagree" => StanceLabel::Disagree,
        "neutral" => StanceLabel::Neutral,
        "clarify" => StanceLabel::Clarify,
        _ => return None,
    };
    Some(Stance { label, score: 1.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nli_argmax_picks_disagree() {
        // entailment scores in TEMPLATES order: [support, disagree, neutral, clarify]
        let s = nli_label_from_entailment(&[0.1, 0.8, 0.2, 0.05]);
        assert_eq!(s.label, StanceLabel::Disagree);
        assert!((s.score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn nli_argmax_picks_support() {
        let s = nli_label_from_entailment(&[0.9, 0.1, 0.2, 0.05]);
        assert_eq!(s.label, StanceLabel::Support);
    }

    #[test]
    fn nli_empty_degenerates_to_first_template() {
        // argmax([]) = 0 -> the first template (support); score 0.
        let s = nli_label_from_entailment(&[]);
        assert_eq!(s.label, StanceLabel::Support);
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn parse_llm_stance_ok() {
        let s =
            parse_llm_stance(r#"回答: {"label": "disagree", "reason": "反対している"}"#).unwrap();
        assert_eq!(s.label, StanceLabel::Disagree);
    }

    #[test]
    fn parse_llm_stance_clarify_and_garbage() {
        assert_eq!(
            parse_llm_stance(r#"{"label":"clarify"}"#).unwrap().label,
            StanceLabel::Clarify
        );
        assert!(parse_llm_stance("no json here").is_none());
        assert!(parse_llm_stance(r#"{"label":"???"}"#).is_none());
    }
}
