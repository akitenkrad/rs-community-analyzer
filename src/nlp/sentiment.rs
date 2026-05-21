//! Sentiment via cl-tohoku BERT-WRIME (`Mizuiro-sakura/bert-base-japanese-v2-wrime-fine-tune`).
//!
//! The model outputs 8 WRIME emotion scores. We map them to the library's
//! `(polarity, magnitude)` contract (design §6.5):
//!
//! - `polarity = (喜び + 期待 + 信頼) − (悲しみ + 怒り + 恐れ + 嫌悪)`, scaled to `[-1, 1]`
//! - `magnitude = Σ|emotion| / 8`, clamped to `[0, 1]`
//!
//! WRIME emotion order (model card):
//! `[喜び, 悲しみ, 期待, 驚き, 怒り, 恐れ, 嫌悪, 信頼]`.

use crate::error::Result;
use crate::nlp::bert_classifier::BertSequenceClassifier;
use crate::nlp::registry::ResolvedConfig;
use crate::nlp::Sentiment;

/// Number of WRIME emotion outputs.
const NUM_EMOTIONS: usize = 8;

// Indices into the 8-emotion logit vector.
const JOY: usize = 0; // 喜び
const SADNESS: usize = 1; // 悲しみ
const ANTICIPATION: usize = 2; // 期待
const _SURPRISE: usize = 3; // 驚き (sign-neutral; excluded from polarity)
const ANGER: usize = 4; // 怒り
const FEAR: usize = 5; // 恐れ
const DISGUST: usize = 6; // 嫌悪
const TRUST: usize = 7; // 信頼

/// A loaded BERT-WRIME sentiment model.
pub struct SentimentModel {
    classifier: BertSequenceClassifier,
}

impl SentimentModel {
    pub fn load(rc: &ResolvedConfig) -> Result<Self> {
        let classifier = BertSequenceClassifier::load(rc, &rc.sentiment_id, NUM_EMOTIONS)?;
        Ok(Self { classifier })
    }

    pub fn sentiment(&self, text: &str) -> Result<Sentiment> {
        let logits = self.classifier.logits(text, None)?;
        Ok(emotions_to_sentiment(&logits))
    }
}

/// Map an 8-emotion logit/score vector to `(polarity, magnitude)`.
///
/// Pure and unit-testable: takes raw model outputs and applies the §6.5
/// mapping with `tanh`-style bounding. Returns a neutral result for any vector
/// that is not exactly length 8.
pub fn emotions_to_sentiment(e: &[f32]) -> Sentiment {
    if e.len() != NUM_EMOTIONS {
        return Sentiment {
            polarity: 0.0,
            magnitude: 0.0,
        };
    }
    let positive = e[JOY] + e[ANTICIPATION] + e[TRUST];
    let negative = e[SADNESS] + e[ANGER] + e[FEAR] + e[DISGUST];
    let raw = positive - negative;
    // Bound to [-1, 1] without assuming a fixed input scale.
    let polarity = raw.tanh();
    let magnitude = (e.iter().map(|x| x.abs()).sum::<f32>() / NUM_EMOTIONS as f32).clamp(0.0, 1.0);
    Sentiment {
        polarity,
        magnitude,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_length_is_neutral() {
        let s = emotions_to_sentiment(&[0.1, 0.2, 0.3]);
        assert_eq!(s.polarity, 0.0);
        assert_eq!(s.magnitude, 0.0);
    }

    #[test]
    fn positive_emotions_give_positive_polarity() {
        // High joy/anticipation/trust, low negatives.
        let e = [0.9, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.7];
        let s = emotions_to_sentiment(&e);
        assert!(s.polarity > 0.0);
        assert!((-1.0..=1.0).contains(&s.polarity));
        assert!((0.0..=1.0).contains(&s.magnitude));
    }

    #[test]
    fn negative_emotions_give_negative_polarity() {
        // High sadness/anger/fear/disgust.
        let e = [0.0, 0.9, 0.0, 0.1, 0.8, 0.7, 0.6, 0.0];
        let s = emotions_to_sentiment(&e);
        assert!(s.polarity < 0.0);
        assert!((-1.0..=1.0).contains(&s.polarity));
    }

    #[test]
    fn balanced_is_near_zero() {
        let e = [0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0];
        let s = emotions_to_sentiment(&e);
        // positive = 0.5+0.5+0 = 1.0; negative = 0.5+0.5+0.5+0 = 1.5 => slightly neg
        assert!(s.polarity <= 0.0);
    }
}
