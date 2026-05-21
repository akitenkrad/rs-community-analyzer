//! Shared cl-tohoku BERT sequence-classification backbone.
//!
//! Both the sentiment model (BERT-WRIME) and the NLI stance model
//! (JSNLI-BERT) are `cl-tohoku/bert-base-japanese-v2` fine-tunes: a standard
//! [`candle_transformers::models::bert::BertModel`] encoder plus a linear
//! classification head over the `[CLS]` (first-token) pooled representation.
//!
//! ## Tokenization (cl-tohoku v2)
//!
//! cl-tohoku v2 uses **MeCab word segmentation followed by WordPiece**. The
//! HuggingFace `tokenizers` crate cannot run MeCab on its own. We handle this
//! in two ways, in priority order:
//!
//! 1. If the repo ships a usable `tokenizer.json` (a fast tokenizer), load it
//!    directly — preferred and self-contained.
//! 2. Otherwise, glue [`lindera`] (MeCab-equivalent word segmentation, only
//!    available under the `lindera` feature) for the word-splitting step and
//!    run WordPiece ourselves against the model's `vocab.txt`.
//!
//! If neither path is available (no `tokenizer.json` AND the `lindera` feature
//! is off), [`BertTokenizer::load`] returns a clear [`CommError`] rather than
//! producing wrong tokenization. See `// TODO(candle):` below.

#[cfg(feature = "lindera")]
use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

use crate::error::{CommError, Result};
use crate::nlp::registry::{resolve_assets, ModelAssets, ResolvedConfig};

fn err(e: impl std::fmt::Display) -> CommError {
    CommError::Nlp(e.to_string())
}

/// Tokenizer for cl-tohoku BERT, with a fast-tokenizer or lindera+WordPiece path.
enum BertTokenizer {
    /// HuggingFace fast tokenizer from `tokenizer.json`.
    Fast(Box<Tokenizer>),
    /// lindera word segmentation + manual WordPiece over `vocab.txt`.
    #[cfg(feature = "lindera")]
    LinderaWordPiece(Box<WordPieceVocab>),
}

#[cfg(feature = "lindera")]
struct WordPieceVocab {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    segmenter: lindera::tokenizer::Tokenizer,
}

impl BertTokenizer {
    fn load(assets: &ModelAssets, model_id: &str) -> Result<Self> {
        if let Some(path) = &assets.tokenizer_json {
            let tk = Tokenizer::from_file(path).map_err(err)?;
            return Ok(BertTokenizer::Fast(Box::new(tk)));
        }
        #[cfg(feature = "lindera")]
        {
            if let Some(vocab_path) = &assets.vocab_txt {
                return Ok(BertTokenizer::LinderaWordPiece(Box::new(
                    WordPieceVocab::load(vocab_path)?,
                )));
            }
        }
        // TODO(candle): cl-tohoku v2 BERT genuinely needs MeCab pre-tokenization.
        // When the model ships neither tokenizer.json nor vocab.txt (and/or the
        // `lindera` feature is disabled), we cannot tokenize correctly — fail
        // loudly rather than emit garbage token ids.
        Err(CommError::Nlp(format!(
            "{model_id}: no tokenizer.json and no usable vocab.txt + lindera path; \
             cl-tohoku BERT requires MeCab pre-tokenization (enable the `lindera` \
             feature or use a repo revision shipping tokenizer.json)"
        )))
    }

    /// Encode a (text, optional pair) into (input_ids, attention_mask, token_type_ids).
    fn encode(&self, text: &str, pair: Option<&str>) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        match self {
            BertTokenizer::Fast(tk) => {
                let enc = match pair {
                    Some(p) => tk
                        .encode((text.to_string(), p.to_string()), true)
                        .map_err(err)?,
                    None => tk.encode(text.to_string(), true).map_err(err)?,
                };
                Ok((
                    enc.get_ids().to_vec(),
                    enc.get_attention_mask().to_vec(),
                    enc.get_type_ids().to_vec(),
                ))
            }
            #[cfg(feature = "lindera")]
            BertTokenizer::LinderaWordPiece(wp) => wp.encode(text, pair),
        }
    }
}

#[cfg(feature = "lindera")]
impl WordPieceVocab {
    fn load(vocab_path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(vocab_path)?;
        let mut vocab = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            vocab.insert(line.to_string(), i as u32);
        }
        let unk_id = *vocab.get("[UNK]").unwrap_or(&1);
        let cls_id = *vocab.get("[CLS]").unwrap_or(&2);
        let sep_id = *vocab.get("[SEP]").unwrap_or(&3);

        // Same IPADIC + Normal-mode config as `morphology::LinderaMorphology`.
        let config = serde_json::json!({
            "segmenter": {
                "dictionary": { "kind": "ipadic" },
                "mode": "normal"
            },
            "character_filters": [],
            "token_filters": []
        });
        let segmenter = lindera::tokenizer::Tokenizer::from_config(&config)
            .map_err(|e| CommError::Nlp(format!("lindera IPADIC unavailable: {e}")))?;

        Ok(Self {
            vocab,
            unk_id,
            cls_id,
            sep_id,
            segmenter,
        })
    }

    /// WordPiece a single pre-segmented word (greedy longest-match).
    fn wordpiece(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut sub_tokens: Vec<u32> = Vec::new();
        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_id: Option<u32> = None;
            while start < end {
                let mut piece: String = chars[start..end].iter().collect();
                if start > 0 {
                    piece = format!("##{piece}");
                }
                if let Some(&id) = self.vocab.get(&piece) {
                    cur_id = Some(id);
                    break;
                }
                end -= 1;
            }
            match cur_id {
                Some(id) => {
                    sub_tokens.push(id);
                    start = end;
                }
                None => {
                    out.push(self.unk_id);
                    return;
                }
            }
        }
        out.extend(sub_tokens);
    }

    fn tokenize_one(&self, text: &str) -> Vec<u32> {
        let lex = self.segmenter.tokenize(text).unwrap_or_default();
        let mut ids = Vec::new();
        for tok in &lex {
            self.wordpiece(tok.text.as_ref(), &mut ids);
        }
        ids
    }

    fn encode(&self, text: &str, pair: Option<&str>) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let mut ids = vec![self.cls_id];
        let mut type_ids = vec![0u32];
        let a = self.tokenize_one(text);
        ids.extend(&a);
        type_ids.extend(std::iter::repeat_n(0u32, a.len()));
        ids.push(self.sep_id);
        type_ids.push(0);
        if let Some(p) = pair {
            let b = self.tokenize_one(p);
            ids.extend(&b);
            type_ids.extend(std::iter::repeat_n(1u32, b.len()));
            ids.push(self.sep_id);
            type_ids.push(1);
        }
        let mask = vec![1u32; ids.len()];
        Ok((ids, mask, type_ids))
    }
}

/// A cl-tohoku BERT encoder + linear classification head.
pub struct BertSequenceClassifier {
    bert: BertModel,
    classifier: Linear,
    tokenizer: BertTokenizer,
    device: Device,
    /// Number of output labels (= classifier out-features). Retained for
    /// diagnostics and future callers that branch on label arity.
    #[allow(dead_code)]
    pub num_labels: usize,
}

impl BertSequenceClassifier {
    /// Load the encoder, classifier head, and tokenizer for `model_id`.
    ///
    /// `num_labels` is the expected number of output classes (8 for WRIME,
    /// 3 for JSNLI). The head weights are read from `classifier.{weight,bias}`.
    pub fn load(rc: &ResolvedConfig, model_id: &str, num_labels: usize) -> Result<Self> {
        let assets = resolve_assets(model_id)?;
        let tokenizer = BertTokenizer::load(&assets, model_id)?;

        let config_bytes = std::fs::read(&assets.config)?;
        let config: Config = serde_json::from_slice(&config_bytes)
            .map_err(|e| CommError::Nlp(format!("parse bert config for {model_id}: {e}")))?;
        let hidden = config.hidden_size;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[assets.weights], DType::F32, &rc.device)
                .map_err(err)?
        };
        let bert = BertModel::load(vb.pp("bert"), &config)
            .or_else(|_| BertModel::load(vb.clone(), &config))
            .map_err(err)?;
        let classifier = candle_nn::linear(hidden, num_labels, vb.pp("classifier")).map_err(err)?;

        Ok(Self {
            bert,
            classifier,
            tokenizer,
            device: rc.device.clone(),
            num_labels,
        })
    }

    /// Run the classifier and return raw logits for `(text, pair)`.
    pub fn logits(&self, text: &str, pair: Option<&str>) -> Result<Vec<f32>> {
        let (ids, mask, type_ids) = self.tokenizer.encode(text, pair)?;
        let seq = ids.len().max(1);
        let input_ids = Tensor::from_vec(ids, (1, seq), &self.device).map_err(err)?;
        let token_type_ids = Tensor::from_vec(type_ids, (1, seq), &self.device).map_err(err)?;
        let attn = Tensor::from_vec(mask, (1, seq), &self.device).map_err(err)?;

        let sequence_output = self
            .bert
            .forward(&input_ids, &token_type_ids, Some(&attn))
            .map_err(err)?; // [1, seq, hidden]
                            // Pool the [CLS] token (index 0).
        let cls = sequence_output.i((.., 0, ..)).map_err(err)?; // [1, hidden]
        let logits = self.classifier.forward(&cls).map_err(err)?; // [1, num_labels]
        let row: Vec<Vec<f32>> = logits.to_vec2().map_err(err)?;
        Ok(row.into_iter().next().unwrap_or_default())
    }
}

/// Numerically stable softmax over a slice.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Index of the maximum value (argmax), or 0 for an empty slice.
pub fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(p.len(), 3);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn softmax_empty_and_argmax() {
        assert!(softmax(&[]).is_empty());
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[]), 0);
    }
}
