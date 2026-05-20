//! Pluggable noun extraction (`Morphology` trait).
//!
//! The library never hard-codes a tokenizer.  `compute_h5` (novel-vocabulary
//! rate) and any future text-analysis steps take `&dyn Morphology` so callers
//! can swap implementations:
//!
//! * [`WhitespaceMorphology`] — always available, no features.  Splits on
//!   ASCII whitespace, lowercases, drops tokens shorter than 2 chars.  Good
//!   default for English-leaning tests and for environments where a
//!   morphological analyzer is unavailable.
//! * [`LinderaMorphology`] (feature `lindera`, default on) — Japanese
//!   noun extraction via lindera + IPADIC, with a character-class-based
//!   fallback identical to `analyzer::text::extract_nouns` in the source
//!   repository when the dictionary is unavailable at runtime.
//!
//! Implementations are expected to be **deterministic** and **side-effect
//! free**.

/// Pluggable content-noun extraction strategy．
pub trait Morphology {
    /// Extract content nouns from `text`.  Returns an empty `Vec` when none
    /// are found.  Implementations should NOT panic on any input.
    fn extract_nouns(&self, text: &str) -> Vec<String>;
}

// =========================================================================
// WhitespaceMorphology — always available, no features.
// =========================================================================

/// Tiny built-in implementation: ASCII-whitespace split, lowercase, length>=2.
///
/// This is the default for tests and for environments where a real
/// morphological analyzer is unavailable.  It is intentionally minimal — it
/// is not appropriate for Japanese-heavy inputs (use [`LinderaMorphology`]
/// for that).
pub struct WhitespaceMorphology;

impl WhitespaceMorphology {
    /// Construct a new whitespace-based morphology．
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhitespaceMorphology {
    fn default() -> Self {
        Self::new()
    }
}

impl Morphology for WhitespaceMorphology {
    fn extract_nouns(&self, text: &str) -> Vec<String> {
        text.split_ascii_whitespace()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s.chars().count() >= 2)
            .collect()
    }
}

// =========================================================================
// LinderaMorphology — feature-gated Japanese tokenizer.
// =========================================================================

#[cfg(feature = "lindera")]
pub use lindera_impl::LinderaMorphology;

#[cfg(feature = "lindera")]
mod lindera_impl {
    use super::Morphology;
    use std::sync::OnceLock;

    use lindera::tokenizer::Tokenizer;

    /// Lindera-backed Japanese noun extractor with character-class fallback.
    ///
    /// Behaviour mirrors the original `analyzer::text::extract_nouns(text,
    /// enable_japanese=true)` from `persona-from-slack`:
    /// * If IPADIC is available at runtime, run lindera and keep tokens whose
    ///   first detail field equals `"名詞"` (noun).
    /// * If IPADIC is missing, fall back to the character-class heuristic
    ///   (CJK ideograph + katakana runs + ASCII technical words minus stop
    ///   words).
    ///
    /// The tokenizer is initialized lazily and cached in a process-global
    /// `OnceLock` (matching the original implementation).
    pub struct LinderaMorphology {
        // unit struct; the cached tokenizer lives in `try_tokenizer()`'s
        // OnceLock so cloning / multiple instances share one initialization.
        _private: (),
    }

    impl LinderaMorphology {
        /// Construct a new lindera-backed morphology．Tokenizer initialization
        /// happens lazily on the first call to [`Morphology::extract_nouns`]．
        pub fn new() -> Self {
            Self { _private: () }
        }
    }

    impl Default for LinderaMorphology {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Morphology for LinderaMorphology {
        fn extract_nouns(&self, text: &str) -> Vec<String> {
            if let Some(tok) = try_tokenizer() {
                extract_nouns_lindera(tok, text)
            } else {
                extract_nouns_fallback(text)
            }
        }
    }

    fn try_tokenizer() -> Option<&'static Tokenizer> {
        static INSTANCE: OnceLock<Option<Tokenizer>> = OnceLock::new();
        INSTANCE
            .get_or_init(|| {
                let config = serde_json::json!({
                    "segmenter": {
                        "dictionary": {
                            "kind": "ipadic"
                        },
                        "mode": "normal"
                    },
                    "character_filters": [],
                    "token_filters": []
                });
                match Tokenizer::from_config(&config) {
                    Ok(tok) => Some(tok),
                    Err(e) => {
                        tracing::warn!(
                            "lindera tokenizer unavailable (IPADIC dictionary not bundled): {}. \
                             Falling back to character-class-based noun extraction.",
                            e
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    fn extract_nouns_lindera(tok: &Tokenizer, text: &str) -> Vec<String> {
        let tokens = match tok.tokenize(text) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut nouns = Vec::new();
        for token in &tokens {
            // IPADIC detail format: "品詞,品詞細分類1,...,原形,読み,発音".
            if let Some(ref details) = token.details {
                if let Some(first) = details.first() {
                    if first.as_ref() == "名詞" {
                        nouns.push(token.text.to_string());
                    }
                }
            }
        }
        nouns
    }

    /// Fallback noun extraction using Unicode character classes．Mirrors the
    /// source `analyzer::text::extract_nouns_fallback` exactly．
    fn extract_nouns_fallback(text: &str) -> Vec<String> {
        let mut nouns = Vec::new();
        let mut current = String::new();
        let mut current_kind: Option<CharKind> = None;

        for c in text.chars() {
            let kind = classify_char(c);
            match kind {
                CharKind::Kanji | CharKind::Katakana => {
                    if current_kind == Some(kind) || current_kind.is_none() || current.is_empty() {
                        current.push(c);
                        current_kind = Some(kind);
                    } else {
                        if !current.is_empty() {
                            nouns.push(std::mem::take(&mut current));
                        }
                        current.push(c);
                        current_kind = Some(kind);
                    }
                }
                _ => {
                    if !current.is_empty() {
                        nouns.push(std::mem::take(&mut current));
                        current_kind = None;
                    }
                }
            }
        }
        if !current.is_empty() {
            nouns.push(current);
        }

        for word in text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            let trimmed = word.trim();
            if trimmed.len() >= 2
                && trimmed
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                let lower = trimmed.to_lowercase();
                if !is_english_stop_word(&lower) {
                    nouns.push(trimmed.to_string());
                }
            }
        }
        nouns
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CharKind {
        Kanji,
        Katakana,
        Other,
    }

    fn classify_char(c: char) -> CharKind {
        let cp = c as u32;
        if is_cjk_ideograph(cp) {
            CharKind::Kanji
        } else if is_katakana(cp) {
            CharKind::Katakana
        } else {
            CharKind::Other
        }
    }

    fn is_cjk_ideograph(cp: u32) -> bool {
        matches!(
            cp,
            0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
        )
    }

    fn is_katakana(cp: u32) -> bool {
        matches!(cp, 0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF65..=0xFF9F)
    }

    fn is_english_stop_word(word: &str) -> bool {
        matches!(
            word,
            "the"
                | "a"
                | "an"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "have"
                | "has"
                | "had"
                | "do"
                | "does"
                | "did"
                | "will"
                | "would"
                | "shall"
                | "should"
                | "may"
                | "might"
                | "must"
                | "can"
                | "could"
                | "of"
                | "in"
                | "to"
                | "for"
                | "with"
                | "on"
                | "at"
                | "by"
                | "from"
                | "as"
                | "into"
                | "through"
                | "during"
                | "before"
                | "after"
                | "and"
                | "but"
                | "or"
                | "nor"
                | "not"
                | "so"
                | "if"
                | "then"
                | "than"
                | "that"
                | "this"
                | "it"
                | "its"
                | "he"
                | "she"
                | "we"
                | "they"
                | "me"
                | "him"
                | "her"
                | "us"
                | "them"
                | "my"
                | "your"
                | "his"
                | "our"
                | "their"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_morphology_basic() {
        let m = WhitespaceMorphology::new();
        let ns = m.extract_nouns("Hello World rust API a");
        assert!(ns.contains(&"hello".to_string()));
        assert!(ns.contains(&"world".to_string()));
        assert!(ns.contains(&"rust".to_string()));
        assert!(ns.contains(&"api".to_string()));
        // single-char "a" dropped
        assert!(!ns.contains(&"a".to_string()));
    }

    #[test]
    fn test_whitespace_morphology_empty() {
        let m = WhitespaceMorphology;
        assert!(m.extract_nouns("").is_empty());
        assert!(m.extract_nouns("   ").is_empty());
    }

    #[cfg(feature = "lindera")]
    #[test]
    fn test_lindera_morphology_extracts_something() {
        let m = LinderaMorphology::new();
        // Either lindera (IPADIC) or the fallback must yield some CJK content.
        let ns = m.extract_nouns("東京タワーは日本の観光名所です");
        assert!(!ns.is_empty(), "expected nouns from Japanese text");
    }
}
