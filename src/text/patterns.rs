//! Pattern matching over message text using Aho-Corasick automata．
//!
//! One automaton is built per detection category．Categories sourced from the
//! configurable [`PatternDict`](crate::config::PatternDict) plus built-in
//! "dissent" / "commitment" / "action mention" lists that are *not* part of
//! the user-facing dictionary (they are stable defaults)．

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};

use crate::config::PatternDict;

/// Built-in Japanese dissent / counter-argument cues．
const DISSENT_PATTERNS: &[&str] = &[
    "ただ",
    "しかし",
    "とはいえ",
    "とは言え",
    "逆に",
    "気になるのは",
    "でも",
    "ですが",
    "別の見方",
    "懸念",
    "とはいうものの",
    "とはいえども",
];

/// Built-in Japanese commitment / promise-to-act cues．
const COMMITMENT_PATTERNS: &[&str] = &[
    "やります",
    "やってみます",
    "進めます",
    "対応します",
    "対応いたします",
    "対応します！",
    "やっておきます",
    "やっておく",
    "引き取ります",
    "巻き取ります",
    "着手します",
    "進めておきます",
    "やらせていただきます",
    "対応する",
];

/// Built-in Japanese "action completed / mentioned" cues．
const ACTION_MENTION_PATTERNS: &[&str] = &[
    "完了",
    "完了しました",
    "対応済",
    "対応しました",
    "対応した",
    "修正しました",
    "修正した",
    "終わりました",
    "終わった",
    "リリース",
    "リリースしました",
    "マージ",
    "マージしました",
    "done",
    "クローズ",
    "close",
    "解決",
    "解決しました",
    "済みました",
    "片付けました",
];

/// Compiled multi-category matcher．
pub struct PatternMatcher {
    hedging: AhoCorasick,
    self_defense: AhoCorasick,
    incomplete: AhoCorasick,
    surface_agreement: AhoCorasick,
    unresolved_marker: AhoCorasick,
    conclusion_marker: AhoCorasick,
    dissent: AhoCorasick,
    commitment: AhoCorasick,
    action_mention: AhoCorasick,
}

fn build_ac(patterns: &[String]) -> anyhow::Result<AhoCorasick> {
    let non_empty: Vec<&str> = patterns
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .build(&non_empty)
        .map_err(|e| anyhow::anyhow!("failed to build Aho-Corasick automaton: {e}"))?;
    Ok(ac)
}

impl PatternMatcher {
    /// Build all category automata from the configured dictionary．
    pub fn build(dict: &PatternDict) -> anyhow::Result<Self> {
        let dissent_owned: Vec<String> = DISSENT_PATTERNS.iter().map(|s| s.to_string()).collect();
        let commitment_owned: Vec<String> =
            COMMITMENT_PATTERNS.iter().map(|s| s.to_string()).collect();
        let action_mention_owned: Vec<String> = ACTION_MENTION_PATTERNS
            .iter()
            .map(|s| s.to_string())
            .collect();
        Ok(Self {
            hedging: build_ac(&dict.hedging)?,
            self_defense: build_ac(&dict.self_defense)?,
            incomplete: build_ac(&dict.incomplete)?,
            surface_agreement: build_ac(&dict.surface_agreement)?,
            unresolved_marker: build_ac(&dict.unresolved_marker)?,
            conclusion_marker: build_ac(&dict.conclusion_marker)?,
            dissent: build_ac(&dissent_owned)?,
            commitment: build_ac(&commitment_owned)?,
            action_mention: build_ac(&action_mention_owned)?,
        })
    }

    pub fn count_hedging(&self, text: &str) -> usize {
        self.hedging.find_iter(text).count()
    }
    pub fn count_self_defense(&self, text: &str) -> usize {
        self.self_defense.find_iter(text).count()
    }
    pub fn count_incomplete(&self, text: &str) -> usize {
        self.incomplete.find_iter(text).count()
    }
    pub fn contains_surface_agreement(&self, text: &str) -> bool {
        self.surface_agreement.is_match(text)
    }
    pub fn contains_unresolved_marker(&self, text: &str) -> bool {
        self.unresolved_marker.is_match(text)
    }
    pub fn contains_conclusion_marker(&self, text: &str) -> bool {
        self.conclusion_marker.is_match(text)
    }
    pub fn count_dissent(&self, text: &str) -> usize {
        self.dissent.find_iter(text).count()
    }
    pub fn count_commitment(&self, text: &str) -> usize {
        self.commitment.find_iter(text).count()
    }
    pub fn contains_commitment(&self, text: &str) -> bool {
        self.commitment.is_match(text)
    }
    pub fn count_action_mention(&self, text: &str) -> usize {
        self.action_mention.find_iter(text).count()
    }
    pub fn contains_action_mention(&self, text: &str) -> bool {
        self.action_mention.is_match(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PatternDict;

    fn matcher() -> PatternMatcher {
        PatternMatcher::build(&PatternDict::default()).expect("build matcher")
    }

    #[test]
    fn test_count_hedging() {
        let m = matcher();
        assert!(m.count_hedging("とりあえずFYIで雑に書きます") >= 2);
    }

    #[test]
    fn test_surface_agreement() {
        let m = matcher();
        assert!(m.contains_surface_agreement("承知しました"));
        assert!(!m.contains_surface_agreement("反対です"));
    }

    #[test]
    fn test_conclusion_marker() {
        let m = matcher();
        assert!(m.contains_conclusion_marker("結論はAで決定"));
    }

    #[test]
    fn test_unresolved_marker() {
        let m = matcher();
        assert!(m.contains_unresolved_marker("これは保留"));
    }

    #[test]
    fn test_count_dissent() {
        let m = matcher();
        assert!(m.count_dissent("ただ、しかし、とはいえ違う") >= 3);
    }

    #[test]
    fn test_count_incomplete() {
        let m = matcher();
        assert!(m.count_incomplete("えーと、うーん") >= 1);
    }

    #[test]
    fn test_count_commitment() {
        let m = matcher();
        assert!(m.count_commitment("対応します．あとでやっておきます") >= 2);
        assert!(m.contains_commitment("引き取ります"));
        assert!(!m.contains_commitment("ただの雑談"));
    }

    #[test]
    fn test_action_mention() {
        let m = matcher();
        assert!(m.contains_action_mention("修正しました"));
        assert!(m.count_action_mention("マージしました．クローズ") >= 2);
        assert!(!m.contains_action_mention("ただの雑談"));
    }

    #[test]
    fn test_empty_pattern_list_is_graceful() {
        let dict = PatternDict {
            hedging: vec![],
            self_defense: vec![],
            surface_agreement: vec![],
            incomplete: vec![],
            unresolved_marker: vec![],
            conclusion_marker: vec![],
        };
        let m = PatternMatcher::build(&dict).expect("build with empty patterns");
        assert_eq!(m.count_hedging("とりあえず雑に"), 0);
        assert!(!m.contains_surface_agreement("承知しました"));
        assert!(m.count_dissent("しかし") >= 1);
        assert!(m.contains_commitment("対応します"));
        assert!(m.contains_action_mention("完了しました"));
    }
}
