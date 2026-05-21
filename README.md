# community-analyzer

In-memory stateless library for organizational / community communication
analysis．Computes five-hypothesis health metrics (**H1** exploration-vs-
justification ／ **H2** psychological safety ／ **H3** power gradient ／
**H4** pseudo-consensus ／ **H5** exploration / idea diversity) over slices of
platform-agnostic chat data．Japanese-focused NLP via in-process
[candle](https://github.com/huggingface/candle) inference (feature `nlp`,
default on)．

The library is platform-agnostic：it takes `Vec<Message>` / `Vec<Channel>` /
`Vec<User>` / `Vec<Reaction>` directly and never touches a database or a
specific platform API．Adapters for Slack / Discord / Teams / GitHub etc．are
the caller's job．

## Origin

Extracted from
[github.com/akitenkrad/persona-from-slack](https://github.com/akitenkrad/persona-from-slack)
（commit `8f75cc9` of the `comm-analyzer` crate）．Original design：
「組織コミュニケーション分析ライブラリ設計書」(2026-05-14)．This 0.1.0
release is a stateless library extraction：the `storage::Database` integration
is intentionally dropped；persistence is the caller's responsibility．

## Quick start

```toml
[dependencies]
community-analyzer = "0.1"            # publish target；until then use a git dep
chrono = { version = "0.4", features = ["serde"] }
```

```rust,no_run
use chrono::Utc;
use community_analyzer::{
    build_summary, compute_h2, compute_h3, AnalysisInput, Channel, CommConfig, Message,
    Reaction, User,
};

let cfg = CommConfig::default();
let messages: Vec<Message> = vec![/* ... */];
let channels: Vec<Channel> = vec![/* ... */];
let users: Vec<User>       = vec![/* ... */];
let reactions: Vec<Reaction> = vec![/* ... */];

let input = AnalysisInput {
    messages: &messages,
    channels: &channels,
    users: &users,
    reactions: &reactions,
    config: &cfg,
};

let h2 = compute_h2(&input).unwrap();
let h3 = compute_h3(&input).unwrap();
let summary = build_summary(None, Some(&h2), Some(&h3), None, None);
println!("health = {:.2}", summary.overall_health_score);
```

A complete runnable example lives in [`examples/basic.rs`]．Run it with
`cargo run --example basic`．

## Data model

| Type | Purpose |
|------|---------|
| `Message` | One chat message．Timestamp is `DateTime<Utc>`．Threading via `thread_root_id: Option<String>` (see below)． |
| `Channel` | Channel / room / forum．Caller fills `category` and `is_decision_channel` from config or platform metadata． |
| `User` | Caller pre-resolves `role` (`Exec` / `Manager` / `Lead` / `Staff` / `Unknown`)． |
| `Reaction` | First-class type．Carries `message_id` + `user_id` + `emoji_name`．No JSON parsing． |

**Thread convention**：a top-level message has `thread_root_id == None`．
A reply has `thread_root_id == Some(root_message_id)`．Adapters may
equivalently encode a root as `Some(self.id)`；
[`analysis::group_threads`] accepts both shapes．

All IDs (`Message::id`, `channel_id`, `author_id`, `User::id`, `Channel::id`,
`Reaction::user_id`, `Reaction::message_id`) are opaque strings．The library
never parses or assumes their shape：Slack IDs, UUIDs, integer-as-string,
email addresses, … all work．

## Hypotheses

| | Name | One-line summary |
|---|------|------------------|
| **H1** | 探索 vs 正当化 | 議論が新しい結論へ動くか，最初の提案を補強するだけか． |
| **H2** | 心理的安全性 | ヘッジ / 自己防衛 / 言い淀みの密度，障害共有の速さ． |
| **H3** | 権力勾配 | 発話・返信・PageRank が一部に偏っているか，反論が管理職へ届くか． |
| **H4** | 擬似合意 | 公開合意の裏で別チャネルに反論が漏れていないか，賛同リアクション付きの本文が否定的か． |
| **H5** | 探索能力 | 新規語彙 / 提案多様性 / 未解決スレッドの比率． |

H1 の埋め込み依存フィールドと H4/H5 の sentiment / クラスタリング依存フィー
ルドは Rust-only の `compute_h*` では計算されず，`None` / `0.0` のまま残る．
NLP バックエンド供給時のみ [`enrich_h1`] / [`enrich_h4`] / [`enrich_h5`] が
それらを上書きする．これらは **同期** 関数で，`&dyn Nlp` を受け取る．

## NLP backend

NLP is in-process via candle (no Python, no subprocess)．The [`Nlp`] trait has
two implementations：

* [`MockNlp`] — deterministic, dependency-free, **always available**．
  Reproduces the historical mock contract byte-for-byte；used by the test
  suite / CI (no models, no network)．
* [`CandleNlp`] — feature `nlp` (default on)．Real Japanese inference：

  | task | model | candle backbone |
  |------|-------|-----------------|
  | embedding | Ruri v3 (ModernBERT-Ja) | `models::modernbert` |
  | sentiment | BERT-WRIME (`Mizuiro-sakura/bert-base-japanese-v2-wrime-fine-tune`) | `models::bert` |
  | stance (NLI) | `Formzu/bert-base-japanese-jsnli` | `models::bert` |
  | stance (LLM, `quality`) | Sarashina2.2-3b | `models::llama` (TODO) |
  | clustering | HDBSCAN | `petal-clustering` |

  `LUKE`-WRIME (the original sentiment pick) was dropped — its entity-aware
  attention has no candle implementation；BERT-WRIME replaces it．

```rust,ignore
use community_analyzer::{CandleNlp, MockNlp, enrich_h1};

// Tests / CI — deterministic, offline:
let nlp = MockNlp;
enrich_h1(&input, &nlp, &mut h1)?;

// Production — real candle inference (feature `nlp`):
let nlp = CandleNlp::new(&cfg.nlp)?;          // models load lazily on first use
enrich_h1(&input, &nlp, &mut h1)?;
```

### Model profiles & weights

`cfg.nlp.profile` selects `fast` / `balanced` (default) / `quality`．Real-model
inference needs HuggingFace weights — these are **auto-downloaded** to
`~/.cache/huggingface` on first use (honouring `HF_HOME` / `HF_HUB_OFFLINE`)，
or pre-fetched via [`nlp::download::download_models`] /
[`nlp::download::download_all`]．**The test suite never loads real weights**：
all tests use `MockNlp`，and the one real-model smoke test is `#[ignore]`d．

This build compiles the **CPU** candle backend only．GPU (`metal` / `cuda`)
can be enabled later via candle's own features．Long LLM inference (the
`quality` profile) should be wrapped in `tokio::task::spawn_blocking` by an
async caller — the `enrich_*` functions are synchronous．

Lightweight builds without NLP：`--no-default-features --features lindera`．

## Morphology

`compute_h5` (monthly novel-vocabulary rate) takes a `&dyn Morphology`．Two
implementations ship：

* `WhitespaceMorphology` — always available．Splits on ASCII whitespace．
* `LinderaMorphology` — feature `lindera` (default on)．Japanese noun
  extraction via lindera + IPADIC，with a character-class fallback when the
  dictionary is unavailable．

The crate compiles **without** the `lindera` feature too — use
`--no-default-features` for builds that can't bring in lindera．

## Ethics

* This library is for **detecting structural distortion** in organizational
  / community communication．It **must not** be repurposed for individual
  evaluation or punishment．
* Reports show **aggregate values only**；the only per-user artifact
  (`tables/h3_pagerank.csv`) is **anonymized by default** with a random
  per-run salt that is never persisted．
* The runtime audit test `test_finalize_report_assets_no_raw_id_leak`
  enforces that no raw platform user ID may appear in `report.md`,
  `metrics.json`, the figures, or the CSV unless anonymization is
  explicitly disabled．

## Status

`0.1.0` — extracted from `persona-from-slack@8f75cc9` as a stateless
library．Platform adapters (Slack / Discord / …) remain in upstream
projects．

---
*This file was generated by Claude Code.*
