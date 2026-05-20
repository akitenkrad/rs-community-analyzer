# community-analyzer

In-memory stateless library for organizational / community communication
analysis．Computes five-hypothesis health metrics (**H1** exploration-vs-
justification ／ **H2** psychological safety ／ **H3** power gradient ／
**H4** pseudo-consensus ／ **H5** exploration / idea diversity) over slices of
platform-agnostic chat data．Japanese-focused NLP via an optional Python
sidecar．

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
NLP サイドカー稼働時のみ [`enrich_h1`] / [`enrich_h4`] / [`enrich_h5`] が
それらを上書きする．

## Optional NLP sidecar

The `tools/comm/` Python sidecar provides Japanese sentiment / stance /
embedding / clustering．Setup（GB-scale ML deps；only when needed）：

```bash
cd tools/comm
uv sync
uv run python scripts/download_models.py --profile balanced  # fast / balanced / quality / all
```

Profiles select model size (cf. `tools/comm/README.md`)．For CI / tests，
**`--mock` mode** uses Python stdlib only and returns deterministic values：

```bash
python3 tools/comm/src/nlp_sidecar.py --mock
```

The Rust integration tests use this `--mock` contract verbatim．

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
