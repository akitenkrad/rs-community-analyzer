[English](../en/use-cases.md) | **日本語**

# ユースケース

`community-analyzer` の具体的なシナリオ．どの例も，インメモリの `Vec` から
[`AnalysisInput`](architecture.md#データモデル) を構築する．データベースも
プラットフォームのクライアントも一切関与しない．

関連: [アーキテクチャ](architecture.md) / [仮説](hypotheses.md) /
[NLP バックエンド](nlp.md) / [倫理](ethics.md)．

## 1. クイックヘルスチェック

インメモリのデータから入力を構築し，Rust のみの仮説を計算して，集計された
健全性スコアとそのフラグを表示する．`compute_h2`（心理的安全性）と
`compute_h3`（権力勾配）は NLP バックエンドを必要としない．

```rust
use chrono::Utc;
use community_analyzer::{
    build_summary, compute_h2, compute_h3, AnalysisInput, Channel, CommConfig, Message,
    Reaction, User,
};

let cfg = CommConfig::default();
let messages: Vec<Message> = vec![/* mapped from your platform */];
let channels: Vec<Channel> = vec![/* ... */];
let users: Vec<User> = vec![/* ... */];
let reactions: Vec<Reaction> = vec![/* ... */];

let input = AnalysisInput {
    messages: &messages,
    channels: &channels,
    users: &users,
    reactions: &reactions,
    config: &cfg,
};

let h2 = compute_h2(&input)?;
let h3 = compute_h3(&input)?;

let summary = build_summary(None, Some(&h2), Some(&h3), None, None);
println!("health = {:.2}", summary.overall_health_score);
for flag in &summary.red_flags {
    println!("RED: {flag}");
}
# Ok::<(), community_analyzer::CommError>(())
```

完全に実行可能な版は `examples/basic.rs`（`cargo run --example basic`）に
ある．

## 2. NLP を使った完全なレポート

すべての仮説を計算し，その後に埋め込み / 感情 / クラスタリングのフィールドを
実際の NLP バックエンドで拡充する．`compute_h5` は `&dyn Morphology`（新規語彙
抽出）を受け取り，`enrich_*` 関数は `&dyn Nlp` を受け取る点に注意する．

```rust
use community_analyzer::{
    build_summary, compute_h1, compute_h2, compute_h3, compute_h4, compute_h5,
    enrich_h1, enrich_h4, enrich_h5, AnalysisInput, CandleNlp, WhitespaceMorphology,
};

# fn run(input: &AnalysisInput<'_>) -> Result<(), community_analyzer::CommError> {
let morph = WhitespaceMorphology;

let mut h1 = compute_h1(input)?;
let h2 = compute_h2(input)?;
let h3 = compute_h3(input)?;
let mut h4 = compute_h4(input)?;
let mut h5 = compute_h5(input, &morph)?;

// Real candle inference (feature `nlp`, default on).
// Japanese models auto-download to ~/.cache/huggingface on first use.
let nlp = CandleNlp::new(&input.config.nlp)?;
enrich_h1(input, &nlp, &mut h1)?;
enrich_h4(input, &nlp, &mut h4)?;
enrich_h5(input, &nlp, &mut h5)?;

let summary = build_summary(Some(&h1), Some(&h2), Some(&h3), Some(&h4), Some(&h5));
println!("health = {:.2}", summary.overall_health_score);
# Ok(())
# }
```

決定的でオフラインの CI 実行では `CandleNlp` を `MockNlp` に差し替える．
`MockNlp` は同じ `Nlp` トレイトを実装しており，モデルもネットワークも使わない
（[NLP バックエンド](nlp.md) を参照）．

## 3. Rust のみ / オフラインモード

ネットワークにアクセスできない環境（CI，エアギャップ環境のデプロイ）では，
2 つの選択肢がある．1 つは NLP フィーチャなしでビルドし
（`--no-default-features --features lindera`），`enrich_*` ステップを完全に
スキップする方法である．この場合，埋め込み / 感情 / クラスタリングのフィールド
は `None` のままになる．もう 1 つはフィーチャを有効にしたまま `MockNlp` を使う
方法で，これは決定的でネットワークに一切アクセスしない．

```rust
use community_analyzer::{
    compute_h1, enrich_h1, AnalysisInput, MockNlp,
};

# fn run(input: &AnalysisInput<'_>) -> Result<(), community_analyzer::CommError> {
let nlp = MockNlp; // deterministic, dependency-free, always available
let mut h1 = compute_h1(input)?;
enrich_h1(input, &nlp, &mut h1)?;
# Ok(())
# }
```

どの仮説フィールドが Rust のみで，どれが NLP による拡充かは，
[仮説](hypotheses.md) に記載されている．

## 4. チャットプラットフォームへの適応

このライブラリはプラットフォーム非依存である．呼び出し側は，自身のレコードを
4 つの入力型へ一度マッピングする．マッピングのルールはソースに関わらず同じで
ある（Slack / Discord / Teams / …）:

* **タイムスタンプ**は `chrono::DateTime<Utc>` になる（プラットフォームの
  エポック文字列や ISO タイムスタンプは呼び出し側が変換する）．
* **スレッド化**は `Message::thread_root_id` で表現する: トップレベルの
  メッセージは `None`，返信は `Some(root_id)`（ルートは等価な表現として
  `Some(self.id)` とエンコードしてもよい）．
* **リアクション**はファーストクラスの `Reaction` 値（`message_id` +
  `user_id` + `emoji_name`）であり，埋め込まれた JSON ではない．
* **ロールとカテゴリ**は呼び出し側があらかじめ解決する: ライブラリへデータを
  渡す前に `User::role`（`Exec` / `Manager` / `Lead` / `Staff` / `Unknown`）と
  `Channel::category` を設定する．

```rust
use chrono::Utc;
use community_analyzer::{Channel, ChannelCategory, Message, Reaction, Role, User};

// Example: map a generic platform record into a `Message`.
# struct PlatformMsg { id: String, channel: String, author: String,
#     body: String, sent_at: chrono::DateTime<Utc>, parent: Option<String>, n_reacts: usize }
fn to_message(raw: PlatformMsg) -> Message {
    Message {
        id: raw.id,
        channel_id: raw.channel,
        author_id: raw.author,
        text: raw.body,
        timestamp: raw.sent_at,          // already DateTime<Utc>
        thread_root_id: raw.parent,      // None => top-level, Some(root) => reply
        reaction_count: raw.n_reacts,
    }
}

let _user = User { id: "u1".into(), display_name: "Alice".into(), role: Role::Manager };
let _channel = Channel {
    id: "c1".into(),
    name: "proj-x".into(),
    category: Some(ChannelCategory::Official),
    is_decision_channel: true,
};
let _reaction = Reaction {
    message_id: "m1".into(),
    channel_id: "c1".into(),
    user_id: "u2".into(),
    emoji_name: "thumbsup".into(), // surrounding ':' stripped, lowercased
};
```

すべての ID は不透明な文字列である．ライブラリはそれらの形式をパースしたり
前提にしたりしないため，Slack ID，UUID，整数を文字列にしたもの，メール
アドレスはいずれも動作する．

## 5. 図表 ＋ 匿名化付きのレポート生成

`report::write_comm_report` は統合された `report.md` を書き出す．
`report::finalize_report_assets` はその後，SVG 図表とユーザー単位の PageRank
CSV を生成し，レポートに `## 図表` セクションを追記する．生のプラットフォーム
のユーザー ID がいずれの出力にも漏洩しないように，[`Anonymizer`](ethics.md) を
渡す．

```rust
use std::path::Path;
use community_analyzer::{
    report, Anonymizer, CommReport, ReplyGraph,
};

# fn run(rep: &CommReport, graph: &ReplyGraph, n_channels: usize, n_messages: usize)
#     -> Result<(), community_analyzer::CommError> {
let dir = Path::new("output/comm");
let nlp_used = true;

report::write_comm_report(dir, "2025-04 .. 2026-03", n_channels, n_messages, rep, nlp_used)?;

// Anonymized by default: a random per-run salt, never persisted.
let anon = Anonymizer::new(true);
let assets = report::finalize_report_assets(dir, rep, Some(graph), &anon, 20)?;
println!("generated {} assets", assets.len());
# Ok(())
# }
```

ユーザー単位の唯一の成果物（`tables/h3_pagerank.csv`）は，明示的に無効化
しない限り匿名化される．ランタイム監査が生 ID の漏洩がないことを保証する
（[倫理](ethics.md) を参照）．
