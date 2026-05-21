[English](../en/architecture.md) | **日本語**

# アーキテクチャ

`community-analyzer` はステートレスかつインメモリで動作する．すべての
`compute_h*` 関数は，借用されたインメモリスライスからなる `AnalysisInput` を
受け取る．このライブラリはデータベースや特定のプラットフォーム API に一切
アクセスしない．

関連: [ユースケース](use-cases.md) / [仮説](hypotheses.md) /
[NLP バックエンド](nlp.md) / [倫理](ethics.md)．

## データモデル

| 型 | 役割 |
|------|---------|
| `Message` | 1 件のチャットメッセージ．`timestamp` は `DateTime<Utc>`．スレッド化は `thread_root_id: Option<String>` で表現する（後述）．集計用のカウントとして `reaction_count` を保持する． |
| `Channel` | チャンネル / ルーム / フォーラム．呼び出し側が設定やプラットフォームのメタデータから `category`（`Option<ChannelCategory>`）と `is_decision_channel` を埋める． |
| `User` | 呼び出し側があらかじめ `role`（`Exec` / `Manager` / `Lead` / `Staff` / `Unknown`）を解決しておく． |
| `Reaction` | ファーストクラスの型．`message_id` + `channel_id` + `user_id` + `emoji_name` を保持する．JSON のパースは行わない． |

`AnalysisInput<'a>` は `&[Message]`，`&[Channel]`，`&[User]`，`&[Reaction]`，
`&CommConfig` をまとめて保持する．呼び出し側のデータを借用するだけで，所有権の
移動は発生しない．

### スレッドの規約

トップレベルのメッセージは `thread_root_id == None` を持つ．返信は
`thread_root_id == Some(root_message_id)` を持つ．アダプタは等価な表現として
ルートを `Some(self.id)` としてエンコードしてもよい．`analysis::group_threads`
は両方の形式を受け付ける．

### 不透明な識別子

すべての ID（`Message::id`，`channel_id`，`author_id`，`User::id`，
`Channel::id`，`Reaction::user_id`，`Reaction::message_id`）は不透明な文字列で
ある．このライブラリはそれらの形式をパースしたり前提にしたりしない．Slack 形式
の ID（`"U0..."`，`"C0..."`），UUID，整数を文字列にしたもの，メールアドレス，
いずれもそのまま動作する．

## モジュール概要

| モジュール | 責務 |
|--------|----------------|
| `types` | プラットフォーム非依存の入力型と `AnalysisInput` のまとめ． |
| `config` | `CommConfig`（TOML）: ロール / チャンネルのマッピング，検出パターン，NLP・出力の設定． |
| `text` | フレーズ / マーカー検出のための `PatternMatcher` とテキストユーティリティ． |
| `analysis` | スレッドのグループ化（`group_threads`）と共有の解析ヘルパー． |
| `graph` | `ReplyGraph` — 返信ネットワーク，PageRank 中心性，階層性スコアリング． |
| `metrics` | `compute_h1..h5`，`enrich_h1/h4/h5`，`build_summary`． |
| `morphology` | `Morphology` トレイト — `WhitespaceMorphology` / `LinderaMorphology`． |
| `nlp` | `Nlp` トレイト — `MockNlp` / `CandleNlp`，および重みのダウンロードヘルパー． |
| `report` | `write_comm_report` と `finalize_report_assets`． |
| `figures` | レポート用の SVG チャート生成． |
| `anonymize` | `Anonymizer` — 出力用にユーザー ID をソルト付きでハッシュ化する． |

## 解析パイプライン

1. **Compute（計算）** — `compute_h1`，`compute_h2`，`compute_h3`，`compute_h4`
   は `&AnalysisInput` を受け取る．`compute_h5` はさらに `&dyn Morphology` を
   受け取る．これらは純粋な Rust であり，NLP バックエンドを必要としない．
2. **Enrich（拡充，任意）** — `enrich_h1`，`enrich_h4`，`enrich_h5` は
   `&dyn Nlp` を受け取り，Rust のみのパスで `None` / `0.0` のままになっている
   埋め込み / 感情 / クラスタリングのフィールドを埋める．これらは**同期的**で
   ある．
3. **Summarize（要約）** — `build_summary` は計算された仮説をまとめ，
   `overall_health_score` と red / yellow / green のフラグを持つ
   `ReportSummary` を生成する．
4. **Report（レポート）** — `report::write_comm_report` は `report.md` を書き
   出す．`report::finalize_report_assets` は図表と匿名化された PageRank CSV を
   生成する（[倫理](ethics.md) を参照）．
