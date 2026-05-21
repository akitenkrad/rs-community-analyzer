[English](../en/hypotheses.md) | **日本語**

# 仮説

`community-analyzer` は，組織 / コミュニティのコミュニケーションの健全性に
ついて 5 つの仮説を評価する．各仮説は `compute_h*` 関数と `H*Metrics` 構造体に
対応する．

関連: [アーキテクチャ](architecture.md) / [ユースケース](use-cases.md) /
[NLP バックエンド](nlp.md)．

| | 名称 | 一行サマリー |
|---|------|------------------|
| **H1** | 探索 vs 正当化 | 議論が新しい結論へ動くか，最初の提案を補強するだけか． |
| **H2** | 心理的安全性 | ヘッジ / 自己防衛 / 言い淀みの密度，障害共有の速さ． |
| **H3** | 権力勾配 | 発話・返信・PageRank が一部に偏っているか，反論が管理職へ届くか． |
| **H4** | 擬似合意 | 公開合意の裏で別チャネルに反論が漏れていないか，賛同リアクション付きの本文が否定的か． |
| **H5** | 探索能力 | 新規語彙 / 提案多様性 / 未解決スレッドの比率． |

## Rust のみ vs NLP による拡充

Rust のみの `compute_h*` 関数はモデルを実行しない．したがって埋め込み，感情，
クラスタリングに依存するフィールドは，`&dyn Nlp` バックエンドとともに
`enrich_h1` / `enrich_h4` / `enrich_h5` が呼ばれるまで `None` / `0.0` のままに
なる．これらの `enrich_*` 関数は**同期的**である．拡充に依存するフィールドは，
以下で **(NLP)** と記している．

## H1 — 探索 vs 正当化 (`H1Metrics`)

`compute_h1(&input)`，その後に任意で `enrich_h1(&input, &nlp, &mut h1)`．

| フィールド | 備考 |
|-------|-------|
| `decision_change_rate` | Rust のみ． |
| `initial_proposal_regression` | **(NLP)** — 後続メッセージの最初の提案に対する埋め込み類似度． |
| `silence_after_manager` (`SilenceRatio`) | Rust のみ: `manager_post_silence_rate`，`staff_post_silence_rate`，`ratio`． |
| `dissent_convergence_speed_minutes` | Rust のみ． |
| `thread_count` | Rust のみ． |

## H2 — 心理的安全性 (`H2Metrics`)

`compute_h2(&input)` — 完全に Rust のみ（パターンベース）．

| フィールド | 備考 |
|-------|-------|
| `hedging_rate` | ヘッジ表現の密度． |
| `self_defense_rate` | 自己防衛的な言い回しの密度． |
| `incomplete_utterance_rate` | 言い淀み / 不完全な発話のマーカー． |
| `escalation_delay_minutes` | 障害の共有 / エスカレーションまでの時間． |
| `dm_dependency_ratio` (`Option<f64>`) | DM データが利用可能なときに存在する． |
| `sample_size` | 対象としたメッセージ数． |

## H3 — 権力勾配 (`H3Metrics`)

`compute_h3(&input)` — Rust のみ（グラフ / エントロピーベース）．

| フィールド | 備考 |
|-------|-------|
| `utterance_entropy_by_channel` (`Vec<(String, f64)>`) | チャンネルごとの発話エントロピー． |
| `normalized_concentration` (`Vec<(String, f64)>`) | チャンネルごとの正規化集中度． |
| `reply_concentration_gini` | 返信集中度のジニ係数． |
| `manager_reaction_rate` | 管理職に対するリアクション率． |
| `dissent_target_bias` (`DissentBias`) | `to_manager`，`to_staff`，`ratio_staff_to_manager`． |
| `hierarchy_score` | 返信グラフの階層性スコア． |
| `pagerank_topk_manager_share` | 管理職が占めるトップ k の PageRank シェア． |

## H4 — 擬似合意 (`H4Metrics`)

`compute_h4(&input)`，その後に任意で `enrich_h4(&input, &nlp, &mut h4)`．

| フィールド | 備考 |
|-------|-------|
| `surface_agreement_rate` | Rust のみ — 表面的な賛同フレーズの密度． |
| `public_private_sentiment_delta` (`Option<f64>`) | **(NLP)** — 公開チャンネルと非公開チャンネルの間の感情ギャップ． |
| `post_meeting_dissent_rate` | Rust のみ． |
| `execution_delay_hours` | Rust のみ． |
| `reaction_text_disagreement` (`Option<f64>`) | **(NLP)** — 否定的な感情を持つ本文に対する賛同リアクション． |

## H5 — 探索能力 (`H5Metrics`)

`compute_h5(&input, &morph)`（`&dyn Morphology` を受け取る），その後に任意で
`enrich_h5(&input, &nlp, &mut h5)`．

| フィールド | 備考 |
|-------|-------|
| `proposal_semantic_diversity` (`Option<f64>`) | **(NLP)** — 提案の埋め込み多様性． |
| `hypothesis_retention_period_hours` | Rust のみ． |
| `unresolved_thread_ratio` | Rust のみ． |
| `novel_vocabulary_rate_monthly` (`Vec<(String, f64)>`) | 月次の新規語彙率 — `Morphology` バックエンドを使用する． |
| `topic_cluster_count_monthly` (`Option<Vec<(String, u32)>>`) | **(NLP)** — クラスタリングによる月次のトピッククラスタ数． |

`build_summary(Some(&h1), Some(&h2), Some(&h3), Some(&h4), Some(&h5))` は，
存在する仮説をまとめて `ReportSummary`（`overall_health_score` ＋
red / yellow / green のフラグ）に集約する．
