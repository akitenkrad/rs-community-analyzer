![community-analyzer](docs/assets/hero.svg)

[English](README.md) | **日本語**

# community-analyzer

組織 / コミュニティのコミュニケーション分析のための，インメモリかつ
ステートレスなライブラリ．プラットフォーム非依存のチャットデータのスライスに
対して，5 つの仮説からなる健全性メトリクス（**H1** 探索 vs 正当化 /
**H2** 心理的安全性 / **H3** 権力勾配 / **H4** 擬似合意 / **H5** 探索 / アイデア
多様性）を計算する．日本語に特化したインプロセスの
[candle](https://github.com/huggingface/candle) NLP を備える（デフォルトで
有効）．

このライブラリはプラットフォーム非依存である．`Vec<Message>` / `Vec<Channel>` /
`Vec<User>` / `Vec<Reaction>` を直接受け取り，データベースや特定の
プラットフォーム API には一切アクセスしない．Slack / Discord / Teams / GitHub
などのアダプタを構築するのは呼び出し側の役割である．

## インストール

このクレートはまだ crates.io に公開されていない．当面は git 依存を使う:

```toml
[dependencies]
community-analyzer = { git = "https://github.com/akitenkrad/rs-community-analyzer" }
chrono = { version = "0.4", features = ["serde"] }
```

公開後は `community-analyzer = "0.1"` として利用できるようになる．

**フィーチャ**: デフォルトのセットは `lindera` + `nlp`（candle 推論．日本語
モデルは初回使用時に自動ダウンロードされる）．NLP モデルなしの軽量ビルドには
`--no-default-features --features lindera` を使う．

## ドキュメント

* [ユースケース](docs/ja/use-cases.md) — 実行可能なシナリオ: クイック
  ヘルスチェック，NLP を使った完全なレポート，オフライン / Rust のみのモード，
  チャットプラットフォームへの適応，図表付きの匿名化レポートの生成．
* [アーキテクチャ](docs/ja/architecture.md) — データモデル，スレッド / ID の
  規約，モジュール概要，解析パイプライン．
* [仮説](docs/ja/hypotheses.md) — H1–H5 の表と，各仮説のメトリクスフィールド
  （Rust のみ vs NLP による拡充）．
* [NLP バックエンド](docs/ja/nlp.md) — `Nlp` トレイト，`MockNlp` vs
  `CandleNlp`，モデル一覧，プロファイル，重みのダウンロード，形態素解析．
* [倫理](docs/ja/ethics.md) — 集計値のみの出力，匿名化，生 ID 漏洩なしの監査．

## ライセンス

MIT．
