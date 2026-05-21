[English](../en/nlp.md) | **日本語**

# NLP バックエンド

NLP は [candle](https://github.com/huggingface/candle) を介してインプロセスで
動作する．Python もサブプロセスも使わない．

関連: [ユースケース](use-cases.md) / [仮説](hypotheses.md) /
[アーキテクチャ](architecture.md)．

## `Nlp` トレイト

すべての拡充は `Nlp` トレイトを通じて行われる:

```rust
pub trait Nlp {
    fn embed(&self, texts: &[String]) -> community_analyzer::Result<Vec<Vec<f32>>>;
    fn sentiment(&self, text: &str) -> community_analyzer::Result<community_analyzer::Sentiment>;
    fn stance(&self, text: &str, context: Option<&str>)
        -> community_analyzer::Result<community_analyzer::Stance>;
    fn cluster(&self, embeddings: &[Vec<f32>], min_cluster_size: usize)
        -> community_analyzer::Result<community_analyzer::ClusterResult>;
}
```

2 つの実装が同梱されている:

* **`MockNlp`** — 決定的で，依存関係がなく，**常に利用可能**である．従来の
  モック契約をバイト単位で再現し，テストスイート / CI で使用される（モデル
  なし，ネットワークなし）．オフラインかつ再現可能な実行に最適である．
* **`CandleNlp`** — フィーチャ `nlp`（デフォルトで有効）．実際の日本語推論を
  行い，`CandleNlp::new(&cfg.nlp)?` で構築する．モデルは初回使用時に遅延
  ロードされる．

## モデル一覧

| タスク | モデル | candle バックボーン |
|------|-------|-----------------|
| embedding | Ruri v3 (ModernBERT-Ja, `cl-nagoya/ruri-v3-*`) | `models::modernbert` |
| sentiment | BERT-WRIME (`Mizuiro-sakura/bert-base-japanese-v2-wrime-fine-tune`) | `models::bert` |
| stance (NLI) | `Formzu/bert-base-japanese-jsnli` | `models::bert` |
| stance (LLM, `quality`) | Sarashina2.2-3b (`sbintuitions/sarashina2.2-3b-instruct-v0.1`) | `models::llama` (TODO) |
| clustering | HDBSCAN | `petal-clustering` |

`LUKE`-WRIME（当初の感情モデルの選択）は除外された．エンティティ認識型の
アテンションには candle の実装がないためであり，BERT-WRIME が置き換えている．

## モデルプロファイルと重み

`cfg.nlp.profile` は `fast` / `balanced`（デフォルト）/ `quality` を選択する:

| プロファイル | Embedding | Stance |
|---------|-----------|--------|
| `fast` | `cl-nagoya/ruri-v3-30m` | NLI |
| `balanced` | `cl-nagoya/ruri-v3-130m` | NLI |
| `quality` | `cl-nagoya/ruri-v3-310m` | LLM (Sarashina2.2-3b) |

実モデルによる推論には HuggingFace の重みが必要であり，初回使用時に
`~/.cache/huggingface` へ**自動ダウンロード**される（`HF_HOME` と
`HF_HUB_OFFLINE` を尊重する）．あるいは `nlp::download::download_models` /
`nlp::download::download_all` で事前取得できる．テストスイートは実際の重みを
一切ロードしない．すべてのテストは `MockNlp` を使用し，唯一の実モデル
スモークテストは `#[ignore]` 指定されている．

## ビルドとランタイムに関する注意

* このビルドは **CPU** 版の candle バックエンドのみをコンパイルする．
  GPU（`metal` / `cuda`）は candle 自身のフィーチャを介して後から有効化できる．
* 長時間の LLM 推論（`quality` プロファイル）は，非同期の呼び出し側が
  `tokio::task::spawn_blocking` でラップすべきである．`enrich_*` 関数は同期的で
  ある．
* NLP なしの軽量ビルド: `--no-default-features --features lindera`．

## 形態素解析

`compute_h5`（月次の新規語彙率）は `&dyn Morphology` を受け取る．2 つの実装が
同梱されている:

* **`WhitespaceMorphology`** — 常に利用可能である．ASCII の空白で分割する．
* **`LinderaMorphology`** — フィーチャ `lindera`（デフォルトで有効）．
  lindera + IPADIC による日本語の名詞抽出を行い，辞書が利用できない場合は
  文字クラスによるフォールバックを用いる．

このクレートは `lindera` フィーチャ**なし**でもコンパイルできる．lindera を
取り込めないビルドでは `--no-default-features` を使うとよい．
