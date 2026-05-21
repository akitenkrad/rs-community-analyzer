**English** | [日本語](../ja/hypotheses.md)

# Hypotheses

`community-analyzer` evaluates five hypotheses about organizational /
community communication health. Each maps to a `compute_h*` function and an
`H*Metrics` struct.

See also: [Architecture](architecture.md) / [Use cases](use-cases.md) /
[NLP backend](nlp.md).

| | Name | One-line summary |
|---|------|------------------|
| **H1** | Exploration vs justification | Does discussion move toward a new conclusion, or merely reinforce the initial proposal? |
| **H2** | Psychological safety | Density of hedging / self-defense / hesitation, and how quickly failures are shared. |
| **H3** | Power gradient | Are utterances, replies, and PageRank concentrated in a few people, and does dissent reach managers? |
| **H4** | Pseudo-consensus | Does dissent leak into other channels behind public agreement, and is text that carries approval reactions actually negative? |
| **H5** | Exploration capacity | Ratio of novel vocabulary / proposal diversity / unresolved threads. |

## Rust-only vs NLP-enriched

The Rust-only `compute_h*` functions never run a model. Embedding-,
sentiment-, and clustering-dependent fields are therefore left as `None` /
`0.0` until `enrich_h1` / `enrich_h4` / `enrich_h5` are called with a
`&dyn Nlp` backend. These `enrich_*` functions are **synchronous**. Fields
that depend on enrichment are marked **(NLP)** below.

## H1 — Exploration vs justification (`H1Metrics`)

`compute_h1(&input)` then optionally `enrich_h1(&input, &nlp, &mut h1)`.

| Field | Notes |
|-------|-------|
| `decision_change_rate` | Rust-only. |
| `initial_proposal_regression` | **(NLP)** — embedding similarity of later messages to the initial proposal. |
| `silence_after_manager` (`SilenceRatio`) | Rust-only: `manager_post_silence_rate`, `staff_post_silence_rate`, `ratio`. |
| `dissent_convergence_speed_minutes` | Rust-only. |
| `thread_count` | Rust-only. |

## H2 — Psychological safety (`H2Metrics`)

`compute_h2(&input)` — fully Rust-only (pattern-based).

| Field | Notes |
|-------|-------|
| `hedging_rate` | Hedge-phrase density. |
| `self_defense_rate` | Self-defensive phrasing density. |
| `incomplete_utterance_rate` | Hesitation / incomplete-utterance markers. |
| `escalation_delay_minutes` | Time to share a failure / escalate. |
| `dm_dependency_ratio` (`Option<f64>`) | Present when DM data is available. |
| `sample_size` | Number of messages considered. |

## H3 — Power gradient (`H3Metrics`)

`compute_h3(&input)` — Rust-only (graph / entropy based).

| Field | Notes |
|-------|-------|
| `utterance_entropy_by_channel` (`Vec<(String, f64)>`) | Per-channel utterance entropy. |
| `normalized_concentration` (`Vec<(String, f64)>`) | Per-channel normalized concentration. |
| `reply_concentration_gini` | Gini of reply concentration. |
| `manager_reaction_rate` | Reaction rate toward managers. |
| `dissent_target_bias` (`DissentBias`) | `to_manager`, `to_staff`, `ratio_staff_to_manager`. |
| `hierarchy_score` | Reply-graph hierarchy score. |
| `pagerank_topk_manager_share` | Top-k PageRank share held by managers. |

## H4 — Pseudo-consensus (`H4Metrics`)

`compute_h4(&input)` then optionally `enrich_h4(&input, &nlp, &mut h4)`.

| Field | Notes |
|-------|-------|
| `surface_agreement_rate` | Rust-only — surface-agreement phrase density. |
| `public_private_sentiment_delta` (`Option<f64>`) | **(NLP)** — sentiment gap between public and private channels. |
| `post_meeting_dissent_rate` | Rust-only. |
| `execution_delay_hours` | Rust-only. |
| `reaction_text_disagreement` (`Option<f64>`) | **(NLP)** — agreement reactions on negative-sentiment text. |

## H5 — Exploration capacity (`H5Metrics`)

`compute_h5(&input, &morph)` (takes a `&dyn Morphology`) then optionally
`enrich_h5(&input, &nlp, &mut h5)`.

| Field | Notes |
|-------|-------|
| `proposal_semantic_diversity` (`Option<f64>`) | **(NLP)** — embedding diversity of proposals. |
| `hypothesis_retention_period_hours` | Rust-only. |
| `unresolved_thread_ratio` | Rust-only. |
| `novel_vocabulary_rate_monthly` (`Vec<(String, f64)>`) | Monthly novel-vocabulary rate — uses the `Morphology` backend. |
| `topic_cluster_count_monthly` (`Option<Vec<(String, u32)>>`) | **(NLP)** — monthly topic-cluster counts via clustering. |

`build_summary(Some(&h1), Some(&h2), Some(&h3), Some(&h4), Some(&h5))`
aggregates whichever hypotheses are present into the `ReportSummary`
(`overall_health_score` + red / yellow / green flags).
