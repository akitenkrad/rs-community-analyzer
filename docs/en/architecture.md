**English** | [日本語](../ja/architecture.md)

# Architecture

`community-analyzer` is stateless and in-memory. All `compute_h*` functions
receive an `AnalysisInput` of borrowed in-memory slices; the library never
touches a database or a specific platform API.

See also: [Use cases](use-cases.md) / [Hypotheses](hypotheses.md) /
[NLP backend](nlp.md) / [Ethics](ethics.md).

## Data model

| Type | Purpose |
|------|---------|
| `Message` | One chat message. `timestamp` is `DateTime<Utc>`. Threading via `thread_root_id: Option<String>` (see below). Carries `reaction_count` for the aggregate count. |
| `Channel` | Channel / room / forum. The caller fills `category` (`Option<ChannelCategory>`) and `is_decision_channel` from config or platform metadata. |
| `User` | The caller pre-resolves `role` (`Exec` / `Manager` / `Lead` / `Staff` / `Unknown`). |
| `Reaction` | First-class type. Carries `message_id` + `channel_id` + `user_id` + `emoji_name`. No JSON parsing. |

`AnalysisInput<'a>` bundles `&[Message]`, `&[Channel]`, `&[User]`,
`&[Reaction]`, and `&CommConfig`. It borrows the caller's data — no ownership
transfer.

### Thread convention

A top-level message has `thread_root_id == None`. A reply has
`thread_root_id == Some(root_message_id)`. Adapters may equivalently encode a
root as `Some(self.id)`; `analysis::group_threads` accepts both shapes.

### Opaque identifiers

All IDs (`Message::id`, `channel_id`, `author_id`, `User::id`, `Channel::id`,
`Reaction::user_id`, `Reaction::message_id`) are opaque strings. The library
never parses or assumes their shape: Slack-style IDs (`"U0..."`, `"C0..."`),
UUIDs, integer-as-string, email addresses — all work unchanged.

## Module overview

| Module | Responsibility |
|--------|----------------|
| `types` | Platform-agnostic input types and the `AnalysisInput` bundle. |
| `config` | `CommConfig` (TOML): role / channel mappings, detection patterns, NLP and output settings. |
| `text` | `PatternMatcher` and text utilities for phrase / marker detection. |
| `analysis` | Thread grouping (`group_threads`) and shared analysis helpers. |
| `graph` | `ReplyGraph` — reply network, PageRank centrality, hierarchy scoring. |
| `metrics` | `compute_h1..h5`, `enrich_h1/h4/h5`, and `build_summary`. |
| `morphology` | `Morphology` trait — `WhitespaceMorphology` / `LinderaMorphology`. |
| `nlp` | `Nlp` trait — `MockNlp` / `CandleNlp`, plus weight download helpers. |
| `report` | `write_comm_report` and `finalize_report_assets`. |
| `figures` | SVG chart generation for the report. |
| `anonymize` | `Anonymizer` — salted hashing of user IDs for output. |

## Analysis pipeline

1. **Compute** — `compute_h1`, `compute_h2`, `compute_h3`, `compute_h4` take
   `&AnalysisInput`; `compute_h5` additionally takes a `&dyn Morphology`. These
   are pure Rust and need no NLP backend.
2. **Enrich** (optional) — `enrich_h1`, `enrich_h4`, `enrich_h5` take a
   `&dyn Nlp` and fill in the embedding / sentiment / clustering fields that
   the Rust-only pass leaves as `None` / `0.0`. They are **synchronous**.
3. **Summarize** — `build_summary` aggregates whichever hypotheses were
   computed into a `ReportSummary` with `overall_health_score` and
   red / yellow / green flags.
4. **Report** — `report::write_comm_report` writes `report.md`;
   `report::finalize_report_assets` generates figures and the anonymized
   PageRank CSV (see [Ethics](ethics.md)).
