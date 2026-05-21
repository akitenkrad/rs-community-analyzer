**English** | [日本語](../ja/use-cases.md)

# Use cases

Concrete scenarios for `community-analyzer`. Every example builds an
[`AnalysisInput`](architecture.md#data-model) from in-memory `Vec`s — there is
no database and no platform client involved.

See also: [Architecture](architecture.md) / [Hypotheses](hypotheses.md) /
[NLP backend](nlp.md) / [Ethics](ethics.md).

## 1. Quick health check

Build the input from in-memory data, compute the Rust-only hypotheses, and
print the aggregate health score with its flags. `compute_h2` (psychological
safety) and `compute_h3` (power gradient) need no NLP backend.

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

A complete runnable version lives in `examples/basic.rs`
(`cargo run --example basic`).

## 2. Full report with NLP

Compute every hypothesis, then enrich the embedding / sentiment / clustering
fields with a real NLP backend. Note that `compute_h5` takes a
`&dyn Morphology` (novel-vocabulary extraction), and the `enrich_*` functions
take a `&dyn Nlp`.

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

For deterministic, offline CI runs swap `CandleNlp` for `MockNlp` — it
implements the same `Nlp` trait with no models and no network (see
[NLP backend](nlp.md)).

## 3. Rust-only / offline mode

For environments without network access (CI, air-gapped deployments) you have
two options. Either build without the NLP feature
(`--no-default-features --features lindera`) and skip the `enrich_*` step
entirely — the embedding / sentiment / clustering fields stay `None` — or keep
the feature on and use `MockNlp`, which is deterministic and never touches the
network.

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

Which hypothesis fields are Rust-only vs NLP-enriched is documented in
[Hypotheses](hypotheses.md).

## 4. Adapting a chat platform

The library is platform-agnostic. A caller maps its own records into the four
input types once. The mapping rules are the same regardless of source
(Slack / Discord / Teams / …):

* **Timestamps** become `chrono::DateTime<Utc>` (platform epoch strings or ISO
  timestamps are converted by the caller).
* **Threading** is expressed through `Message::thread_root_id`: `None` for a
  top-level message, `Some(root_id)` for a reply (a root may equivalently be
  encoded as `Some(self.id)`).
* **Reactions** are first-class `Reaction` values (`message_id` + `user_id` +
  `emoji_name`), not embedded JSON.
* **Roles and categories** are pre-resolved by the caller: set `User::role`
  (`Exec` / `Manager` / `Lead` / `Staff` / `Unknown`) and `Channel::category`
  before handing data to the library.

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

All IDs are opaque strings — the library never parses or assumes their shape,
so Slack IDs, UUIDs, integer-as-string, and email addresses all work.

## 5. Generating a report with figures + anonymization

`report::write_comm_report` writes the integrated `report.md`;
`report::finalize_report_assets` then generates the SVG figures and the
per-user PageRank CSV, appending a `## 図表` (Figures) section to the report.
Pass an [`Anonymizer`](ethics.md) so that no raw platform user ID leaks into
any output.

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

The only per-user artifact (`tables/h3_pagerank.csv`) is anonymized unless you
explicitly disable it. The runtime audit guarantees no raw ID leak — see
[Ethics](ethics.md).
