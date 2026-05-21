#![doc = include_str!("../README.md")]

pub mod analysis;
pub mod anonymize;
pub mod config;
pub mod error;
pub mod figures;
pub mod graph;
pub mod metrics;
pub mod models;
pub mod morphology;
pub mod nlp;
pub mod report;
pub mod text;
pub mod types;

pub use anonymize::Anonymizer;
pub use config::CommConfig;
pub use error::{CommError, Result};
pub use graph::ReplyGraph;
pub use metrics::{
    build_summary, compute_h1, compute_h2, compute_h3, compute_h4, compute_h5, enrich_h1,
    enrich_h4, enrich_h5,
};
pub use models::*;
#[cfg(feature = "lindera")]
pub use morphology::LinderaMorphology;
pub use morphology::{Morphology, WhitespaceMorphology};
#[cfg(feature = "nlp")]
pub use nlp::CandleNlp;
pub use nlp::{ClusterResult, MockNlp, Nlp, Sentiment, Stance, StanceLabel};
pub use text::PatternMatcher;
pub use types::{AnalysisInput, Channel, Message, Reaction, User};
