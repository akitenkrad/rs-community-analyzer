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
pub mod nlp_sidecar;
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
pub use nlp_sidecar::{NlpRequest, NlpResponse, NlpSidecar};
pub use text::PatternMatcher;
pub use types::{AnalysisInput, Channel, Message, Reaction, User};
