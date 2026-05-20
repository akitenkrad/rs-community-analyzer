mod h1_justification;
mod h2_safety;
mod h3_power;
mod h4_consensus;
mod h5_diversity;
mod nlp_enrich;
mod summary;

pub use h1_justification::compute_h1;
pub use h2_safety::compute_h2;
pub use h3_power::compute_h3;
pub use h4_consensus::compute_h4;
pub use h5_diversity::compute_h5;
pub use nlp_enrich::{enrich_h1, enrich_h4, enrich_h5};
pub use summary::build_summary;
