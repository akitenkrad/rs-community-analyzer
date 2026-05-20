use thiserror::Error;

/// Error type for the `community-analyzer` crate.
#[derive(Error, Debug)]
pub enum CommError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Config error: {0}")]
    Config(String),
    #[error("Figment error: {0}")]
    Figment(String),
    #[error("NLP sidecar error: {0}")]
    Nlp(String),
    #[error("{0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, CommError>;
