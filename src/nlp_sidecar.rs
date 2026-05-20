//! Async client for the Python NLP sidecar．
//!
//! The sidecar (`tools/comm/src/nlp_sidecar.py`) is spawned as a child process
//! and speaks JSONL: one request object per stdin line, one response object
//! per stdout line, FIFO order．Heavy ML work happens entirely in Python;
//! Rust only ferries serialized requests/responses．
//!
//! Failure handling: spawn / IO / serde errors map to [`CommError::Nlp`]; an
//! `error`-task response from a [`NlpSidecar::request`] becomes
//! `Err(CommError::Nlp(message))`．Callers in the enrichment layer treat any
//! such error as graceful degradation (leave the metric `None`)．

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::error::{CommError, Result};

/// A single request to the sidecar．Serialized as one JSON line．
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "task", rename_all = "lowercase")]
pub enum NlpRequest {
    Stance {
        id: String,
        text: String,
        context: Option<String>,
    },
    Sentiment {
        id: String,
        text: String,
    },
    Embed {
        id: String,
        texts: Vec<String>,
    },
    Cluster {
        embeddings: Vec<Vec<f32>>,
        min_cluster_size: usize,
    },
}

/// A single response from the sidecar．Deserialized from one JSON line．
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "task", rename_all = "lowercase")]
pub enum NlpResponse {
    Stance {
        id: String,
        label: String,
        score: f32,
    },
    Sentiment {
        id: String,
        polarity: f32,
        magnitude: f32,
    },
    Embed {
        id: String,
        vectors: Vec<Vec<f32>>,
    },
    Cluster {
        labels: Vec<i32>,
        num_clusters: u32,
    },
    Error {
        #[serde(default)]
        id: String,
        message: String,
    },
}

/// A live handle to a spawned NLP sidecar child process．
pub struct NlpSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl NlpSidecar {
    /// Spawn the sidecar from a [`NlpSidecarConfig`](crate::config::NlpSidecarConfig)．
    ///
    /// `cfg.python` is split on whitespace into program + leading args (e.g．
    /// `"uv run python"` -> `uv run python`)，then `cfg.script`, `--profile
    /// <profile>`, `--device <device>` are appended．When the environment
    /// variable `COMM_NLP_MOCK=1` is set，`--mock` is also appended (stdlib-only
    /// deterministic mode — used by tests/CI)．
    pub async fn spawn(cfg: &crate::config::NlpSidecarConfig) -> Result<Self> {
        let mut parts = cfg.python.split_whitespace();
        let program = parts
            .next()
            .ok_or_else(|| CommError::Nlp("empty python command".into()))?;
        let mut args: Vec<String> = parts.map(|s| s.to_string()).collect();
        args.push(cfg.script.clone());
        args.push("--profile".into());
        args.push(cfg.profile.clone());
        args.push("--device".into());
        args.push(cfg.device.clone());
        if std::env::var("COMM_NLP_MOCK").as_deref() == Ok("1") {
            args.push("--mock".into());
        }
        Self::spawn_cmd(program, &args).await
    }

    /// Spawn an explicit `program` with explicit `args` (test/diagnostic ctor)．
    pub async fn spawn_cmd(program: &str, args: &[String]) -> Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| CommError::Nlp(format!("spawn {program}: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CommError::Nlp("sidecar stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CommError::Nlp("sidecar stdout unavailable".into()))?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    async fn write_request(&mut self, req: &NlpRequest) -> Result<()> {
        let mut line = serde_json::to_string(req)
            .map_err(|e| CommError::Nlp(format!("serialize request: {e}")))?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| CommError::Nlp(format!("write request: {e}")))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| CommError::Nlp(format!("flush request: {e}")))?;
        Ok(())
    }

    async fn read_response(&mut self) -> Result<NlpResponse> {
        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .await
            .map_err(|e| CommError::Nlp(format!("read response: {e}")))?;
        if n == 0 {
            return Err(CommError::Nlp("sidecar closed stdout (EOF)".into()));
        }
        serde_json::from_str(line.trim())
            .map_err(|e| CommError::Nlp(format!("deserialize response {line:?}: {e}")))
    }

    /// Send one request and read its single response (FIFO)．
    pub async fn request(&mut self, req: NlpRequest) -> Result<NlpResponse> {
        self.write_request(&req).await?;
        let resp = self.read_response().await?;
        if let NlpResponse::Error { message, .. } = &resp {
            return Err(CommError::Nlp(message.clone()));
        }
        Ok(resp)
    }

    /// Send all `reqs` then read exactly `reqs.len()` responses (preserving order)．
    pub async fn batch_request(&mut self, reqs: Vec<NlpRequest>) -> Result<Vec<NlpResponse>> {
        for r in &reqs {
            self.write_request(r).await?;
        }
        let mut out = Vec::with_capacity(reqs.len());
        for _ in 0..reqs.len() {
            out.push(self.read_response().await?);
        }
        Ok(out)
    }

    /// Close stdin (signals EOF) and wait for the child to exit．
    pub async fn shutdown(mut self) -> Result<()> {
        drop(self.stdin);
        self.child
            .wait()
            .await
            .map_err(|e| CommError::Nlp(format!("wait child: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar_path() -> String {
        // crate root = CARGO_MANIFEST_DIR (single crate, no workspace layer).
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest)
            .join("tools/comm/src/nlp_sidecar.py")
            .to_string_lossy()
            .into_owned()
    }

    fn python3() -> Option<String> {
        let out = std::process::Command::new("which").arg("python3").output();
        match out {
            Ok(o) if o.status.success() => {
                let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if p.is_empty() {
                    None
                } else {
                    Some(p)
                }
            }
            _ => None,
        }
    }

    #[test]
    fn test_request_serde_tags() {
        let s = serde_json::to_string(&NlpRequest::Stance {
            id: "x".into(),
            text: "t".into(),
            context: None,
        })
        .unwrap();
        assert!(s.contains("\"task\":\"stance\""));

        let s = serde_json::to_string(&NlpRequest::Sentiment {
            id: "x".into(),
            text: "t".into(),
        })
        .unwrap();
        assert!(s.contains("\"task\":\"sentiment\""));

        let s = serde_json::to_string(&NlpRequest::Embed {
            id: "x".into(),
            texts: vec!["a".into()],
        })
        .unwrap();
        assert!(s.contains("\"task\":\"embed\""));

        let s = serde_json::to_string(&NlpRequest::Cluster {
            embeddings: vec![vec![0.1, 0.2]],
            min_cluster_size: 2,
        })
        .unwrap();
        assert!(s.contains("\"task\":\"cluster\""));
    }

    #[test]
    fn test_response_deserialize_including_error() {
        let r: NlpResponse =
            serde_json::from_str(r#"{"task":"stance","id":"a","label":"neutral","score":0.5}"#)
                .unwrap();
        assert!(matches!(r, NlpResponse::Stance { label, .. } if label == "neutral"));

        let r: NlpResponse = serde_json::from_str(
            r#"{"task":"sentiment","id":"b","polarity":-0.2,"magnitude":0.4}"#,
        )
        .unwrap();
        assert!(matches!(r, NlpResponse::Sentiment { .. }));

        let r: NlpResponse =
            serde_json::from_str(r#"{"task":"embed","id":"c","vectors":[[1.0,0.0]]}"#).unwrap();
        assert!(matches!(r, NlpResponse::Embed { .. }));

        let r: NlpResponse =
            serde_json::from_str(r#"{"task":"cluster","labels":[0,1],"num_clusters":2}"#).unwrap();
        assert!(matches!(
            r,
            NlpResponse::Cluster {
                num_clusters: 2,
                ..
            }
        ));

        let r: NlpResponse =
            serde_json::from_str(r#"{"task":"error","id":"","message":"boom"}"#).unwrap();
        assert!(matches!(r, NlpResponse::Error { message, .. } if message == "boom"));
    }

    #[tokio::test]
    async fn test_mock_sidecar_roundtrip() {
        let py = match python3() {
            Some(p) => p,
            None => {
                eprintln!("skip: python3 not found");
                return;
            }
        };
        let args = vec![sidecar_path(), "--mock".to_string()];
        let mut sc = NlpSidecar::spawn_cmd(&py, &args)
            .await
            .expect("spawn mock sidecar");

        let r = sc
            .request(NlpRequest::Stance {
                id: "s1".into(),
                text: "そうですね".into(),
                context: Some("提案です".into()),
            })
            .await
            .unwrap();
        match r {
            NlpResponse::Stance { id, label, score } => {
                assert_eq!(id, "s1");
                assert_eq!(label, "neutral");
                assert!((score - 0.5).abs() < 1e-6);
            }
            other => panic!("expected stance, got {other:?}"),
        }

        let r = sc
            .request(NlpRequest::Sentiment {
                id: "se1".into(),
                text: "これはテスト".into(),
            })
            .await
            .unwrap();
        match r {
            NlpResponse::Sentiment {
                polarity,
                magnitude,
                ..
            } => {
                assert!((-1.0..=1.0).contains(&polarity));
                assert!((0.0..=1.0).contains(&magnitude));
            }
            other => panic!("expected sentiment, got {other:?}"),
        }

        let r = sc
            .request(NlpRequest::Embed {
                id: "e1".into(),
                texts: vec!["x".into(), "y".into()],
            })
            .await
            .unwrap();
        match r {
            NlpResponse::Embed { vectors, .. } => {
                assert_eq!(vectors.len(), 2);
                assert_eq!(vectors[0].len(), 8);
            }
            other => panic!("expected embed, got {other:?}"),
        }

        let r = sc
            .request(NlpRequest::Cluster {
                embeddings: vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
                min_cluster_size: 2,
            })
            .await
            .unwrap();
        match r {
            NlpResponse::Cluster {
                labels,
                num_clusters,
            } => {
                assert!(num_clusters >= 1);
                assert_eq!(labels.len(), 3);
            }
            other => panic!("expected cluster, got {other:?}"),
        }

        let batch = sc
            .batch_request(vec![
                NlpRequest::Sentiment {
                    id: "b0".into(),
                    text: "a".into(),
                },
                NlpRequest::Sentiment {
                    id: "b1".into(),
                    text: "bb".into(),
                },
                NlpRequest::Sentiment {
                    id: "b2".into(),
                    text: "ccc".into(),
                },
            ])
            .await
            .unwrap();
        assert_eq!(batch.len(), 3);
        for (i, resp) in batch.iter().enumerate() {
            match resp {
                NlpResponse::Sentiment { id, .. } => assert_eq!(id, &format!("b{i}")),
                other => panic!("expected sentiment, got {other:?}"),
            }
        }

        sc.shutdown().await.expect("shutdown ok");
    }
}
