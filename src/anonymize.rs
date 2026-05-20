//! User-ID anonymization．
//!
//! The anonymization salt is **random per run** and is **never persisted**,
//! so identical user IDs hash to *different* `ANON-xxxxxx` codes across runs
//! (cross-run untraceable)．The reverse mapping is only available *within* a
//! single run via the run-local `anonymize_map.tsv` (chmod 0600 on unix)．
//! Sampling-reproducibility seeds (if any) are intentionally separate from
//! this salt．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::error::Result;

/// Anonymizes raw platform user IDs with a stable (within this instance)
/// salted SHA-256 prefix, recording the reverse mapping for a run-local TSV．
pub struct Anonymizer {
    enabled: bool,
    salt: [u8; 16],
    map: RefCell<BTreeMap<String, String>>,
}

impl Anonymizer {
    /// Build an anonymizer with a **random per-run** salt
    /// (`ChaCha20Rng::from_entropy`)．The salt is NOT derived from any
    /// caller-provided seed and NOT persisted．
    pub fn new(enabled: bool) -> Self {
        let mut salt = [0u8; 16];
        ChaCha20Rng::from_entropy().fill_bytes(&mut salt);
        Self {
            enabled,
            salt,
            map: RefCell::new(BTreeMap::new()),
        }
    }

    /// Build an anonymizer with a fixed salt．**Test / reproducibility only.**
    pub fn with_salt(enabled: bool, salt: [u8; 16]) -> Self {
        Self {
            enabled,
            salt,
            map: RefCell::new(BTreeMap::new()),
        }
    }

    /// Anonymize a user ID．
    ///
    /// `enabled == false` returns the input unchanged．Otherwise returns
    /// `"ANON-" + first 6 lowercase hex chars of SHA256(salt ++ user_id)`，
    /// stable within this instance and recorded in the reverse map．
    pub fn anon(&self, user_id: &str) -> String {
        if !self.enabled {
            return user_id.to_string();
        }
        if let Some(code) = self.map.borrow().get(user_id) {
            return code.clone();
        }
        let mut hasher = Sha256::new();
        hasher.update(self.salt);
        hasher.update(user_id.as_bytes());
        let digest = hasher.finalize();
        let mut code = String::with_capacity(11);
        code.push_str("ANON-");
        for b in digest.iter().take(3) {
            code.push_str(&format!("{b:02x}"));
        }
        self.map
            .borrow_mut()
            .insert(user_id.to_string(), code.clone());
        code
    }

    /// Whether anonymization is enabled．
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Write `dir/anonymize_map.tsv`．chmod 0600 on unix．If disabled or the
    /// map is empty, nothing is written．
    pub fn write_map(&self, dir: &Path) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let map = self.map.borrow();
        if map.is_empty() {
            return Ok(());
        }
        let mut rows: Vec<(&String, &String)> = map.iter().map(|(o, a)| (a, o)).collect();
        rows.sort();
        let mut out = String::new();
        out.push_str("# anon\toriginal  (run-local; salt not persisted)\n");
        for (anon, original) in rows {
            out.push_str(anon);
            out.push('\t');
            out.push_str(original);
            out.push('\n');
        }
        std::fs::create_dir_all(dir)?;
        let path = dir.join("anonymize_map.tsv");
        std::fs::write(&path, out)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn test_anon_stable_within_instance() {
        let a = Anonymizer::with_salt(true, [1u8; 16]);
        let x = a.anon("USER_ALICE");
        let y = a.anon("USER_ALICE");
        assert_eq!(x, y);
        assert_ne!(x, a.anon("USER_BOB"));
    }

    #[test]
    fn test_anon_format() {
        let a = Anonymizer::with_salt(true, [9u8; 16]);
        let re = Regex::new(r"^ANON-[0-9a-f]{6}$").unwrap();
        assert!(re.is_match(&a.anon("USER_ALICE")));
    }

    #[test]
    fn test_disabled_passthrough() {
        let a = Anonymizer::with_salt(false, [3u8; 16]);
        assert_eq!(a.anon("USER_ALICE"), "USER_ALICE");
    }

    #[test]
    fn test_with_salt_deterministic() {
        let a = Anonymizer::with_salt(true, [42u8; 16]);
        let b = Anonymizer::with_salt(true, [42u8; 16]);
        assert_eq!(a.anon("USER_ALICE"), b.anon("USER_ALICE"));
    }

    #[test]
    fn test_two_new_instances_differ() {
        let a = Anonymizer::new(true);
        let b = Anonymizer::new(true);
        assert_ne!(a.anon("USER_ALICE"), b.anon("USER_ALICE"));
    }

    #[test]
    fn test_write_map_disabled_or_empty_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let a = Anonymizer::with_salt(false, [1u8; 16]);
        a.anon("USER_ALICE");
        a.write_map(dir.path()).unwrap();
        assert!(!dir.path().join("anonymize_map.tsv").exists());

        let dir2 = tempfile::tempdir().unwrap();
        let b = Anonymizer::with_salt(true, [1u8; 16]);
        b.write_map(dir2.path()).unwrap();
        assert!(!dir2.path().join("anonymize_map.tsv").exists());
    }

    #[test]
    fn test_write_map_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let a = Anonymizer::with_salt(true, [7u8; 16]);
        let ca = a.anon("USER_ALICE");
        let cb = a.anon("USER_BOB");
        a.write_map(dir.path()).unwrap();
        let body = std::fs::read_to_string(dir.path().join("anonymize_map.tsv")).unwrap();
        assert!(body.starts_with("# anon\toriginal"));
        assert!(body.contains(&format!("{ca}\tUSER_ALICE")));
        assert!(body.contains(&format!("{cb}\tUSER_BOB")));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join("anonymize_map.tsv"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }
}
