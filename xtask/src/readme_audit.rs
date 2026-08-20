//! Per-crate README audit.
//!
//! Every crate in the workspace is a registry landing page: crates.io and docs.rs show
//! its README and its manifest description, so a crate without one publishes blank. This
//! audit enforces three properties over `crates/*`:
//!
//! 1. the crate has a `README.md`,
//! 2. the manifest carries a `description`, and
//! 3. `src/lib.rs` single-sources that README as its crate documentation via
//!    `#![doc = include_str!("../README.md")]`, so the registry page and the docs.rs
//!    front page cannot drift apart.
//!
//! Property 3 is compiler-enforced once wired; the audit exists to catch a *new* crate
//! that never wires it. `SINGLE_SOURCE_EXEMPT` is a non-increasing ratchet, not a
//! license: entries name the backlog item that removes them.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Crates whose README is not yet the crate documentation.
///
/// Both carry substantial `//!` documentation that differs from their README, so folding
/// the two is a content merge rather than a mechanical wiring step. Tracked by
/// KW-DOC-105; this list only shrinks.
const SINGLE_SOURCE_EXEMPT: &[&str] = &["kwavers-driver", "kwavers-python"];

const INCLUDE_DIRECTIVE: &str = r#"#![doc = include_str!("../README.md")]"#;

/// One crate's audit result.
struct CrateAudit {
    name: String,
    missing_readme: bool,
    missing_description: bool,
    unwired: bool,
}

impl CrateAudit {
    fn is_clean(&self) -> bool {
        !self.missing_readme && !self.missing_description && !self.unwired
    }
}

/// Audit every crate under `<workspace_root>/crates`.
///
/// # Errors
/// Returns an error if the crates directory or a crate's manifest cannot be read, or if
/// any crate fails the audit.
pub fn check_readmes(workspace_root: &Path) -> Result<()> {
    println!("🔍 Auditing per-crate READMEs (registry landing pages)...");

    let crates_dir = workspace_root.join("crates");
    let mut audits = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(&crates_dir)
        .with_context(|| format!("Failed to read {}", crates_dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("Cargo.toml").is_file())
        .collect();
    entries.sort();

    for crate_dir in entries {
        let name = crate_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();

        let manifest = fs::read_to_string(crate_dir.join("Cargo.toml"))
            .with_context(|| format!("Failed to read manifest for {name}"))?;

        let lib_rs = crate_dir.join("src").join("lib.rs");
        let lib_source = fs::read_to_string(&lib_rs).unwrap_or_default();

        let exempt = SINGLE_SOURCE_EXEMPT.contains(&name.as_str());

        audits.push(CrateAudit {
            missing_readme: !crate_dir.join("README.md").is_file(),
            missing_description: !manifest
                .lines()
                .any(|line| line.trim_start().starts_with("description")),
            unwired: !exempt && lib_rs.is_file() && !lib_source.contains(INCLUDE_DIRECTIVE),
            name,
        });
    }

    let failures: Vec<&CrateAudit> = audits.iter().filter(|a| !a.is_clean()).collect();

    if failures.is_empty() {
        println!(
            "✅ {} crates: README, description, and single-sourced crate docs present",
            audits.len()
        );
        return Ok(());
    }

    println!("❌ {} crate(s) fail the README audit:", failures.len());
    for audit in &failures {
        if audit.missing_readme {
            println!(
                "  {}: no README.md — the crates.io page would be blank",
                audit.name
            );
        }
        if audit.missing_description {
            println!("  {}: no `description` in Cargo.toml", audit.name);
        }
        if audit.unwired {
            println!(
                "  {}: src/lib.rs does not carry `{INCLUDE_DIRECTIVE}` — README and crate docs can drift",
                audit.name
            );
        }
    }

    anyhow::bail!("{} crate(s) fail the README audit", failures.len())
}
