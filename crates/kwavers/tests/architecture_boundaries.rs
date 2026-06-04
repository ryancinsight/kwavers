//! Architectural-layering guard.
//!
//! After the `kwavers-domain` decomposition (2026-06) the codebase is a set of
//! single-responsibility crates forming a directed acyclic dependency graph.
//! Cargo already enforces acyclicity, but it does *not* prevent a low-layer
//! crate from *gaining* an upward dependency (a developer could add the
//! manifest line and the `use`). These tests pin the intended direction:
//! foundation/model/device-tier crates must never reference the orchestration
//! tiers (`kwavers_solver`, `kwavers_simulation`, `kwavers_analysis`).

use std::fs;
use std::path::{Path, PathBuf};

fn rust_files_under(root: &Path) -> Vec<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    files.sort();
    files
}

/// Workspace `crates/` directory, derived from this crate's manifest dir.
fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/kwavers has a parent (crates/)")
        .to_path_buf()
}

/// Domain-tier crates that sit below the orchestration layers and must not
/// reference solver/simulation/analysis.
const DOMAIN_TIER_CRATES: &[&str] = &[
    "kwavers-grid",
    "kwavers-field",
    "kwavers-signal",
    "kwavers-optics",
    "kwavers-mesh",
    "kwavers-medium",
    "kwavers-phantom",
    "kwavers-boundary",
    "kwavers-source",
    "kwavers-receiver",
    "kwavers-transducer",
    "kwavers-imaging",
];

const FORBIDDEN_UPWARD: &[&str] = &[
    "kwavers_solver::",
    "kwavers_simulation::",
    "kwavers_analysis::",
];

#[test]
fn domain_tier_crates_have_no_upward_imports() {
    let crates = crates_dir();
    let mut violations = Vec::new();

    for crate_name in DOMAIN_TIER_CRATES {
        let src = crates.join(crate_name).join("src");
        assert!(
            src.is_dir(),
            "expected domain-tier crate source at {}",
            src.display()
        );
        for file in rust_files_under(&src) {
            let source = fs::read_to_string(&file).unwrap();
            for forbidden in FORBIDDEN_UPWARD {
                if source.contains(forbidden) {
                    violations.push(format!("{crate_name}: {} contains {forbidden}", file.display()));
                }
            }
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

/// `kwavers-domain` was fully decomposed and removed; guard against accidental
/// resurrection (which would re-introduce the mega-crate the split eliminated).
#[test]
fn kwavers_domain_crate_is_gone() {
    let domain = crates_dir().join("kwavers-domain");
    assert!(
        !domain.exists(),
        "kwavers-domain reappeared at {}; it was decomposed into domain-specific crates",
        domain.display()
    );
}
