use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const BURN_MANIFEST_DEPS: &[&str] = &["burn", "burn-ndarray"];

const BURN_SOURCE_TOKENS: &[&str] = &[
    "burn::",
    "burn_ndarray",
    "AutodiffBackend",
    "GradientsParams",
    "TensorData",
    "Shape::new",
    "Param<",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "BurnPINN",
    "BurnWave",
    "burn_wave_equation",
];

const LEGACY_MANIFEST_DEPS: &[&str] = &[
    "nalgebra",
    "nalgebra-sparse",
    "ndarray",
    "burn",
    "burn-ndarray",
    "tokio",
    "rayon",
    "approx",
    "num-traits",
    "rustfft",
];

const LEGACY_SOURCE_TOKENS: &[&str] = &[
    "nalgebra::",
    "ndarray::",
    "burn::",
    "burn_ndarray",
    "tokio::",
    "rayon::",
    "approx::",
    "num_traits::",
    "rustfft::",
];

const ATLAS_MANIFEST_DEPS: &[&str] = &["moirai", "leto", "hephaestus", "coeus"];

const ALLOWLIST_REL_PATH: &str = "xtask/legacy_surface.allowlist";
const BURN_ALLOWLIST_REL_PATH: &str = "xtask/burn_surface.allowlist";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceReference {
    pub(crate) path: PathBuf,
    pub(crate) count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LegacyMigrationReport {
    pub(crate) manifest_dependencies: Vec<PathBuf>,
    pub(crate) source_references: Vec<SourceReference>,
    pub(crate) atlas_manifest_references: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CrateBurnSurface {
    pub(crate) manifest_dependency: bool,
    pub(crate) source_reference_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnMigrationReport {
    pub(crate) manifest_dependencies: Vec<PathBuf>,
    pub(crate) source_references: Vec<SourceReference>,
    pub(crate) by_crate: BTreeMap<String, CrateBurnSurface>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AllowlistDiff {
    new_entries: Vec<String>,
    stale_entries: Vec<String>,
}

pub(crate) fn print_legacy_migration_audit(root: &Path) -> Result<()> {
    let report = scan_legacy_migration_surface(root)?;
    let diff = compare_with_allowlist(root, &report)?;

    println!("Legacy migration audit");
    println!("======================");
    println!();
    println!(
        "Manifest files with legacy deps ({})",
        report.manifest_dependencies.len()
    );
    for path in &report.manifest_dependencies {
        println!("  - {}", path.display());
    }

    println!();
    println!(
        "Source files with legacy tokens ({})",
        report.source_references.len()
    );
    for source in &report.source_references {
        println!("  - {} ({})", source.path.display(), source.count);
    }

    println!();
    println!(
        "Manifest files already using Atlas deps ({})",
        report.atlas_manifest_references.len()
    );
    for path in &report.atlas_manifest_references {
        println!("  - {}", path.display());
    }

    println!();
    if diff.new_entries.is_empty() {
        println!("Allowlist status: clean");
    } else {
        println!(
            "Allowlist drift: {} new legacy surfaces not in {}",
            diff.new_entries.len(),
            ALLOWLIST_REL_PATH
        );
        for entry in &diff.new_entries {
            println!("  - {entry}");
        }
    }

    if !diff.stale_entries.is_empty() {
        println!();
        println!("Allowlist cleanup candidates (already migrated):");
        for entry in &diff.stale_entries {
            println!("  - {entry}");
        }
    }

    if !diff.new_entries.is_empty() {
        bail!(
            "legacy migration allowlist drift detected; run `cargo run -p xtask -- refresh-legacy-allowlist` after intentional changes"
        );
    }

    Ok(())
}

pub(crate) fn print_burn_migration_audit(root: &Path) -> Result<()> {
    let report = scan_burn_migration_surface(root)?;
    let diff = compare_burn_with_allowlist(root, &report)?;

    println!("Burn migration audit");
    println!("====================");
    println!();
    println!(
        "Manifest files with Burn deps ({})",
        report.manifest_dependencies.len()
    );
    for path in &report.manifest_dependencies {
        println!("  - {}", path.display());
    }

    println!();
    println!(
        "Source files with Burn-surface tokens ({})",
        report.source_references.len()
    );
    for source in &report.source_references {
        println!("  - {} ({})", source.path.display(), source.count);
    }

    println!();
    println!("Crate summary:");
    for (name, surface) in &report.by_crate {
        println!(
            "  - {name}: burn(dep={}, tokens={})",
            surface.manifest_dependency, surface.source_reference_count
        );
    }

    println!();
    if diff.new_entries.is_empty() {
        println!("Allowlist status: clean");
    } else {
        println!(
            "Allowlist drift: {} new Burn surfaces not in {}",
            diff.new_entries.len(),
            BURN_ALLOWLIST_REL_PATH
        );
        for entry in &diff.new_entries {
            println!("  - {entry}");
        }
    }

    if !diff.stale_entries.is_empty() {
        println!();
        println!("Allowlist cleanup candidates (already migrated):");
        for entry in &diff.stale_entries {
            println!("  - {entry}");
        }
    }

    if !diff.new_entries.is_empty() {
        bail!(
            "Burn migration allowlist drift detected; run `cargo run -p xtask -- refresh-burn-allowlist` after intentional changes"
        );
    }

    Ok(())
}

pub(crate) fn refresh_legacy_allowlist(root: &Path) -> Result<()> {
    let report = scan_legacy_migration_surface(root)?;
    let mut entries = BTreeSet::new();

    for path in &report.manifest_dependencies {
        entries.insert(manifest_allowlist_entry(path));
    }
    for source in &report.source_references {
        entries.insert(source_allowlist_entry(&source.path));
    }

    let allowlist_path = root.join(ALLOWLIST_REL_PATH);
    if let Some(parent) = allowlist_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating {}", parent.display()))?;
    }

    let mut body = String::from(
        "# Auto-generated by `cargo run -p xtask -- refresh-legacy-allowlist`\n\
         # Each entry tracks an approved legacy surface during Atlas migration.\n",
    );
    for entry in entries {
        body.push_str(&entry);
        body.push('\n');
    }

    fs::write(&allowlist_path, body)
        .with_context(|| format!("failed writing {}", allowlist_path.display()))?;
    println!("Updated {}", allowlist_path.display());
    Ok(())
}

pub(crate) fn refresh_burn_allowlist(root: &Path) -> Result<()> {
    let report = scan_burn_migration_surface(root)?;
    let mut entries = BTreeSet::new();

    for path in &report.manifest_dependencies {
        entries.insert(manifest_allowlist_entry(path));
    }
    for source in &report.source_references {
        entries.insert(source_allowlist_entry(&source.path));
    }

    let allowlist_path = root.join(BURN_ALLOWLIST_REL_PATH);
    if let Some(parent) = allowlist_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating {}", parent.display()))?;
    }

    let mut body = String::from(
        "# Auto-generated by `cargo run -p xtask -- refresh-burn-allowlist`\n\
         # Each entry tracks an approved Burn surface during Coeus migration.\n",
    );
    for entry in entries {
        body.push_str(&entry);
        body.push('\n');
    }

    fs::write(&allowlist_path, body)
        .with_context(|| format!("failed writing {}", allowlist_path.display()))?;
    println!("Updated {}", allowlist_path.display());
    Ok(())
}

pub(crate) fn scan_legacy_migration_surface(root: &Path) -> Result<LegacyMigrationReport> {
    let mut manifest_dependencies = Vec::new();
    let mut source_references = Vec::new();
    let mut atlas_manifest_references = Vec::new();

    visit_files(root, &mut |path| {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            return Ok(());
        };

        if file_name == "Cargo.toml" {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            let rel = relative(root, path);
            if has_any_manifest_dep(&text, LEGACY_MANIFEST_DEPS) {
                manifest_dependencies.push(rel.clone());
            }
            if has_any_manifest_dep(&text, ATLAS_MANIFEST_DEPS) {
                atlas_manifest_references.push(rel);
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            let count = LEGACY_SOURCE_TOKENS
                .iter()
                .map(|token| text.matches(token).count())
                .sum::<usize>();
            if count > 0 {
                source_references.push(SourceReference {
                    path: relative(root, path),
                    count,
                });
            }
        }

        Ok(())
    })?;

    manifest_dependencies.sort();
    atlas_manifest_references.sort();
    source_references.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(LegacyMigrationReport {
        manifest_dependencies,
        source_references,
        atlas_manifest_references,
    })
}

pub(crate) fn scan_burn_migration_surface(root: &Path) -> Result<BurnMigrationReport> {
    let mut manifest_dependencies = Vec::new();
    let mut source_references = Vec::new();
    let mut by_crate = BTreeMap::<String, CrateBurnSurface>::new();

    visit_files(root, &mut |path| {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            return Ok(());
        };

        if file_name == "Cargo.toml" {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            if has_any_manifest_dep(&text, BURN_MANIFEST_DEPS) {
                let rel = relative(root, path);
                if let Some(crate_name) = crate_name_from_manifest(root, path) {
                    by_crate.entry(crate_name).or_default().manifest_dependency = true;
                }
                manifest_dependencies.push(rel);
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed reading {}", path.display()))?;
            let count = BURN_SOURCE_TOKENS
                .iter()
                .map(|token| text.matches(token).count())
                .sum::<usize>();
            if count > 0 {
                if let Some(crate_name) = crate_name_from_source(root, path) {
                    by_crate
                        .entry(crate_name)
                        .or_default()
                        .source_reference_count += count;
                }
                source_references.push(SourceReference {
                    path: relative(root, path),
                    count,
                });
            }
        }

        Ok(())
    })?;

    manifest_dependencies.sort();
    source_references.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(BurnMigrationReport {
        manifest_dependencies,
        source_references,
        by_crate,
    })
}

fn compare_with_allowlist(root: &Path, report: &LegacyMigrationReport) -> Result<AllowlistDiff> {
    let allowlist_path = root.join(ALLOWLIST_REL_PATH);
    if !allowlist_path.exists() {
        bail!(
            "missing {}; run `cargo run -p xtask -- refresh-legacy-allowlist`",
            allowlist_path.display()
        );
    }

    let allowed = load_allowlist(&allowlist_path)?;
    let mut current = BTreeSet::new();

    for path in &report.manifest_dependencies {
        current.insert(manifest_allowlist_entry(path));
    }
    for source in &report.source_references {
        current.insert(source_allowlist_entry(&source.path));
    }

    let new_entries = current
        .difference(&allowed)
        .cloned()
        .collect::<Vec<String>>();
    let stale_entries = allowed
        .difference(&current)
        .cloned()
        .collect::<Vec<String>>();

    Ok(AllowlistDiff {
        new_entries,
        stale_entries,
    })
}

fn compare_burn_with_allowlist(root: &Path, report: &BurnMigrationReport) -> Result<AllowlistDiff> {
    let allowlist_path = root.join(BURN_ALLOWLIST_REL_PATH);
    if !allowlist_path.exists() {
        bail!(
            "missing {}; run `cargo run -p xtask -- refresh-burn-allowlist`",
            allowlist_path.display()
        );
    }

    let allowed = load_allowlist(&allowlist_path)?;
    let mut current = BTreeSet::new();

    for path in &report.manifest_dependencies {
        current.insert(manifest_allowlist_entry(path));
    }
    for source in &report.source_references {
        current.insert(source_allowlist_entry(&source.path));
    }

    let new_entries = current
        .difference(&allowed)
        .cloned()
        .collect::<Vec<String>>();
    let stale_entries = allowed
        .difference(&current)
        .cloned()
        .collect::<Vec<String>>();

    Ok(AllowlistDiff {
        new_entries,
        stale_entries,
    })
}

fn load_allowlist(path: &Path) -> Result<BTreeSet<String>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed reading {}", path.display()))?;
    let mut entries = BTreeSet::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        entries.insert(trimmed.to_owned());
    }
    Ok(entries)
}

fn has_any_manifest_dep(text: &str, deps: &[&str]) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('[') {
            return false;
        }

        let Some((raw_key, _)) = trimmed.split_once('=') else {
            return false;
        };

        let key = raw_key.trim();
        deps.iter()
            .any(|dep| key == *dep || key == format!("{dep}.workspace"))
    })
}

fn visit_files(root: &Path, visit: &mut impl FnMut(&Path) -> Result<()>) -> Result<()> {
    if should_skip(root) {
        return Ok(());
    }

    for entry in fs::read_dir(root).with_context(|| format!("failed reading {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if should_skip(&path) {
            continue;
        }
        if path.is_dir() {
            visit_files(&path, visit)?;
        } else {
            visit(&path)?;
        }
    }

    Ok(())
}

fn should_skip(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".git" | "target" | "xtask")
    )
}

fn relative(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

fn crate_name_from_manifest(root: &Path, manifest: &Path) -> Option<String> {
    let rel = manifest.strip_prefix(root).ok()?;
    let mut parts = rel.components();
    match parts.next()?.as_os_str().to_str()? {
        "crates" => parts.next()?.as_os_str().to_str().map(str::to_owned),
        "xtask" => Some("xtask".to_owned()),
        _ => None,
    }
}

fn crate_name_from_source(root: &Path, source: &Path) -> Option<String> {
    let rel = source.strip_prefix(root).ok()?;
    let mut parts = rel.components();
    match parts.next()?.as_os_str().to_str()? {
        "crates" => parts.next()?.as_os_str().to_str().map(str::to_owned),
        "xtask" => Some("xtask".to_owned()),
        _ => None,
    }
}

fn manifest_allowlist_entry(path: &Path) -> String {
    format!("manifest:{}", normalized(path))
}

fn source_allowlist_entry(path: &Path) -> String {
    format!("source:{}", normalized(path))
}

fn normalized(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn burn_audit_detects_manifest_and_source_surface() {
        let root = temp_root();
        fs::create_dir_all(root.join("crates/kwavers-solver/src/inverse/pinn/ml")).unwrap();
        fs::write(
            root.join("crates/kwavers-solver/Cargo.toml"),
            "[dependencies]\nburn = { workspace = true }\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/kwavers-solver/src/inverse/pinn/ml/lib.rs"),
            "use burn::module::Module;\n\
             use burn_ndarray::NdArray;\n\
             type Model<B> = BurnPINN1DWave<B>;\n",
        )
        .unwrap();

        let report = scan_burn_migration_surface(&root).unwrap();

        assert_eq!(
            report.manifest_dependencies,
            vec![PathBuf::from("crates/kwavers-solver/Cargo.toml")]
        );
        assert_eq!(report.source_references.len(), 1);
        assert_eq!(
            report.by_crate.get("kwavers-solver"),
            Some(&CrateBurnSurface {
                manifest_dependency: true,
                source_reference_count: 3,
            })
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn burn_audit_does_not_flag_coeus_tensor_surface() {
        let root = temp_root();
        fs::create_dir_all(root.join("crates/kwavers-solver/src/inverse/pinn/ml")).unwrap();
        fs::write(
            root.join("crates/kwavers-solver/src/inverse/pinn/ml/lib.rs"),
            "use coeus_tensor::Tensor;\n\
             type Model<B> = Tensor<f32, B>;\n",
        )
        .unwrap();

        let report = scan_burn_migration_surface(&root).unwrap();

        assert!(report.manifest_dependencies.is_empty());
        assert!(report.source_references.is_empty());
        assert!(report.by_crate.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    fn temp_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kwavers-migration-audit-{nanos}"))
    }
}
