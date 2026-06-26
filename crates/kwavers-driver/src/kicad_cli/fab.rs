//! The [`FabBundle`] — the set of fabrication artifacts ([`super::KiCadCli::export_fab`] produces
//! Gerbers, drill files, a pick-and-place CSV, an optional 3D render, and an optional BOM) plus its
//! one-line-per-artifact summary for the example log.

use std::path::{Path, PathBuf};

/// Bundle of artifacts produced by [`super::KiCadCli::export_fab`].
#[derive(Debug, Clone, Default)]
pub struct FabBundle {
    /// Directory holding the layer Gerbers (one .gbr per copper + Edge.Cuts).
    pub gerber_dir: PathBuf,
    /// Directory holding drill files (.drl, drill map).
    pub drill_dir: PathBuf,
    /// Pick-and-place position CSV.
    pub position_csv: PathBuf,
    /// Optional 3D render PNG (skip if bundled 3D models are absent).
    pub render_png: Option<PathBuf>,
    /// Optional BOM CSV (only emitted when a schematic was supplied).
    pub bom_csv: Option<PathBuf>,
}

impl FabBundle {
    /// Emitted-bundle summary (one line per artifact) for the example log.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "fab: gerbers={} ({} files)",
            self.gerber_dir.display(),
            count_dir_files(&self.gerber_dir, &["gbr", "gtl", "gbl", "gbs", "gts"])
        ));
        out.push(format!("fab: drill={}", self.drill_dir.display()));
        if let Some(p) = &self.render_png {
            out.push(format!("fab: render={}", p.display()));
        }
        out.push(format!("fab: position={}", self.position_csv.display()));
        if let Some(p) = &self.bom_csv {
            out.push(format!("fab: bom={}", p.display()));
        }
        out
    }
}

fn count_dir_files(dir: &Path, exts: &[&str]) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    rd.filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| {
                    exts.iter()
                        .any(|want| want.trim_start_matches('.').eq_ignore_ascii_case(ext))
                })
        })
        .count()
}
