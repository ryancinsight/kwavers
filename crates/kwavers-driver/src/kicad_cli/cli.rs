//! The [`KiCadCli`] process wrapper: locate a usable `kicad-cli`, spawn it, and drive the
//! `pcb drc` / `pcb render` / Gerber-drill-pos-BOM export subcommands.
//!
//! Auto-detection: [`KiCadCli::locate`] accepts any of
//! * the user's `PATH` (so a system install works);
//! * common Windows install paths (`%LOCALAPPDATA%\Programs\KiCad\<ver>\bin\kicad-cli.exe`,
//!   `C:\Program Files\KiCad\<ver>\bin\kicad-cli.exe`);
//! * macOS / Linux bins.
//!
//! A missing binary surfaces a single clear error (the example then fails loudly) rather than a
//! silent green build.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::drc::{parse_drc_json, DrcReport};
use super::fab::FabBundle;

/// Candidate paths to probe for a `kicad-cli` install, in order. The first hit wins; the wrapped
/// `PATH` search runs first so a system install is preferred over a hard-coded local path.
const DEFAULT_PROBE_PATHS: &[&str] = &[
    // Windows (KiCad 7+ install via the official installer).
    r"C:\Users\RyanClanton\AppData\Local\Programs\KiCad\10.0\bin\kicad-cli.exe",
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
    r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe",
    r"C:\Program Files\KiCad\8.0\bin\kicad-cli.exe",
    // Linux.
    "/usr/bin/kicad-cli",
    "/usr/local/bin/kicad-cli",
    // macOS app bundle.
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
];

/// Locate (and lazily instantiate) a usable `kicad-cli`.
#[derive(Debug, Clone)]
pub struct KiCadCli {
    /// Path to the executable that will be spawned.
    pub path: PathBuf,
}

/// Options for KiCad PCB design-rule checking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrcOptions {
    /// Refill copper zones before DRC. Required when the board was modified programmatically.
    pub refill_zones: bool,
    /// Persist the refilled board. KiCad only accepts this together with `--refill-zones`.
    pub save_board: bool,
}

impl KiCadCli {
    /// Locate the binary; return an error string naming the searched paths when none is found.
    pub fn locate() -> Result<Self, String> {
        if let Ok(p) = locate_on_path("kicad-cli") {
            return Ok(Self { path: p });
        }
        for c in DEFAULT_PROBE_PATHS {
            let p = PathBuf::from(c);
            if p.exists() {
                return Ok(Self { path: p });
            }
        }
        Err(format!(
            "kicad-cli not found. Searched PATH plus: {DEFAULT_PROBE_PATHS:?}. \
             Install KiCad 7+ or place kicad-cli on PATH."
        ))
    }

    /// Construct from an explicit executable path (used by tests + tools that discover elsewhere).
    #[must_use]
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Run kicad-cli with `args`, returning combined stdout on success.
    fn run(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new(&self.path)
            .args(args)
            .output()
            .map_err(|e| format!("failed to spawn {}: {e}", self.path.display()))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            return Err(format!(
                "kicad-cli {args:?} exited with status {}\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}",
                output.status
            ));
        }
        Ok(stdout)
    }

    /// Run **design-rule check** with JSON output to a temp file. Parses the JSON for violation,
    /// unconnected_items and warning counts; the full text is retained in [`DrcReport::raw_json`]
    /// so callers can produce pretty diagnostics for any non-zero count.
    pub fn drc(&self, pcb: &Path) -> Result<DrcReport, String> {
        let tmp = std::env::temp_dir().join("kicad-routing-drc.json");
        self.drc_to(pcb, &tmp)
    }

    /// Run design-rule check and persist the KiCad JSON report at `out_json`.
    pub fn drc_to(&self, pcb: &Path, out_json: &Path) -> Result<DrcReport, String> {
        self.drc_to_with_options(pcb, out_json, DrcOptions::default())
    }

    /// Run design-rule check with explicit KiCad zone-refill behavior.
    ///
    /// Programmatic routing changes can leave copper-zone fill geometry stale. Use
    /// `DrcOptions { refill_zones: true, save_board: true }` for candidate boards whose vias,
    /// tracks, or drilled pads changed; KiCad requires `--save-board` to be paired with
    /// `--refill-zones`.
    pub fn drc_to_with_options(
        &self,
        pcb: &Path,
        out_json: &Path,
        options: DrcOptions,
    ) -> Result<DrcReport, String> {
        if std::fs::remove_file(out_json).is_err() {
            // File may not exist yet on the first run; that is the expected case.
        }
        let pcb_str = pcb
            .to_str()
            .ok_or_else(|| format!("non-utf8 pcb path: {}", pcb.display()))?;
        let out_str = out_json
            .to_str()
            .ok_or_else(|| format!("non-utf8 DRC output path: {}", out_json.display()))?;
        let args = drc_args(pcb_str, out_str, options)?;
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        self.run(&argv)?;
        let text = std::fs::read_to_string(out_json)
            .map_err(|e| format!("failed to read DRC report {}: {e}", out_json.display()))?;
        Ok(parse_drc_json(&text))
    }

    /// Render the board as a PNG (top + side views by default). The renderer path string is logged
    /// so the caller can decide which one to ship to the fab bundle.
    pub fn render(&self, pcb: &Path, out_png: &Path) -> Result<(), String> {
        let pcb_str = pcb
            .to_str()
            .ok_or_else(|| format!("non-utf8 pcb path: {}", pcb.display()))?;
        let out_str = out_png
            .to_str()
            .ok_or_else(|| format!("non-utf8 png path: {}", out_png.display()))?;
        self.run(&["pcb", "render", "--output", out_str, pcb_str])
            .map(|_| ())
    }

    /// Export the **fabrication bundle** in one call: Gerbers, drill files, position CSV, and BOM
    /// (schematic side) — each into its own subdirectory so a fab house can pick them straight up.
    pub fn export_fab(
        &self,
        pcb: &Path,
        sch: Option<&Path>,
        out_dir: &Path,
    ) -> Result<FabBundle, String> {
        std::fs::create_dir_all(out_dir)
            .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;
        let pcb_str = pcb
            .to_str()
            .ok_or_else(|| format!("non-utf8 pcb path: {}", pcb.display()))?;

        let gerbers = out_dir.join("gerbers");
        std::fs::create_dir_all(&gerbers)
            .map_err(|e| format!("failed to create {}: {e}", gerbers.display()))?;
        let g_out = gerbers
            .to_str()
            .ok_or_else(|| format!("non-utf8 gerber path: {}", gerbers.display()))?;
        self.run(&["pcb", "export", "gerbers", "--output", g_out, pcb_str])?;

        let drill = out_dir.join("drill");
        std::fs::create_dir_all(&drill)
            .map_err(|e| format!("failed to create {}: {e}", drill.display()))?;
        let d_out = drill
            .to_str()
            .ok_or_else(|| format!("non-utf8 drill path: {}", drill.display()))?;
        self.run(&["pcb", "export", "drill", "--output", d_out, pcb_str])?;

        let pos_path = out_dir.join("position.csv");
        let p_out = pos_path
            .to_str()
            .ok_or_else(|| format!("non-utf8 pos path: {}", pos_path.display()))?;
        self.run(&[
            "pcb", "export", "pos", "--format", "csv", "--output", p_out, pcb_str,
        ])?;

        // 3D render — for the bundle's "art of the build" cover sheet.
        let render_path = out_dir.join("render.png");
        if let Err(_e) = self.render(pcb, &render_path) {
            // render failure is non-fatal: fab bundle is still valid without the 3D render
        }

        // BOM needs the schematic.
        let bom_path = out_dir.join("bom.csv");
        let bom_str = bom_path
            .to_str()
            .ok_or_else(|| format!("non-utf8 bom path: {}", bom_path.display()))?;
        let mut bom_exported = false;
        if let Some(sch) = sch {
            let sch_str = sch
                .to_str()
                .ok_or_else(|| format!("non-utf8 sch path: {}", sch.display()))?;
            if self
                .run(&["sch", "export", "bom", "--output", bom_str, sch_str])
                .is_ok()
            {
                bom_exported = true;
            }
        }

        Ok(FabBundle {
            gerber_dir: gerbers,
            drill_dir: drill,
            position_csv: pos_path,
            render_png: Some(render_path),
            bom_csv: if bom_exported { Some(bom_path) } else { None },
        })
    }

    /// Print the kicad-cli version (useful diagnostic when a vendor install misbehaves).
    pub fn version(&self) -> Result<String, String> {
        self.run(&["version"])
    }
}

/// Assemble the `pcb drc` argument vector, rejecting the KiCad constraint that `--save-board`
/// requires `--refill-zones`.
pub(super) fn drc_args(
    pcb: &str,
    out_json: &str,
    options: DrcOptions,
) -> Result<Vec<String>, String> {
    if options.save_board && !options.refill_zones {
        return Err("KiCad DRC --save-board requires --refill-zones".to_string());
    }

    let mut args = vec!["pcb".to_string(), "drc".to_string()];
    if options.refill_zones {
        args.push("--refill-zones".to_string());
    }
    if options.save_board {
        args.push("--save-board".to_string());
    }
    args.extend([
        "--format".to_string(),
        "json".to_string(),
        "--output".to_string(),
        out_json.to_string(),
        pcb.to_string(),
    ]);
    Ok(args)
}

/// Cross-platform locating of `cmd` on `PATH`; on Windows we also probe `cmd.exe`.
pub(super) fn locate_on_path(cmd: &str) -> Result<PathBuf, ()> {
    let path = std::env::var_os("PATH").ok_or(())?;
    for dir in std::env::split_paths(&path) {
        let direct = dir.join(cmd);
        if direct.exists() {
            return Ok(direct);
        }
        if cfg!(windows) {
            let ext = dir.join(format!("{cmd}.exe"));
            if ext.exists() {
                return Ok(ext);
            }
        }
    }
    Err(())
}
