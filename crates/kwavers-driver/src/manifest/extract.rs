//! [`hv_manifest_from_board`] — build the HV-board portion of a [`DriverManifest`] from a
//! placed/routed design: discover the `TX_N` nets on the transducer connector, order them by
//! channel, derive the article-scaled aperture, and stamp the article operating-point constants.

use std::path::Path;

use crate::board::Board;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::ssot::{MANIFEST_ARTICLE_APERTURE_M, MANIFEST_ARTICLE_ELEMENTS};

use super::driver_manifest::DriverManifest;
use super::stimulation::StimulationProgram;

/// Build the HV board portion of the manifest from the placed/routed design.
pub fn hv_manifest_from_board(
    board_path: &Path,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    tx_connector: &str,
) -> Result<DriverManifest, String> {
    let c = comps
        .iter()
        .find(|c| c.refdes == tx_connector)
        .ok_or_else(|| format!("missing transducer connector {tx_connector}"))?;
    let mut tx_nets: Vec<(usize, String)> = c
        .nets
        .iter()
        .filter_map(|n| {
            let n = (*n)?;
            let name = &board.nets[n.0 as usize].name;
            let idx = name.strip_prefix("TX_")?.parse::<usize>().ok()?;
            Some((idx, name.clone()))
        })
        .collect();
    tx_nets.sort_by_key(|(idx, _)| *idx);
    let names: Vec<String> = tx_nets.into_iter().map(|(_, n)| n).collect();
    if names.is_empty() {
        return Err(format!(
            "{tx_connector} must expose at least one TX_N net; none found on its pads"
        ));
    }
    if lib[c.fp].pads.len() < names.len() {
        return Err(format!(
            "{tx_connector} footprint has fewer pads than TX nets"
        ));
    }
    let aperture_m = if names.len() <= 1 {
        0.0
    } else {
        MANIFEST_ARTICLE_APERTURE_M * (names.len() - 1) as f64
            / (MANIFEST_ARTICLE_ELEMENTS - 1) as f64
    };
    Ok(DriverManifest {
        hv_board: board_path
            .file_name()
            .and_then(|s| s.to_str())
            // Fallback applies only when the path has no file component (e.g., a bare directory
            // path passed during testing). In production the path is always a .kicad_pcb file.
            .unwrap_or("hv_board.kicad_pcb")
            .to_string(),
        tx_connector: tx_connector.to_string(),
        tx_nets: names,
        // "pending-controller-manifest" is a placeholder: the FPGA tile example replaces this
        // field with the actual controller-manifest path after cooptimize completes on the FPGA
        // board. Until that path is known, manifests are generated with this sentinel value so
        // consumers can detect an incomplete manifest rather than silently using a stale one.
        programming: "pending-controller-manifest".into(),
        aperture_m,
        // Article operating-point constants (IEEE TBME 2024, focal ultrasound neurostimulation):
        // 2 MHz centre frequency, 1540 m/s tissue speed, 10 mm focal depth, 5 ns timing step.
        // These are fixed by the article's 4.3 mm aperture / 16-element array geometry; a
        // production driver with a different transducer must override them in the manifest.
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        // Article-class stimulation program: 1 kHz PRF × 0.5 ms bursts × 300 ms sonication ×
        // 3 s ISI × 18 s total at 150 V. Bumped-to-v2 means every new emission carries the
        // full program; downstream acoustic/lifecycle accounting reads it directly.
        stimulation: Some(StimulationProgram::article_default()),
        // hv_manifest_from_board builds the HV-board portion only; per-tile stimulation
        // profiles are populated later once all tiles are cooptimized and assembled.
        tile_profiles: Vec::new(),
    })
}
