//! [`DriverManifest`] itself: the board-backed beamforming schema, its deterministic key-value text
//! round-trip (`to_text`/`from_text`/`read`, v1 + v2 / single-stim + per-tile forms), and the
//! protocol-load accessors. The energy-budget validator lives in [`super::energy_budget`].

use std::fmt::Write as _;
use std::path::Path;

use crate::ssot::{MANIFEST_FORMAT_V1, MANIFEST_FORMAT_V2, TX_LANES_V2};

use super::stimulation::{StimulationProgram, TileStimulationProfile};

/// Board-backed beamforming configuration for one generated driver artifact set.
#[derive(Debug, Clone, PartialEq)]
pub struct DriverManifest {
    /// Source HV board filename.
    pub hv_board: String,
    /// Transducer output connector reference designator.
    pub tx_connector: String,
    /// Routed transducer output net names, ordered by channel.
    pub tx_nets: Vec<String>,
    /// FPGA/programming evidence string from the controller artifact.
    pub programming: String,
    /// Array aperture, first-to-last element centre span (m).
    pub aperture_m: f64,
    /// Drive frequency (Hz).
    pub frequency_hz: f64,
    /// Medium sound speed (m/s).
    pub sound_speed_m_s: f64,
    /// Nominal focal depth (m).
    pub focal_m: f64,
    /// Hardware timing quantum (s).
    pub timing_step_s: f64,
    /// Article-class stimulation program (PRF/TBD/SD/ISI/TT/vpp/dead-time). `None` in v1
    /// parses for backwards compatibility; new emissions always populate it.
    pub stimulation: Option<StimulationProgram>,
    /// Track D v2 follow-up: per-tile stimulation programme (each of the 4 HV tiles carries
    /// its own PRF/SHIFT/PHASE/RAMP override plus the inherited protocol fields). v2 tile-form
    /// emissions populate this with one entry per HV tile in slot order; v1 and pre-tile v2
    /// emissions leave it empty so the article-class single `stimulation` block is the source
    /// of truth for legacy consumers.
    pub tile_profiles: Vec<TileStimulationProfile>,
}

impl DriverManifest {
    /// Number of transducer channels proven by the manifest.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.tx_nets.len()
    }

    /// Serialize to deterministic key-value text. v2 schema: stimulation keys written as
    /// `stim_*` (single article-class preset, retained for legacy consumers) **or** as
    /// `stim_tile_{i}_*` (Track D v2 follow-up per-tile profile with PRF/SHIFT/PHASE/RAMP
    /// plus the inherited protocol fields); v1 emits no stimulation block. The choice is
    /// driven by `tile_profiles`: any non-empty list emits the tile-form; an empty list emits
    /// the legacy single-stim form when `stimulation` is present.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "format={MANIFEST_FORMAT_V2}");
        let _ = writeln!(s, "hv_board={}", self.hv_board);
        let _ = writeln!(s, "tx_connector={}", self.tx_connector);
        let _ = writeln!(s, "tx_nets={}", self.tx_nets.join(","));
        let _ = writeln!(s, "programming={}", self.programming);
        let _ = writeln!(s, "aperture_m={:.12e}", self.aperture_m);
        let _ = writeln!(s, "frequency_hz={:.12e}", self.frequency_hz);
        let _ = writeln!(s, "sound_speed_m_s={:.12e}", self.sound_speed_m_s);
        let _ = writeln!(s, "focal_m={:.12e}", self.focal_m);
        let _ = writeln!(s, "timing_step_s={:.12e}", self.timing_step_s);
        if !self.tile_profiles.is_empty() {
            for (i, p) in self.tile_profiles.iter().enumerate() {
                let _ = writeln!(s, "stim_tile_{i}_prf_hz={:.12e}", p.prf_hz);
                let _ = writeln!(s, "stim_tile_{i}_shift_s={:.12e}", p.shift_s);
                let _ = writeln!(s, "stim_tile_{i}_phase_deg={:.12e}", p.phase_deg);
                let _ = writeln!(s, "stim_tile_{i}_ramp_s={:.12e}", p.ramp_s);
                let _ = writeln!(s, "stim_tile_{i}_tbd_s={:.12e}", p.tbd_s);
                let _ = writeln!(s, "stim_tile_{i}_sd_s={:.12e}", p.sd_s);
                let _ = writeln!(s, "stim_tile_{i}_isi_s={:.12e}", p.isi_s);
                let _ = writeln!(s, "stim_tile_{i}_tt_s={:.12e}", p.tt_s);
                let _ = writeln!(s, "stim_tile_{i}_vpp_v={:.12e}", p.vpp_v);
                let _ = writeln!(s, "stim_tile_{i}_dead_time_s={:.12e}", p.dead_time_s);
            }
        } else if let Some(stim) = &self.stimulation {
            let _ = writeln!(s, "stim_prf_hz={:.12e}", stim.prf_hz);
            let _ = writeln!(s, "stim_tbd_s={:.12e}", stim.tbd_s);
            let _ = writeln!(s, "stim_sd_s={:.12e}", stim.sd_s);
            let _ = writeln!(s, "stim_isi_s={:.12e}", stim.isi_s);
            let _ = writeln!(s, "stim_tt_s={:.12e}", stim.tt_s);
            let _ = writeln!(s, "stim_vpp_v={:.12e}", stim.vpp_v);
            let _ = writeln!(s, "stim_dead_time_s={:.12e}", stim.dead_time_s);
        }
        s
    }

    /// Parse deterministic key-value text. Accepts v1 (no stimulation block) and v2
    /// (with `stim_*` keys) for backwards compatibility.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut map = std::collections::BTreeMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Some((k, v)) = line.split_once('=') else {
                return Err(format!("manifest line has no '=': {line}"));
            };
            map.insert(k.trim(), v.trim());
        }
        let format = map.get("format").copied().unwrap_or("");
        if format != MANIFEST_FORMAT_V1 && format != MANIFEST_FORMAT_V2 {
            return Err(format!(
                "manifest format mismatch: expected {MANIFEST_FORMAT_V1} or {MANIFEST_FORMAT_V2}, got {format:?}"
            ));
        }
        let get = |k: &str| {
            map.get(k)
                .copied()
                .ok_or_else(|| format!("manifest missing {k}"))
        };
        let parse = |k: &str| -> Result<f64, String> {
            get(k)?
                .parse::<f64>()
                .map_err(|e| format!("manifest {k} parse failed: {e}"))
        };
        let tx_nets: Vec<String> = get("tx_nets")?
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if tx_nets.is_empty() {
            return Err("manifest has no TX nets".into());
        }
        // v1 ⇒ no stimulation block; v2 ⇒ either tile-form (one entry per
        // `stim_tile_{i}_*` cluster) **or** single-stim form (`stim_*` keys).
        //
        // Two parser guards enforce a clean schema on the input file:
        // 1. **Mixed-keyset guard**: when any `stim_tile_*` cluster is present, the legacy
        //    `stim_*` block must NOT also be present (would be silently shadowed by tile-form).
        //    Reject with an actionable error so the caller can pick one schema.
        // 2. **Partial-tile-form guard**: the tile loop probes `stim_tile_{tile_i}_prf_hz` and
        //    breaks on missing indices, which would silently parse a gappy/truncated file into
        //    a 1-tile manifest. After the loop, scan map keys for any `stim_tile_{i}_` with
        //    `i > last_parsed_index` and reject if found.
        let any_tile_key = map.keys().any(|k| k.starts_with("stim_tile_"));
        // Legacy `stim_*` keys are *any* key starting with `stim_` that is not a tile-form
        // `stim_tile_*` cluster. Future contributors adding new single-stim keys (`stim_carrier_hz`,
        // `stim_pre_emphasis_us`, etc.) automatically fall into this branch without needing
        // an explicit enumerated list — keeps the guard maintainable.
        let any_legacy_key = map
            .keys()
            .any(|k| k.starts_with("stim_") && !k.starts_with("stim_tile_"));
        if any_tile_key && any_legacy_key && format == MANIFEST_FORMAT_V2 {
            return Err(
                "mixed schemas: present both `stim_tile_*` clusters and legacy `stim_*` keys; pick one"
                    .to_string(),
            );
        }
        let mut tile_profiles: Vec<TileStimulationProfile> = Vec::new();
        let mut tile_i = 0usize;
        loop {
            let probe = format!("stim_tile_{tile_i}_prf_hz");
            if !map.contains_key(probe.as_str()) {
                break;
            }
            let ptk = |suffix: &str| -> Result<f64, String> {
                let key = format!("stim_tile_{tile_i}_{suffix}");
                map.get(key.as_str())
                    .copied()
                    .ok_or_else(|| format!("manifest missing {key}"))
                    .and_then(|raw| {
                        raw.parse::<f64>()
                            .map_err(|e| format!("manifest {key} parse failed: {e}"))
                    })
            };
            tile_profiles.push(TileStimulationProfile {
                prf_hz: ptk("prf_hz")?,
                shift_s: ptk("shift_s")?,
                phase_deg: ptk("phase_deg")?,
                ramp_s: ptk("ramp_s")?,
                tbd_s: ptk("tbd_s")?,
                sd_s: ptk("sd_s")?,
                isi_s: ptk("isi_s")?,
                tt_s: ptk("tt_s")?,
                vpp_v: ptk("vpp_v")?,
                dead_time_s: ptk("dead_time_s")?,
            });
            tile_i += 1;
        }
        // Partial-tile-form guard: the loop terminated; if any `stim_tile_{i}_*` keys exist
        // for `i >= tile_profiles.len()`, the sequence is truncated — kwavers would produce a
        // wrong beam profile from the silently partial tile set.
        if any_tile_key && tile_profiles.len() < 4 {
            let first_gap = (tile_profiles.len()..).find(|&i| {
                map.keys()
                    .any(|k| k.starts_with(&format!("stim_tile_{i}_")))
            });
            if let Some(i) = first_gap {
                return Err(format!(
                    "gappy tile sequence: `stim_tile_{i}_*` keys present but earlier tile indices 0..{} are incomplete",
                    i,
                ));
            }
        }
        let stimulation = if !tile_profiles.is_empty() {
            None
        } else if format == MANIFEST_FORMAT_V2 {
            Some(StimulationProgram {
                prf_hz: parse("stim_prf_hz")?,
                tbd_s: parse("stim_tbd_s")?,
                sd_s: parse("stim_sd_s")?,
                isi_s: parse("stim_isi_s")?,
                tt_s: parse("stim_tt_s")?,
                vpp_v: parse("stim_vpp_v")?,
                dead_time_s: parse("stim_dead_time_s")?,
            })
        } else {
            None
        };
        Ok(DriverManifest {
            hv_board: get("hv_board")?.to_string(),
            tx_connector: get("tx_connector")?.to_string(),
            tx_nets,
            programming: get("programming")?.to_string(),
            aperture_m: parse("aperture_m")?,
            frequency_hz: parse("frequency_hz")?,
            sound_speed_m_s: parse("sound_speed_m_s")?,
            focal_m: parse("focal_m")?,
            timing_step_s: parse("timing_step_s")?,
            stimulation,
            tile_profiles,
        })
    }

    /// Read a manifest from disk.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_text(&text)
    }

    /// True iff this manifest carries the **per-tile stimulation programme** for the full
    /// 96-channel shield stack (4 tiles × 24 channels ⇒ `TX_0..TX_95`). Returns `false` for
    /// article-class single-tile baselines, v1 (no stim), and pre-tile v2 emissions. The
    /// `stimulation.is_none()` gate matches the validator's implicit contract — to_text
    /// enforces it on emission, so this catches hand-constructed manifests.
    #[must_use]
    pub fn is_full_stack_v2(&self) -> bool {
        self.stimulation.is_none()
            && self.tile_profiles.len() == 4
            && self.tx_nets.len() == TX_LANES_V2
    }

    /// Per-tile aggregation of [`TileStimulationProfile::protocol_load_j_s`] (J·s). Returns an
    /// empty `Vec` for v1 / pre-tile-v2 manifests so callers can branch on `len()`.
    #[must_use]
    pub fn per_tile_load_j_s(&self) -> Vec<f64> {
        self.tile_profiles
            .iter()
            .map(TileStimulationProfile::protocol_load_j_s)
            .collect()
    }

    /// Stack-level protocol-load proxy (J·s) summed across every tile, or the legacy
    /// single-stim value (`stimulation.protocol_load_j_s()`) for pre-tile v2 emissions so
    /// downstream kwavers validation can consume either shape via the same scalar.
    #[must_use]
    pub fn stack_load_j_s(&self) -> f64 {
        if !self.tile_profiles.is_empty() {
            self.tile_profiles
                .iter()
                .map(TileStimulationProfile::protocol_load_j_s)
                .sum()
        } else if let Some(s) = &self.stimulation {
            s.protocol_load_j_s()
        } else {
            0.0
        }
    }
}
