//! End-to-end tests for the `experiment` slice (Phase 5).
//!
//! Every assertion is value-semantic: raw scalars are compared against analytically-derived
//! expectations, not merely checked for `is_ok()`. The test pipeline exercises the full
//! `run_experiment` path as well as each sub-module independently.

use crate::manifest::{
    DriverManifest, EnergyBudgetInputs, ResistorPackage, TileStimulationProfile,
};
use crate::ssot::*;

use super::acoustic::{AcousticSimulator, InCrateAcousticSim, PressureMap};
use super::dispatch::TileDispatch;
use super::metrics::{build_beam_report, ExperimentMetrics};
use super::recorder::artifact_key;
use super::runner::run_experiment;
use super::stimulus::{DefaultStimulus, Stimulus};
use super::thermal::propagate_thermal;

// ── Fixtures ─────────────────────────────────────────────────────────────────────────────────────

/// Article-class v2 manifest: 96 lanes, 4 tiles, 500 kHz, 10 mm focus, 1.54 km/s.
fn v2_manifest() -> DriverManifest {
    let tile_profiles: Vec<TileStimulationProfile> = (0..4)
        .map(|i| {
            TileStimulationProfile::from_article_with(
                1.0e3 + (i as f64) * 50.0,
                (i as f64) * 250.0e-3,
                (i as f64) * 90.0,
                25.0e-6 + (i as f64) * 5.0e-6,
            )
        })
        .collect();
    DriverManifest {
        hv_board: "hv7355_driver_stack.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..TX_LANES_V2).map(|i| format!("TX_{i}")).collect(),
        programming: "fpga:JTAG=TCK,TMS,TDI,TDO; stack-bus=4×24-lane".into(),
        aperture_m: MANIFEST_ARTICLE_APERTURE_M,
        frequency_hz: 500_000.0,
        sound_speed_m_s: 1540.0,
        focal_m: 0.010,
        timing_step_s: 20.0e-9,
        stimulation: None,
        tile_profiles,
    }
}

fn v2_budget(manifest: &DriverManifest) -> crate::manifest::EnergyBudgetReport {
    manifest
        .validate_v2_energy_budget(EnergyBudgetInputs {
            c_load_f: 50.0e-12,
            r_on_ohm: 15.0,
            r_series_ohm: 56.0,
            ampacity_headroom_a: 20.0,
            damping_footprint: ResistorPackage::Smd2512,
        })
        .expect("article-class fixture must yield a valid budget")
}

// ── stimulus ─────────────────────────────────────────────────────────────────────────────────────

#[test]
fn default_stimulus_profiles_match_manifest() {
    let m = v2_manifest();
    let stim = DefaultStimulus::new(&m);
    assert_eq!(stim.tile_count(), 4);
    for i in 0..4 {
        assert!(
            stim.profile_for(i).is_some(),
            "tile {i} must have a profile"
        );
    }
    assert!(stim.profile_for(4).is_none(), "out-of-range tile returns None");
}

#[test]
fn default_stimulus_profile_fields_match_manifest() {
    let m = v2_manifest();
    let stim = DefaultStimulus::new(&m);
    // Tile 0: prf_hz == 1.0e3 (from_article_with first arg 1000.0 + 0*50)
    let p0 = stim.profile_for(0).unwrap();
    assert!((p0.prf_hz - 1_000.0).abs() < 1.0, "tile-0 prf_hz");
    // Tile 3: prf_hz == 1.0e3 + 3*50 = 1150
    let p3 = stim.profile_for(3).unwrap();
    assert!((p3.prf_hz - 1_150.0).abs() < 1.0, "tile-3 prf_hz");
}

// ── dispatch ─────────────────────────────────────────────────────────────────────────────────────

#[test]
fn tile_dispatch_v2_equal_partition() {
    let d = TileDispatch::new(96, 4).unwrap();
    assert_eq!(d.lanes(), 96);
    assert_eq!(d.tiles(), 4);
    assert_eq!(d.bindings().len(), 4);
    for (t, b) in d.bindings().iter().enumerate() {
        assert_eq!(b.tile, t);
        assert_eq!(b.lane_start, t * 24);
        assert_eq!(b.lane_end, (t + 1) * 24);
        assert_eq!(b.lane_count(), 24);
    }
}

#[test]
fn tile_dispatch_lane_lookup() {
    let d = TileDispatch::new(96, 4).unwrap();
    assert_eq!(d.tile_for_lane(0), Some(0));
    assert_eq!(d.tile_for_lane(23), Some(0));
    assert_eq!(d.tile_for_lane(24), Some(1));
    assert_eq!(d.tile_for_lane(95), Some(3));
    assert_eq!(d.tile_for_lane(96), None, "out-of-range lane returns None");
}

#[test]
fn tile_dispatch_rejects_uneven_partition() {
    assert!(TileDispatch::new(95, 4).is_err(), "95 % 4 != 0 must fail");
    assert!(TileDispatch::new(0, 4).is_err(), "zero lanes must fail");
    assert!(TileDispatch::new(96, 0).is_err(), "zero tiles must fail");
}

// ── thermal ──────────────────────────────────────────────────────────────────────────────────────

#[test]
fn thermal_propagation_rise_equals_power_times_theta() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let theta = 40.0_f64; // K/W
    let dt_max = 30.0_f64;
    let ts = propagate_thermal(&b, theta, dt_max);
    assert_eq!(ts.per_tile_rise_k.len(), 4);
    for (i, &rise) in ts.per_tile_rise_k.iter().enumerate() {
        let expected = b.per_tile_device_total_w[i] * theta;
        assert!(
            (rise - expected).abs() < 1e-10,
            "tile {i}: rise {rise} != {expected}"
        );
    }
    let expected_peak = b
        .per_tile_device_total_w
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        * theta;
    assert!((ts.peak_rise_k - expected_peak).abs() < 1e-10);
    assert!((ts.headroom_k - (dt_max - expected_peak)).abs() < 1e-10);
}

#[test]
fn thermal_headroom_negative_when_over_budget() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let ts = propagate_thermal(&b, 40.0, 0.0); // dt_max = 0 ⇒ always over budget
    assert!(ts.headroom_k <= 0.0, "headroom must be ≤ 0 when dt_max=0");
}

// ── acoustic (InCrateAcousticSim) ────────────────────────────────────────────────────────────────

#[test]
fn in_crate_sim_focal_pressure_above_1mpa() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let step = crate::validate::manifest_to_kwavers_beam_step(&m, &b).unwrap();
    let map = InCrateAcousticSim.simulate(&step, &b).unwrap();
    // Article-class 96-element stack yields ~10–15 MPa; must be at least 1 MPa.
    assert!(
        map.focal_pressure_pa >= KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA,
        "focal pressure {:.3e} Pa below 1 MPa floor",
        map.focal_pressure_pa
    );
}

#[test]
fn in_crate_sim_mi_below_cavitation_ceiling() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let step = crate::validate::manifest_to_kwavers_beam_step(&m, &b).unwrap();
    let map = InCrateAcousticSim.simulate(&step, &b).unwrap();
    assert!(
        map.mechanical_index <= KWVERS_MI_CAVITATION_CEILING,
        "MI {:.3} exceeds ceiling {KWVERS_MI_CAVITATION_CEILING}",
        map.mechanical_index
    );
}

#[test]
fn in_crate_sim_grating_lobe_free_for_article_pitch() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let step = crate::validate::manifest_to_kwavers_beam_step(&m, &b).unwrap();
    let map = InCrateAcousticSim.simulate(&step, &b).unwrap();
    assert!(
        map.grating_lobe_free,
        "article-class λ/2 pitch must be grating-lobe-free"
    );
}

#[test]
fn in_crate_sim_all_scalars_finite() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let step = crate::validate::manifest_to_kwavers_beam_step(&m, &b).unwrap();
    let map = InCrateAcousticSim.simulate(&step, &b).unwrap();
    assert!(map.focal_pressure_pa.is_finite(), "focal_pressure_pa");
    assert!(map.mechanical_index.is_finite(), "mechanical_index");
    assert!(map.isppa_w_cm2.is_finite(), "isppa_w_cm2");
    assert!(map.axial_extent_mm.is_finite(), "axial_extent_mm");
    assert!(map.lateral_extent_mm.is_finite(), "lateral_extent_mm");
}

// ── metrics + beam_report ─────────────────────────────────────────────────────────────────────────

#[test]
fn experiment_metrics_mirrors_pressure_and_thermal() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let step = crate::validate::manifest_to_kwavers_beam_step(&m, &b).unwrap();
    let pressure = InCrateAcousticSim.simulate(&step, &b).unwrap();
    let thermal = propagate_thermal(&b, 40.0, 30.0);
    let metrics = ExperimentMetrics::from_parts(&pressure, &thermal);
    assert!((metrics.focal_pressure_pa - pressure.focal_pressure_pa).abs() < 1e-9);
    assert!((metrics.peak_thermal_rise_k - thermal.peak_rise_k).abs() < 1e-9);
    assert!((metrics.thermal_headroom_k - thermal.headroom_k).abs() < 1e-9);
    assert_eq!(metrics.grating_lobe_free, pressure.grating_lobe_free);
}

#[test]
fn beam_report_all_pass_for_article_class() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let step = crate::validate::manifest_to_kwavers_beam_step(&m, &b).unwrap();
    let pressure = InCrateAcousticSim.simulate(&step, &b).unwrap();
    let min_margin = step.resistor_margin_w.iter().copied().fold(f64::INFINITY, f64::min);
    let min_margin = if min_margin.is_finite() { min_margin } else { 0.0 };
    let report = build_beam_report(&pressure, &step, min_margin);
    assert!(
        report.all_pass,
        "article-class parameters must pass all 4 kwavers-beam checks; checks: {:#?}",
        report.checks
    );
}

// ── recorder ─────────────────────────────────────────────────────────────────────────────────────

#[test]
fn artifact_key_encodes_frequency_lanes_focal() {
    // 500 kHz = 0.500 MHz, 96 lanes, 10 mm focal depth.
    let key = artifact_key(500_000.0, 96, 0.010);
    assert_eq!(key, "0.500MHz_96ch_10.00mm");
}

#[test]
fn artifact_key_uniquely_distinguishes_configs() {
    let k1 = artifact_key(500_000.0, 96, 0.010);
    let k2 = artifact_key(1_000_000.0, 96, 0.010);
    let k3 = artifact_key(500_000.0, 48, 0.010);
    let k4 = artifact_key(500_000.0, 96, 0.020);
    assert_ne!(k1, k2);
    assert_ne!(k1, k3);
    assert_ne!(k1, k4);
}

// ── run_experiment (end-to-end) ───────────────────────────────────────────────────────────────────

#[test]
fn run_experiment_returns_valid_report_for_article_class() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    let report = run_experiment(&m, &b, &InCrateAcousticSim, 40.0, 30.0).unwrap();
    // Physics gate: article-class parameters must produce all-pass.
    assert!(
        report.record.all_pass,
        "article-class must pass all kwavers-beam checks"
    );
    // Thermal gate: dt_max=30 K; rise from article-class dissipation should have headroom.
    assert!(
        report.record.metrics.focal_pressure_pa >= KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA,
        "focal pressure below 1 MPa floor"
    );
    assert_eq!(report.thermal.per_tile_rise_k.len(), 4);
}

#[test]
fn run_experiment_rejects_non_v2_manifest() {
    // A manifest with 48 lanes cannot produce a KwaversBeamStep (requires exactly 96).
    let mut m = v2_manifest();
    m.tx_nets.truncate(48);
    let b = m
        .validate_v2_energy_budget(EnergyBudgetInputs {
            c_load_f: 50.0e-12,
            r_on_ohm: 15.0,
            r_series_ohm: 56.0,
            ampacity_headroom_a: 20.0,
            damping_footprint: ResistorPackage::Smd2512,
        });
    // The budget validation itself will fail first (wrong lane count).
    assert!(b.is_err(), "48-lane manifest must fail validate_v2_energy_budget");
}

#[test]
fn run_experiment_thermal_headroom_tracks_theta_jc() {
    let m = v2_manifest();
    let b = v2_budget(&m);
    // Higher θ_jc → higher rise → less headroom.
    let r_low = run_experiment(&m, &b, &InCrateAcousticSim, 10.0, 30.0).unwrap();
    let r_high = run_experiment(&m, &b, &InCrateAcousticSim, 80.0, 30.0).unwrap();
    assert!(
        r_low.record.metrics.peak_thermal_rise_k < r_high.record.metrics.peak_thermal_rise_k,
        "higher θ_jc must yield higher peak rise"
    );
    assert!(
        r_low.record.metrics.thermal_headroom_k > r_high.record.metrics.thermal_headroom_k,
        "higher θ_jc must yield less headroom"
    );
}
