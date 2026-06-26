//! Consolidated tests for the `manifest` slice (Phase 4f carve-out): the stimulation-protocol
//! schema, the `DriverManifest` text round-trip (v1/v2, single-stim + per-tile forms, schema
//! guards), the v2 energy-budget validator, and the board extractor. Moved verbatim from the flat
//! `src/manifest.rs` `mod tests` block; `super::*` resolves the slice facade and the SSOT
//! schema-key/lane constants come from `crate::ssot`.

use super::*;
use crate::ssot::*;

fn sample_v2() -> DriverManifest {
    DriverManifest {
        hv_board: "hv.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..24).map(|i| format!("TX_{i}")).collect(),
        programming: "JTAG:TCK,TMS,TDI,TDO".into(),
        aperture_m: 4.3e-3,
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        stimulation: Some(StimulationProgram::article_default()),
        tile_profiles: Vec::new(),
    }
}

#[test]
fn manifest_v2_round_trips_with_stimulation_block() {
    let m = sample_v2();
    let text = m.to_text();
    // v2 format header is present.
    assert!(text.starts_with(&format!("format={MANIFEST_FORMAT_V2}\n")));
    // All stim_* keys are emitted exactly once.
    for k in [
        "stim_prf_hz",
        "stim_tbd_s",
        "stim_sd_s",
        "stim_isi_s",
        "stim_tt_s",
        "stim_vpp_v",
        "stim_dead_time_s",
    ] {
        assert_eq!(text.matches(&format!("{k}=")).count(), 1, "{k} emitted");
    }
    let parsed = DriverManifest::from_text(&text).expect("v2 manifest must parse");
    assert_eq!(parsed, m);
    assert_eq!(parsed.channel_count(), 24);
    // Stimulation block survives the round trip.
    let stim = parsed.stimulation.expect("v2 must carry stimulation");
    assert!((stim.prf_hz - 1.0e3).abs() < 1e-9);
    assert!((stim.tbd_s - 0.5e-3).abs() < 1e-12);
    assert!((stim.sd_s - 300.0e-3).abs() < 1e-9);
    assert!((stim.isi_s - 3.0).abs() < 1e-9);
    assert!((stim.tt_s - 18.0).abs() < 1e-9);
    assert!((stim.vpp_v - 150.0).abs() < 1e-9);
    assert!((stim.dead_time_s - 0.5e-3).abs() < 1e-12);
}

#[test]
fn v1_manifest_still_parses_for_backwards_compat() {
    // Pre-stimulation manifests in `output/full_driver/*.kv` and earlier examples are v1.
    // The parser still accepts them and yields `stimulation = None`.
    let v1 = "\
format=kicad-routing-driver-manifest-v1\n\
hv_board=hv7355_driver_tile.kicad_pcb\n\
tx_connector=J2\n\
tx_nets=TX_0,TX_1,TX_2,TX_3\n\
programming=JTAG:TCK,TMS,TDI,TDO\n\
aperture_m=4.300000000000e-03\n\
frequency_hz=2.000000000000e+06\n\
sound_speed_m_s=1.540000000000e+03\n\
focal_m=1.000000000000e-02\n\
timing_step_s=5.000000000000e-09\n";
    let m = DriverManifest::from_text(v1).expect("v1 must still parse");
    assert_eq!(m.tx_nets, vec!["TX_0", "TX_1", "TX_2", "TX_3"]);
    assert!(m.stimulation.is_none(), "v1 ⇒ no stimulation block");
}

#[test]
fn article_preset_matches_the_paper() {
    let s = StimulationProgram::article_default();
    assert_eq!(s.prf_hz, 1.0e3, "PRF = 1 kHz per MWSCAS 2024 article");
    assert_eq!(s.tbd_s, 0.5e-3, "TBD = 0.5 ms tone burst");
    assert_eq!(s.sd_s, 300.0e-3, "SD = 300 ms sonication window");
    assert_eq!(s.isi_s, 3.0, "ISI = 3 s inter-stimulus interval");
    assert_eq!(s.tt_s, 18.0, "TT = 18 s total stimulation time");
    assert_eq!(s.vpp_v, 150.0, "VPP = 150 V peak-to-peak at pulser output");
    // 18 s = 1 + ceil((18 − 0.3) / (0.3 + 3)) = 1 + 6 = 7 sonications.
    assert_eq!(s.sonication_count(), 7);
    // Tonal duty cycle: TBD × PRF = 0.5 ms × 1 kHz = 0.5 cycle-equivalents per PRF period
    // (i.e. each tone burst occupies roughly half of an inter-pulse interval).  This is a
    // duty in [0, 1]; clipping to u64 floors to 0 (the trailing-edge count is 1).  We
    // assert the float directly so the test reads identically to the physics.
    let tonal_duty = s.tbd_s * s.prf_hz;
    assert!(
        (tonal_duty - 0.5).abs() < 1e-12,
        "TBD × PRF = {tonal_duty} (expected ~0.5)"
    );
    // Cycle count per burst uses the **carrier** frequency (2 MHz, not PRF): 0.5 ms × 2 MHz
    // = 1000 cycles per tone burst — the figure acoustic simulations actually consume.
    let carrier_hz = 2.0e6_f64;
    let cycles_per_burst = (s.tbd_s * carrier_hz) as u64;
    assert_eq!(
        cycles_per_burst, 1_000,
        "0.5 ms TBD × 2 MHz carrier ⇒ 1000 cycles per burst"
    );
    // Protocol-load proxy sanity: VPP² × TT × duty = 150² × 18 × (0.3 / 3.3).
    let expected_load = 150.0 * 150.0 * s.tt_s * (s.sd_s / (s.sd_s + s.isi_s));
    assert!((s.protocol_load_j_s() - expected_load).abs() < 1e-6);
}

#[test]
fn sonication_count_handles_degenerate_programs() {
    let zero_sd = StimulationProgram {
        sd_s: 0.0,
        ..StimulationProgram::article_default()
    };
    assert_eq!(zero_sd.sonication_count(), 1, "zero SD ⇒ one safe default");
    let neg_tt = StimulationProgram {
        tt_s: -1.0,
        ..StimulationProgram::article_default()
    };
    assert_eq!(
        neg_tt.sonication_count(),
        1,
        "negative TT ⇒ one safe default"
    );
    // tt = sd exactly: first sonication fits, no second one possible.
    let tight = StimulationProgram {
        tt_s: 0.3,
        sd_s: 0.3,
        ..StimulationProgram::article_default()
    };
    assert_eq!(tight.sonication_count(), 1);
}

fn four_tile_v2_manifest() -> DriverManifest {
    // Four HV7355 tiles, each with its own PRF/SHIFT/PHASE/RAMP override. The 96-lane
    // binding is `4 × 24` so the kwavers validator can consume it directly.
    let mut tile_profiles = Vec::new();
    for i in 0..4 {
        tile_profiles.push(TileStimulationProfile::from_article_with(
            1.0e3 + i as f64 * 50.0,     // PRF tile-stagger: +50 Hz per tile
            i as f64 * 250.0e-3,         // SHIFT tile-stagger: +250 ms per tile
            i as f64 * 90.0,             // PHASE tile-stagger: +90° per tile
            25.0e-6 + i as f64 * 5.0e-6, // RAMP tile-stagger: +5 µs per tile
        ));
    }
    DriverManifest {
        hv_board: "hv7355_driver_stack.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..96).map(|i| format!("TX_{i}")).collect(),
        programming: "fpga:JTAG=TCK,TMS,TDI,TDO; stack-bus=4×24-lane".into(),
        aperture_m: 4.3e-3 * 95.0 / 15.0,
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        stimulation: None,
        tile_profiles,
    }
}
/// Relative-tolerance float comparison for round-trip tests. The `.12e` serialization
/// loses a couple of bits of mantissa, so a strict `==` fails on chained arithmetic
/// (`25e-6 + i * 5e-6`, `4.3e-3 * 95.0 / 15.0`); `1e-9` is comfortably tighter than any
/// physical transducer tolerance (PRF Hz, shift ms, phase °) and catches value-level bugs.
fn close_f64(label: &str, want: f64, got: f64) {
    let denom = want.abs().max(got.abs()).max(1e-30);
    assert!(
        (want - got).abs() / denom < 1e-9,
        "{label}: want {want}, got {got}"
    );
}

fn assert_manifest_approx_eq(got: &DriverManifest, want: &DriverManifest, message: &str) {
    assert_eq!(got.hv_board, want.hv_board, "{message}: hv_board");
    assert_eq!(
        got.tx_connector, want.tx_connector,
        "{message}: tx_connector"
    );
    assert_eq!(got.tx_nets, want.tx_nets, "{message}: tx_nets");
    assert_eq!(got.programming, want.programming, "{message}: programming");
    assert_eq!(got.stimulation, want.stimulation, "{message}: stimulation");
    close_f64(
        &format!("{message}: aperture_m"),
        want.aperture_m,
        got.aperture_m,
    );
    close_f64(
        &format!("{message}: frequency_hz"),
        want.frequency_hz,
        got.frequency_hz,
    );
    close_f64(
        &format!("{message}: sound_speed_m_s"),
        want.sound_speed_m_s,
        got.sound_speed_m_s,
    );
    close_f64(&format!("{message}: focal_m"), want.focal_m, got.focal_m);
    close_f64(
        &format!("{message}: timing_step_s"),
        want.timing_step_s,
        got.timing_step_s,
    );
    assert_eq!(
        got.tile_profiles.len(),
        want.tile_profiles.len(),
        "{message}: tile count"
    );
    for (i, (a, b)) in got
        .tile_profiles
        .iter()
        .zip(want.tile_profiles.iter())
        .enumerate()
    {
        close_f64(&format!("{message}: tile[{i}].prf_hz"), b.prf_hz, a.prf_hz);
        close_f64(
            &format!("{message}: tile[{i}].shift_s"),
            b.shift_s,
            a.shift_s,
        );
        close_f64(
            &format!("{message}: tile[{i}].phase_deg"),
            b.phase_deg,
            a.phase_deg,
        );
        close_f64(&format!("{message}: tile[{i}].ramp_s"), b.ramp_s, a.ramp_s);
        close_f64(&format!("{message}: tile[{i}].tbd_s"), b.tbd_s, a.tbd_s);
        close_f64(&format!("{message}: tile[{i}].sd_s"), b.sd_s, a.sd_s);
        close_f64(&format!("{message}: tile[{i}].isi_s"), b.isi_s, a.isi_s);
        close_f64(&format!("{message}: tile[{i}].tt_s"), b.tt_s, a.tt_s);
        close_f64(&format!("{message}: tile[{i}].vpp_v"), b.vpp_v, a.vpp_v);
        close_f64(
            &format!("{message}: tile[{i}].dead_time_s"),
            b.dead_time_s,
            a.dead_time_s,
        );
    }
}

#[test]
fn manifest_v2_tile_form_round_trips_four_distinct_profiles() {
    let m = four_tile_v2_manifest();
    assert!(
        m.is_full_stack_v2(),
        "4 tiles × 24 channels ⇒ 96-lane binding"
    );
    let text = m.to_text();
    // Each tile emits exactly 10 `stim_tile_{i}_*` keys, no legacy `stim_*` keys.
    for i in 0..4 {
        for suffix in [
            "prf_hz",
            "shift_s",
            "phase_deg",
            "ramp_s",
            "tbd_s",
            "sd_s",
            "isi_s",
            "tt_s",
            "vpp_v",
            "dead_time_s",
        ] {
            let key = format!("stim_tile_{i}_{suffix}");
            assert_eq!(text.matches(&format!("{key}=")).count(), 1, "{key} present");
        }
    }
    for k in [
        "stim_prf_hz",
        "stim_tbd_s",
        "stim_sd_s",
        "stim_isi_s",
        "stim_tt_s",
        "stim_vpp_v",
        "stim_dead_time_s",
    ] {
        assert_eq!(
            text.matches(&format!("{k}=")).count(),
            0,
            "no legacy {k} in tile-form"
        );
    }
    let parsed = DriverManifest::from_text(&text).expect("v2 tile-form must parse");
    assert_manifest_approx_eq(&parsed, &m, "tile-form round trip must equal original");
    assert!(parsed.stimulation.is_none(), "tile-form ⇒ no legacy stim");
    // Distinct per-tile PRFs survive the round trip (the .12e serialization is reversible
    // to <1e-9 relative, which is comfortably tighter than any physical PRF tolerance).
    for (i, p) in parsed.tile_profiles.iter().enumerate() {
        let close = |label: &str, want: f64, got: f64| {
            let denom = want.abs().max(1e-30);
            assert!(
                (want - got).abs() / denom < 1e-9,
                "{label}[{i}]: want {want}, got {got}"
            );
        };
        close("PRF", 1.0e3 + i as f64 * 50.0, p.prf_hz);
        close("SHIFT", (i as f64) * 250.0e-3, p.shift_s);
        close("PHASE", (i as f64) * 90.0, p.phase_deg);
        close("RAMP", 25.0e-6 + i as f64 * 5.0e-6, p.ramp_s);
    }
}

#[test]
fn v2_tile_form_validation_passes_for_a_well_sized_routed_board() {
    let m = four_tile_v2_manifest();
    // Routed HV7355 board: 8 A ampacity headroom (well above the tile-stagger peak).
    // The Smd4527 footprint (2 W) covers the article-class 50 pF + 150 V Vpp + +50 Hz PRF
    // stagger operating point with ~0.87 W margin on the worst-case tile[3] (1.13 W
    // dissipation ⇒ 2.0 − 1.13 = 0.87 W). Without `Smd4527`, an article-class board has
    // to retune the matching cap (35 pF drops dissipation to 0.69–0.79 W) to fit Smd2512,
    // or pick Smd1206 → reject (see the v2_tile_form_validation_rejects_underrated
    // _resistor_package test for the 1206 rejection path). Smd4527 keeps the article's
    // 50 pF clamped load intact.
    let inputs = EnergyBudgetInputs {
        c_load_f: 50e-12,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        ampacity_headroom_a: 8.0,
        damping_footprint: ResistorPackage::Smd4527,
    };
    let r = m
        .validate_v2_energy_budget(inputs)
        .expect("well-sized board");
    assert_eq!(r.lanes, 96);
    // Per-tile vector invariants: the validator must surface one entry per HV tile
    // (4 tiles × 24 channels ⇒ TX_0..TX_95) on every per-tile channel. Asserting the
    // lengths explicitly catches a future contributor who changes the validator loop
    // body to drop a tile — silently producing a Vec<3> is the kind of regression
    // kwavers downstream consumes as a wrong beam profile.
    assert_eq!(r.per_tile_protocol_load_j_s.len(), 4);
    assert_eq!(r.per_tile_device_total_w.len(), 4);
    assert_eq!(r.per_tile_pulser_total_w.len(), 4);
    assert_eq!(r.per_tile_resistor_w.len(), 4);
    assert_eq!(
        r.per_tile_resistor_margin_w.len(),
        4,
        "margin vector must mirror tile count"
    );
    // Per-tile margin sanity: each tile's dissipation under 1 W Smd2512 ⇒ margin ≥ 0.
    for &m in &r.per_tile_resistor_margin_w {
        assert!(
            m.is_finite() && m >= 0.0,
            "post-rejection margin is ≥ 0 W: got {m}"
        );
    }
    // Total = sum of 4 nearly-identical loads (~ 150²·18·0.0909 ≈ 36 818 J·s each).
    let expected_one = 150.0 * 150.0 * 18.0 * (0.3 / 3.3);
    for &tile_load in &r.per_tile_protocol_load_j_s {
        assert!((tile_load - expected_one).abs() < 1e-6);
    }
    assert!(
        (r.total_protocol_load_j_s - 4.0 * expected_one).abs() < 1e-3,
        "stack total = 4 × per-tile"
    );
    assert!(r.max_frame_duty > 0.0 && r.max_frame_duty <= 1.0);
    assert!(r.headroom_margin_a >= 0.0, "ample ampacity ⇒ no shortfall");
    // stack_load_j_s must equal the budget sum.
    assert!((m.stack_load_j_s() - r.total_protocol_load_j_s).abs() < 1e-9);
}

#[test]
fn v2_tile_form_validation_rejects_wrong_lane_count() {
    let mut m = four_tile_v2_manifest();
    m.tx_nets = (0..24).map(|i| format!("TX_{i}")).collect(); // single tile
    let inputs = EnergyBudgetInputs {
        c_load_f: 50e-12,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        ampacity_headroom_a: 8.0,
        damping_footprint: ResistorPackage::Smd2512,
    };
    let err = m
        .validate_v2_energy_budget(inputs)
        .expect_err("96-lane binding fails");
    assert!(
        err.contains("96 TX lanes"),
        "error names the required count"
    );
}

#[test]
fn v2_tile_form_validation_rejects_insufficient_ampacity_headroom() {
    let m = four_tile_v2_manifest();
    // Route ships with 0.05 A headroom ⇒ far below the article-class tile peak ⇒ fails.
    let inputs = EnergyBudgetInputs {
        c_load_f: 50e-12,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        ampacity_headroom_a: 0.05,
        damping_footprint: ResistorPackage::Smd2512,
    };
    let err = m
        .validate_v2_energy_budget(inputs)
        .expect_err("headroom overflow");
    assert!(
        err.contains("routed ampacity"),
        "error names the ampacity overflow"
    );
}

#[test]
fn single_stim_v2_falls_back_when_no_tile_keys() {
    // Backwards-compat: an existing v2 consumer that kept the legacy `stim_*` keys still
    // parses; tile_profiles is empty and stimulation carries the preset.
    let mut legacy = sample_v2();
    legacy.tx_nets = (0..24).map(|i| format!("TX_{i}")).collect();
    legacy.tile_profiles = Vec::new();
    let text = legacy.to_text();
    let parsed = DriverManifest::from_text(&text).expect("legacy v2 must parse");
    assert_manifest_approx_eq(&parsed, &legacy, "legacy v2 round trip");
    assert!(parsed.stimulation.is_some());
    assert!(parsed.tile_profiles.is_empty());
}

#[test]
fn from_text_rejects_mixed_tile_and_legacy_stim_keys() {
    // Hand-editing a v2 tile-form file with legacy `stim_*` keys is ambiguous; reject.
    let m = four_tile_v2_manifest();
    let mut text = m.to_text();
    // Append a legacy `stim_*` block alongside the tile-form keys.
    text.push_str("stim_prf_hz=1.000000000000e+03\n");
    let err = DriverManifest::from_text(&text)
        .expect_err("mixed tile + legacy keys must be a parser error");
    assert!(
        err.contains("mixed schemas"),
        "error names the schema conflict; got: {err}"
    );
}

#[test]
fn from_text_rejects_gappy_tile_sequence() {
    // Truncated file: tile_0 present, tile_2 present, tile_1 missing entirely.
    let mut text = String::new();
    text.push_str("format=kicad-routing-driver-manifest-v2\n");
    text.push_str("hv_board=hv.gicad_pcb\n");
    text.push_str("tx_connector=J2\n");
    text.push_str("tx_nets=");
    for i in 0..96 {
        if i > 0 {
            text.push(',');
        }
        text.push_str(&format!("TX_{i}"));
    }
    text.push('\n');
    // Tile 0 (full cluster):
    for suffix in [
        "prf_hz",
        "shift_s",
        "phase_deg",
        "ramp_s",
        "tbd_s",
        "sd_s",
        "isi_s",
        "tt_s",
        "vpp_v",
        "dead_time_s",
    ] {
        text.push_str(&format!("stim_tile_0_{suffix}=1.000000000000e+00\n"));
    }
    // Tile 2 (full cluster) but tile 1 missing:
    for suffix in [
        "prf_hz",
        "shift_s",
        "phase_deg",
        "ramp_s",
        "tbd_s",
        "sd_s",
        "isi_s",
        "tt_s",
        "vpp_v",
        "dead_time_s",
    ] {
        text.push_str(&format!("stim_tile_2_{suffix}=1.000000000000e+00\n"));
    }
    let err =
        DriverManifest::from_text(&text).expect_err("gappy tile sequence must be a parser error");
    assert!(
        err.contains("gappy tile sequence"),
        "error names the gap; got: {err}"
    );
}

#[test]
fn v2_tile_form_validation_surfaces_signed_margins_for_underrated_resistor_package() {
    let m = four_tile_v2_manifest();
    // 1206 footprint (250 mW) -- the article-class 56 ohm / 50 pF operating point
    // dissipates ~0.79-0.90 W across the series damping resistor (C.V^2.f.R_series/(R_on+
    // R_series) at the 0.5 frame duty on the +50 Hz PRF stagger), well above the
    // IPC-7351 70 degC rating for 1206. The validator used to return `Err(...)`
    // (`tile[..]: 1206 rated 0.250 W ...`); with the inline rejection gate LIFTED,
    // it now returns `Ok(report)` with a SIGNED per-tile margin vec (every entry
    // negative -- under-rate magnitude ~~ 0.54-0.65 W) and the kwavers-side 4th
    // `Check` is the sole gatekeeper (`report.all_pass == false` once it runs
    // through `validate_against_budget`).
    let inputs = EnergyBudgetInputs {
        c_load_f: 50e-12,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        ampacity_headroom_a: 8.0,
        damping_footprint: ResistorPackage::Smd1206,
    };
    let r = m
        .validate_v2_energy_budget(inputs)
        .expect("1206 must reach the kwavers gate (rejection gate lifted)");
    assert_eq!(r.per_tile_resistor_margin_w.len(), 4);
    // Sanity: every per-tile entry is NEGATIVE on 1206 + 50 pF + 150 V + +50 Hz stagger.
    let all_negative = r.per_tile_resistor_margin_w.iter().all(|&m| m < 0.0);
    assert!(
        all_negative,
        "1206 under-rates every tile at the article-class operating point: {:?}",
        r.per_tile_resistor_margin_w
    );
    // Largest under-rate magnitude = min margin. The +50 Hz PRF stagger lifts
    // dissipation from 0.984 W (tile[0]) up to 1.13 W (tile[3]) => worst-case
    // under-rate on tile[3] ~~ 1.13 - 0.250 = 0.88 W.
    let min_margin = r
        .per_tile_resistor_margin_w
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_margin < -0.5,
        "worst-case under-rate should be substantial (< -0.5 W); got {min_margin}"
    );
    // The LIFTED rejection pathway must NOT produce a user-visible Err string on
    // over-rate; sanity-shape test that protocol-load is still positive.
    assert!(r.per_tile_protocol_load_j_s.iter().all(|&x| x > 0.0));
}

#[test]
fn power_margin_w_returns_signed_under_over_and_at_the_edge() {
    // Pure unit test of the [`ResistorPackage::power_margin_w`] API -- rename-and-replace
    // of the former [`ResistorPackage::power_rating_check`] returning
    // `Result<f64, ResistorRatingError>`. Three branches pinned at exact signed floats:
    // * well-rated   --  dissipation <=  rated => positive headroom
    // * at-the-edge  --  dissipation ==  rated => signed zero
    // * over-rated   --  dissipation  >  rated => negative under-rate (kwavers opens here)
    let r_1206 = ResistorPackage::Smd1206;
    assert_eq!(r_1206.max_power_w(), 0.250, "1206 = 250 mW");
    assert_eq!(r_1206.name(), "1206");
    // 100 mW dissipation in a 1206 => +150 mW headroom.
    assert!(
        (r_1206.power_margin_w(0.100) - 0.150).abs() < 1e-12,
        "1206 - 0.100 = +0.150 margin"
    );
    // Exactly 250 mW => signed zero (edge case, fits without under-rate).
    assert!(r_1206.power_margin_w(0.250).abs() < 1e-12);
    // 300 mW => -50 mW under-rate magnitude (kwavers-side Check opens here).
    assert!(
        (r_1206.power_margin_w(0.300) + 0.050).abs() < 1e-12,
        "1206 - 0.300 = -0.050 under-rate"
    );
    // 2512 handles everything 1206 over-rates.
    let r_2512 = ResistorPackage::Smd2512;
    assert_eq!(r_2512.max_power_w(), 1.0, "2512 = 1 W");
    assert_eq!(r_2512.name(), "2512");
    assert!((r_2512.power_margin_w(0.984) - 0.016).abs() < 1e-12);
    // 2512-HE: the "tight-but-still-fits" middle envelope. Same 2512 land pattern
    // as Smd2512 but ~50 % more dissipation headroom (1.5 W vs 1 W) -- the
    // designer's preference on the 50 pF / 150 V article-class operating point
    // when they want to stay on the 2512 footprint and avoid the +150 % jump to
    // Smd4527. The kwavers consumer reads the resulting signed margin verbatim
    // (positive = headroom for protocol tweaks; negative = under-rate magnitude
    // for matching-cap tightening). Fits the article envelope WITHOUT cap retune.
    let r_2512he = ResistorPackage::Smd2512He;
    assert_eq!(r_2512he.max_power_w(), 1.5, "2512-HE = 1.5 W");
    assert_eq!(r_2512he.name(), "2512-HE");
    // 1.13 W (article-class worst-case tile[3]): +0.37 W margin without cap retune.
    assert!(
        (r_2512he.power_margin_w(1.13) - 0.37).abs() < 1e-12,
        "2512-HE margin = 1.5 - 1.13 = 0.37 W on article envelope"
    );
    // Exactly 1.5 W ==> signed zero (edge case, fits without under-rate).
    assert!(r_2512he.power_margin_w(1.5).abs() < 1e-12);
    // 2.5 W ==> signed -1.0 W under-rate magnitude.
    assert!(
        (r_2512he.power_margin_w(2.5) + 1.0).abs() < 1e-12,
        "2512-HE - 2.5 = -1.0 W under-rate"
    );
    // 4527 covers the article-class envelope (50 pF / 150 V / +50 Hz stagger => ~1.13 W
    // worst-case dissipation); with 0.87 W margin on tile[3] signed positive.
    let r_4527 = ResistorPackage::Smd4527;
    assert_eq!(r_4527.max_power_w(), 2.0, "4527 = 2 W");
    assert_eq!(r_4527.name(), "4527");
    assert!(
        (r_4527.power_margin_w(1.13) - 0.87).abs() < 1e-12,
        "4527 margin = 2.0 - 1.13 = 0.87 W"
    );
    // Over-rated on 4527: signed -0.5 W under-rate. Formerly a structured
    // `ResistorRatingError`; now a plain signed float the kwavers consumer reads.
    assert!(
        (r_4527.power_margin_w(2.5) + 0.5).abs() < 1e-12,
        "4527 - 2.5 = -0.5 W under-rate"
    );
}
