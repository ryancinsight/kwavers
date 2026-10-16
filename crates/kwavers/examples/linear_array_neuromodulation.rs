//! A 2 MHz, 30 × 1.5 mm linear array focused at 7–8 cm in brain tissue.
//!
//! Question: can a 120 kPa focal target at 6–8 cm, reached with 30 Vp,
//! drive neuromodulation, and what would higher pressures need?
//!
//! Stages, one shared element model:
//! 1. `focus`: full-wave PSTD of the 30 × 1.5 mm, 2 MHz array focused at 70 and
//!    80 mm in absorbing brain, against the analytical rectangular-Rayleigh oracle.
//! 2. `gain`: analytical gains at 60/70/80 mm and the axial profile.
//! 3. `vp`: drive-voltage scan of the sensitivity each focal target requires.
//! 4. `calibration`: square-plane source against the baffled Rayleigh value,
//!    lossless and absorbing, at λ/3 and λ/8.
//! 5. `convergence` (run only when named): the focus-stage foci at λ/4 and with the
//!    elevation extent doubled.
//!
//! Usage: `linear_array_neuromodulation [--out DIR] [--only STAGE]`, with STAGE one of
//! `focus`, `gain`, `vp`, `calibration`, `convergence`, or `all` (default; excludes `convergence`).

#![expect(
    clippy::print_stdout,
    reason = "an example is a program; its progress and output paths on stdout are its user interface"
)]

#[path = "linear_array_neuromodulation_common/mod.rs"]
mod common;

use anyhow::Result;
use common::analytic;
use common::metrics::{self, mechanical_index};
use common::plots;
use common::simulate::{self, RunOptions};
use common::{
    attenuation_np_m, time_step_s, wavelength_m, GridSpec, LinearArraySpec, ALPHA_POWER,
    FREQUENCY_HZ, POINTS_PER_WAVELENGTH, SOURCE_AMPLITUDE_PA,
};
use eunomia::convert::IntegerTarget;
use eunomia::FloatElement;
use leto::Array2;
use plotters::prelude::{BLACK, BLUE, GREEN, RED};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Array aperture and elevation, metres.
const ARRAY_APERTURE_M: f64 = 30.0e-3;
const ARRAY_ELEVATION_M: f64 = 1.5e-3;
/// Proposal acceptance pressure at 6–8 cm for 30 Vp, pascals.
const PROPOSAL_TARGET_PA: f64 = 120.0e3;
/// Proposal drive voltage for the acceptance pressure, volts peak.
const PROPOSAL_VP: f64 = 30.0;
/// Focal depths in question, millimetres.
const FOCUS_FOCI_MM: [f64; 2] = [70.0, 80.0];
/// Simulated time: arrival from the farthest element at 80 mm (53 µs), the
/// 4 µs burst, the 1 µs focusing spread and a tail.
const FOCUS_DURATION_S: f64 = 65.0e-6;
/// Physical extent of the focus-stage domain (lateral, elevation, depth), metres:
/// 160 × 128 × 384 cells at λ/3. The 1.5 mm elevation radiates a beam 43–48 mm
/// tall at the foci; with 16.5 mm of elevation the field striking the CPML at
/// grazing incidence returned and raised the 80 mm focal amplitude by 17%, while
/// 33 mm agrees with the oracle to within the λ/3 absorption error.
const FOCUS_EXTENT_M: [f64; 3] = [41.23e-3, 32.98e-3, 98.96e-3];
/// Depth of the array plane inside the domain, metres (12 cells at λ/3).
const ARRAY_DEPTH_M: f64 = 3.092e-3;
/// Depth past the array beyond which the axial comparison runs, metres: outside
/// the near field, where a pulsed peak and a CW amplitude differ.
const AXIAL_COMPARISON_START_M: f64 = 20.0e-3;

fn parse_args() -> (PathBuf, String) {
    let mut out = PathBuf::from("target/linear_array_neuromodulation");
    let mut only = String::from("all");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                if let Some(path) = args.next() {
                    out = PathBuf::from(path);
                }
            }
            "--only" => {
                if let Some(stage) = args.next() {
                    only = stage;
                }
            }
            _ => {}
        }
    }
    (out, only)
}

/// Grid holding the array aperture, its elevation, and the axis to 93 mm.
///
/// CPML absorbs what leaves the domain, so the lateral and elevation extent need
/// only contain the source and the focal path, not the whole diverging beam.
/// `points_per_wavelength` sets the resolution and `elevation_scale` multiplies
/// the elevation extent; the convergence stage varies both.
fn focus_grid(points_per_wavelength: f64, elevation_scale: f64) -> GridSpec {
    let dx = wavelength_m() / points_per_wavelength;
    let cells = |extent_m: f64| 2 * usize::from_truncated((0.5 * extent_m / dx).round());
    GridSpec {
        nx: cells(FOCUS_EXTENT_M[0]),
        ny: cells(FOCUS_EXTENT_M[1] * elevation_scale),
        nz: cells(FOCUS_EXTENT_M[2]),
        dx,
        array_plane_k: usize::from_truncated((ARRAY_DEPTH_M / dx).round()),
        pml_cells: pml_cells(points_per_wavelength),
    }
}

/// CPML cells for a resolution: 10 at λ/3, scaled so the layer stays 3.3 λ thick
/// and its absorption does not change with the grid under test.
fn pml_cells(points_per_wavelength: f64) -> usize {
    usize::from_truncated((10.0 * points_per_wavelength / POINTS_PER_WAVELENGTH).round())
}

fn cell(position_m: f64, dx: f64) -> usize {
    usize::from_truncated((position_m / dx).round())
}

/// One full-wave focus of the array compared with the brain oracle.
struct FocusResult {
    focal_mm: f64,
    sim_plane: Array2<f64>,
    oracle_plane: Array2<f64>,
    sim_gain: f64,
    oracle_gain: f64,
    p_neg_gain: f64,
    peak_gain: f64,
    peak_depth_mm: f64,
    oracle_peak_gain: f64,
    oracle_peak_depth_mm: f64,
    fwhm_lateral_mm: f64,
    oracle_fwhm_lateral_mm: f64,
    axial_relative_error: f64,
}

fn run_focus(grid: &GridSpec, array: &LinearArraySpec, focal_mm: f64) -> Result<FocusResult> {
    let dx = grid.dx;
    let focus = array.focus_m(focal_mm * 1.0e-3, 0.0, grid);
    let (fi, fj, fk) = (cell(focus[0], dx), cell(focus[1], dx), cell(focus[2], dx));
    println!(
        "[{focal_mm} mm focus] PSTD on {} x {} x {} cells, dx {:.4} mm, brain absorption without dispersion",
        grid.nx,
        grid.ny,
        grid.nz,
        dx * 1.0e3
    );
    let steps = usize::from_truncated((FOCUS_DURATION_S / time_step_s(dx)).round());
    let options = RunOptions {
        nonlinear: false,
        absorbing: true,
        steps,
    };
    let run = simulate::run_focused(grid, array, focus, SOURCE_AMPLITUDE_PA, options)?;
    let apertures = analytic::element_apertures(array, focus, 1.0)?;
    let oracle_plane = analytic::xz_plane(grid, fj, &apertures)?;
    // The oracle is read at the same cell as the simulation, not at the exact focus.
    let focal_cell_m = [
        f64::from_count(fi) * dx,
        f64::from_count(fj) * dx,
        f64::from_count(fk) * dx,
    ];
    let oracle_gain = analytic::amplitudes_at(&[focal_cell_m], &apertures)?[0];

    let mut sim_plane = Array2::<f64>::zeros([grid.nx, grid.nz]);
    for i in 0..grid.nx {
        for k in 0..grid.nz {
            sim_plane[[i, k]] = run.harmonic_gain[[i, fj, k]];
        }
    }
    let lateral = |plane: &Array2<f64>| (0..grid.nx).map(|i| plane[[i, fk]]).collect::<Vec<f64>>();
    let axial = |plane: &Array2<f64>| (0..grid.nz).map(|k| plane[[fi, k]]).collect::<Vec<f64>>();
    let axial_sim = axial(&sim_plane);
    let axial_oracle = axial(&oracle_plane);

    let start = grid.array_plane_k + usize::from_truncated((AXIAL_COMPARISON_START_M / dx).round());
    let end = grid.nz - grid.pml_cells;
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for k in start..end {
        num += (axial_sim[k] - axial_oracle[k]).powi(2);
        den += axial_oracle[k].powi(2);
    }
    // The focal bump of a weakly focused aperture peaks short of the nominal focus but
    // beyond half of it; nearer maxima belong to the near-field interference pattern.
    let bump_start = start
        .max(grid.array_plane_k + usize::from_truncated((0.5 * focal_mm * 1.0e-3 / dx).round()));
    let peak_of = |profile: &[f64]| {
        (bump_start..end).fold((bump_start, 0.0_f64), |best, k| {
            if profile[k] > best.1 {
                (k, profile[k])
            } else {
                best
            }
        })
    };
    let (sim_peak_k, peak_gain) = peak_of(&axial_sim);
    let (oracle_peak_k, oracle_peak_gain) = peak_of(&axial_oracle);
    let depth_mm = |k: usize| f64::from_count(k - grid.array_plane_k) * dx * 1.0e3;
    let width_mm = |profile: &[f64]| metrics::fwhm(profile, dx).unwrap_or(f64::NAN) * 1.0e3;
    Ok(FocusResult {
        focal_mm,
        sim_gain: run.harmonic_gain[[fi, fj, fk]],
        oracle_gain,
        p_neg_gain: run.peak_neg[[fi, fj, fk]] / SOURCE_AMPLITUDE_PA,
        peak_gain,
        peak_depth_mm: depth_mm(sim_peak_k),
        oracle_peak_gain,
        oracle_peak_depth_mm: depth_mm(oracle_peak_k),
        fwhm_lateral_mm: width_mm(&lateral(&sim_plane)),
        oracle_fwhm_lateral_mm: width_mm(&lateral(&oracle_plane)),
        axial_relative_error: (num / den.max(f64::MIN_POSITIVE)).sqrt(),
        sim_plane,
        oracle_plane,
    })
}

fn stage_focus(out: &Path) -> Result<()> {
    let grid = focus_grid(POINTS_PER_WAVELENGTH, 1.0);
    let array = LinearArraySpec::centered(ARRAY_APERTURE_M, ARRAY_ELEVATION_M, &grid);
    println!(
        "array: {} elements, pitch {:.4} mm, aperture {:.1} x {:.1} mm",
        array.element_count(),
        array.pitch_m * 1.0e3,
        ARRAY_APERTURE_M * 1.0e3,
        ARRAY_ELEVATION_M * 1.0e3
    );
    let mut summary = String::from(
        "focal_mm,sim_gain,oracle_gain,p_neg_gain,sim_peak_gain,sim_peak_depth_mm,oracle_peak_gain,oracle_peak_depth_mm,fwhm_lat_mm,oracle_fwhm_lat_mm,axial_rel_error,surface_kpa_for_120kpa,mi_at_120kpa\n",
    );
    let mut results = Vec::new();
    for focal_mm in FOCUS_FOCI_MM {
        let r = run_focus(&grid, &array, focal_mm)?;
        let label = format!("d2_{focal_mm:.0}mm");
        write_plane_csv(&out.join(format!("sim_xz_{label}.csv")), &r.sim_plane)?;
        write_plane_csv(&out.join(format!("oracle_xz_{label}.csv")), &r.oracle_plane)?;
        let surface = metrics::required_surface_pa(PROPOSAL_TARGET_PA, r.sim_gain);
        writeln!(
            summary,
            "{},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4},{:.2},{:.3},{:.3},{:.4},{:.1},{:.4}",
            r.focal_mm,
            r.sim_gain,
            r.oracle_gain,
            r.p_neg_gain,
            r.peak_gain,
            r.peak_depth_mm,
            r.oracle_peak_gain,
            r.oracle_peak_depth_mm,
            r.fwhm_lateral_mm,
            r.oracle_fwhm_lateral_mm,
            r.axial_relative_error,
            surface * 1.0e-3,
            mechanical_index(PROPOSAL_TARGET_PA, FREQUENCY_HZ)
        )?;
        println!(
            "  focal gain sim {:.4} / oracle {:.4}; axial peak sim {:.4} at {:.1} mm / oracle {:.4} at {:.1} mm; axial error {:.3}; lateral FWHM sim {:.2} mm / oracle {:.2} mm",
            r.sim_gain,
            r.oracle_gain,
            r.peak_gain,
            r.peak_depth_mm,
            r.oracle_peak_gain,
            r.oracle_peak_depth_mm,
            r.axial_relative_error,
            r.fwhm_lateral_mm,
            r.oracle_fwhm_lateral_mm
        );
        results.push(r);
    }
    fs::write(out.join("focus_summary.csv"), &summary)?;

    let dx_mm = grid.dx * 1.0e3;
    let fi = cell(grid.lateral_center_m(), grid.dx);
    if let Some(first) = results.first() {
        plots::write_db_maps(
            &out.join("fig1_d2_xz_70mm_sim_vs_oracle.png"),
            &[
                ("kwavers full-wave (dB re focus)", &first.sim_plane),
                ("Rayleigh oracle (dB re focus)", &first.oracle_plane),
            ],
            dx_mm,
            first.sim_gain,
            -30.0,
        )?;
        let fk = cell(array.z_m + first.focal_mm * 1.0e-3, grid.dx);
        let lateral_mm: Vec<f64> = (0..grid.nx)
            .map(|i| (f64::from_count(i) - f64::from_count(fi)) * dx_mm)
            .collect();
        plots::write_profiles(
            &out.join("fig3_d2_focal_lateral_70mm.png"),
            "Lateral profile at the 70 mm focus",
            "lateral position (mm)",
            "pressure gain (Pa/Pa)",
            &lateral_mm,
            &[
                (
                    "kwavers",
                    BLUE,
                    (0..grid.nx).map(|i| first.sim_plane[[i, fk]]).collect(),
                ),
                (
                    "oracle",
                    RED,
                    (0..grid.nx).map(|i| first.oracle_plane[[i, fk]]).collect(),
                ),
            ],
        )?;
    }
    let depth_mm: Vec<f64> = (0..grid.nz)
        .map(|k| (f64::from_count(k) - f64::from_count(grid.array_plane_k)) * dx_mm)
        .collect();
    let colours = [(BLUE, RED), (BLACK, GREEN)];
    let series: Vec<(String, plotters::style::RGBColor, Vec<f64>)> = results
        .iter()
        .zip(colours)
        .flat_map(|(r, (sim_colour, oracle_colour))| {
            [
                (
                    format!("kwavers, {:.0} mm focus", r.focal_mm),
                    sim_colour,
                    (0..grid.nz)
                        .map(|k| r.sim_plane[[fi, k]])
                        .collect::<Vec<f64>>(),
                ),
                (
                    format!("oracle, {:.0} mm focus", r.focal_mm),
                    oracle_colour,
                    (0..grid.nz)
                        .map(|k| r.oracle_plane[[fi, k]])
                        .collect::<Vec<f64>>(),
                ),
            ]
        })
        .collect();
    let named: Vec<(&str, plotters::style::RGBColor, Vec<f64>)> = series
        .iter()
        .map(|(name, colour, values)| (name.as_str(), *colour, values.clone()))
        .collect();
    plots::write_profiles(
        &out.join("fig2_d2_axial_gain.png"),
        "On-axis pressure gain in brain: kwavers vs oracle",
        "depth from array (mm)",
        "pressure gain (Pa/Pa)",
        &depth_mm,
        &named,
    )?;
    Ok(())
}

/// Grid and domain sensitivity of the focus-stage result.
///
/// Repeats the 70 mm focus at λ/4 (resolution) and the 80 mm focus with the
/// elevation extent doubled again (boundary proximity of the spreading elevation beam),
/// each against the oracle on its own grid; `focus_summary.csv` holds the
/// λ/3 baseline of both foci.
fn stage_convergence(out: &Path) -> Result<()> {
    let cases = [
        ("resolution lambda/4", 4.0, 1.0, 70.0),
        ("elevation x2", POINTS_PER_WAVELENGTH, 2.0, 80.0),
    ];
    let mut csv = String::from(
        "case,points_per_wavelength,elevation_scale,nx,ny,nz,focal_mm,sim_gain,oracle_gain,sim_over_oracle,axial_rel_error,fwhm_lat_mm\n",
    );
    for (label, ppw, elevation_scale, focal_mm) in cases {
        let grid = focus_grid(ppw, elevation_scale);
        let array = LinearArraySpec::centered(ARRAY_APERTURE_M, ARRAY_ELEVATION_M, &grid);
        println!("[convergence] {label}, {focal_mm} mm");
        let r = run_focus(&grid, &array, focal_mm)?;
        writeln!(
            csv,
            "{label},{ppw},{elevation_scale},{},{},{},{focal_mm},{:.4},{:.4},{:.4},{:.4},{:.3}",
            grid.nx,
            grid.ny,
            grid.nz,
            r.sim_gain,
            r.oracle_gain,
            r.sim_gain / r.oracle_gain,
            r.axial_relative_error,
            r.fwhm_lateral_mm
        )?;
        println!(
            "  sim {:.4} / oracle {:.4} (ratio {:.4}); axial error {:.4}; lateral FWHM {:.2} mm",
            r.sim_gain,
            r.oracle_gain,
            r.sim_gain / r.oracle_gain,
            r.axial_relative_error,
            r.fwhm_lateral_mm
        );
        fs::write(out.join("convergence.csv"), &csv)?;
    }
    Ok(())
}

/// Source scaling and absorption against the baffled Rayleigh value of a square
/// plane, lossless and with brain absorption, at the production resolution and at λ/8.
fn stage_calibration(out: &Path) -> Result<()> {
    let mut csv = String::from("points_per_wavelength,absorbing,sim_over_rayleigh\n");
    for points_per_wavelength in [POINTS_PER_WAVELENGTH, 8.0] {
        for absorbing in [false, true] {
            let ratio = simulate::run_plane_wave_calibration(
                SOURCE_AMPLITUDE_PA,
                points_per_wavelength,
                absorbing,
            )?;
            println!(
                "calibration at lambda/{points_per_wavelength}, absorbing {absorbing}: sim / Rayleigh {ratio:.4} (expect 1)"
            );
            writeln!(csv, "{points_per_wavelength},{absorbing},{ratio:.5}")?;
            fs::write(out.join("calibration.csv"), &csv)?;
        }
    }
    Ok(())
}

fn write_plane_csv(path: &Path, plane: &Array2<f64>) -> Result<()> {
    let [nx, nz] = plane.shape();
    let mut text = String::with_capacity(nx * nz * 12);
    for i in 0..nx {
        let row: Vec<String> = (0..nz).map(|k| format!("{:.6e}", plane[[i, k]])).collect();
        text.push_str(&row.join(","));
        text.push('\n');
    }
    fs::write(path, text)?;
    Ok(())
}

fn stage_gain(out: &Path) -> Result<f64> {
    let array = LinearArraySpec::at(ARRAY_APERTURE_M, ARRAY_ELEVATION_M, [0.0, 0.0, 0.0]);
    println!(
        "array: {} elements at pitch {:.4} mm, aperture {:.1} x {:.1} mm",
        array.element_count(),
        array.pitch_m * 1.0e3,
        ARRAY_APERTURE_M * 1.0e3,
        ARRAY_ELEVATION_M * 1.0e3
    );
    let mut csv = String::from("focal_mm,gain_pa_per_pa,surface_kpa_for_120kpa,mi_at_120kpa\n");
    let mut gain_70 = f64::NAN;
    for focal_mm in [60.0_f64, 70.0, 80.0] {
        let focus = [0.0, 0.0, focal_mm * 1.0e-3];
        let gain = analytic::focal_amplitude(&array, focus, 1.0)?;
        if (focal_mm - 70.0).abs() < f64::EPSILON {
            gain_70 = gain;
        }
        let required = metrics::required_surface_pa(PROPOSAL_TARGET_PA, gain);
        writeln!(
            csv,
            "{focal_mm},{gain:.6},{:.2},{:.4}",
            required * 1.0e-3,
            mechanical_index(PROPOSAL_TARGET_PA, FREQUENCY_HZ)
        )?;
        println!(
            "  focus {focal_mm} mm: gain {gain:.4} Pa/Pa, surface for 120 kPa {:.1} kPa",
            required * 1.0e-3
        );
    }
    fs::write(out.join("array_gain.csv"), csv)?;

    // Axial gain along the broadside axis, oracle only, with attenuation.
    let depths: Vec<f64> = (0..=400)
        .map(|index| 30.0 + 0.15 * f64::from_count(index))
        .collect();
    let points: Vec<[f64; 3]> = depths.iter().map(|&z| [0.0, 0.0, z * 1.0e-3]).collect();
    let apertures = analytic::element_apertures(&array, [0.0, 0.0, 0.07], 1.0)?;
    let gains = analytic::amplitudes_at(&points, &apertures)?;
    plots::write_profiles(
        &out.join("fig4_d2_axial_gain.png"),
        "Array axial gain (oracle, 70 mm focus, broadside)",
        "depth z (mm)",
        "pressure gain (Pa/Pa)",
        &depths,
        &[("focal gain", BLACK, gains)],
    )?;

    // Focal-plane cuts at each focus. The 1.5 mm elevation is unfocused and
    // diffracts to a half-maximum width of about 40 mm, wider than the full-wave
    // grid, so its extent comes from the oracle.
    let offsets_mm: Vec<f64> = (0..=800)
        .map(|index| -40.0 + 0.1 * f64::from_count(index))
        .collect();
    let width_mm = |profile: &[f64]| metrics::fwhm(profile, 0.1e-3).unwrap_or(f64::NAN) * 1.0e3;
    let mut cuts = String::from("focal_mm,offset_mm,lateral_gain,elevation_gain\n");
    for focal_mm in FOCUS_FOCI_MM {
        let depth_m = focal_mm * 1.0e-3;
        let focused = analytic::element_apertures(&array, [0.0, 0.0, depth_m], 1.0)?;
        let cut = |axis: usize| -> Result<Vec<f64>> {
            let points: Vec<[f64; 3]> = offsets_mm
                .iter()
                .map(|&offset| {
                    let mut point = [0.0, 0.0, depth_m];
                    point[axis] = offset * 1.0e-3;
                    point
                })
                .collect();
            analytic::amplitudes_at(&points, &focused)
        };
        let lateral = cut(0)?;
        let elevation = cut(1)?;
        println!(
            "  {focal_mm} mm focal spot (oracle): FWHM lateral {:.2} mm, elevation {:.1} mm",
            width_mm(&lateral),
            width_mm(&elevation)
        );
        for ((offset, lat), elev) in offsets_mm.iter().zip(&lateral).zip(&elevation) {
            writeln!(cuts, "{focal_mm},{offset:.1},{lat:.6},{elev:.6}")?;
        }
        if (focal_mm - 70.0).abs() < f64::EPSILON {
            plots::write_profiles(
                &out.join("fig7_d2_focal_spot_70mm.png"),
                "Focal spot at 70 mm (oracle): narrow along the array, wide in elevation",
                "offset from axis (mm)",
                "pressure gain (Pa/Pa)",
                &offsets_mm,
                &[
                    ("along the array", BLUE, lateral),
                    ("elevation", RED, elevation),
                ],
            )?;
        }
    }
    fs::write(out.join("focal_plane.csv"), cuts)?;
    Ok(gain_70)
}

fn stage_vp(out: &Path, gain_70: f64) -> Result<()> {
    let implied_sensitivity = PROPOSAL_TARGET_PA / (gain_70 * PROPOSAL_VP);
    println!(
        "implied surface sensitivity of the 120 kPa target at 70 mm and 30 Vp: {implied_sensitivity:.1} Pa/V"
    );
    let vps: Vec<f64> = (0..=27)
        .map(|index| 30.0 + 10.0 * f64::from_count(index))
        .collect();
    let mut csv = String::from("vp_volts,target_kpa,required_sensitivity_pa_per_v,focal_kpa_at_implied_sensitivity,surface_kpa,surface_shock_distance_mm\n");
    let targets = [120.0e3_f64, 500.0e3, 1.0e6, 2.0e6];
    let mut series = Vec::new();
    for &target in &targets {
        let mut required = Vec::new();
        for &vp in &vps {
            let sensitivity = target / (gain_70 * vp);
            required.push(sensitivity);
        }
        series.push(required);
    }
    for (row, &vp) in vps.iter().enumerate() {
        for (column, &target) in targets.iter().enumerate() {
            // Distortion accrues where the pressure is highest, at the array face.
            let surface = implied_sensitivity * vp;
            writeln!(
                csv,
                "{vp:.1},{:.0},{:.2},{:.1},{:.1},{:.2}",
                target * 1.0e-3,
                series[column][row],
                gain_70 * surface * 1.0e-3,
                surface * 1.0e-3,
                metrics::shock_distance_m(surface, FREQUENCY_HZ) * 1.0e3
            )?;
        }
    }
    fs::write(out.join("vp_scan.csv"), csv)?;

    let labels = ["120 kPa", "500 kPa", "1 MPa", "2 MPa"];
    let colours = [BLUE, RED, BLACK, plotters::prelude::GREEN];
    let named: Vec<(&str, plotters::style::RGBColor, Vec<f64>)> = labels
        .iter()
        .zip(colours)
        .zip(series)
        .map(|((&label, colour), values)| (label, colour, values))
        .collect();
    plots::write_profiles(
        &out.join("fig5_vp_required_sensitivity.png"),
        "Surface sensitivity needed for each focal target, array at 70 mm",
        "drive voltage Vp (V)",
        "required surface pressure per volt (Pa/V)",
        &vps,
        &named,
    )?;

    // Linear focal pressure at the target-implied sensitivity, with the targets.
    let focal_kpa: Vec<f64> = vps
        .iter()
        .map(|&vp| gain_70 * implied_sensitivity * vp * 1.0e-3)
        .collect();
    let constant = |kpa: f64| vec![kpa; vps.len()];
    plots::write_profiles(
        &out.join("fig6_vp_focal_pressure.png"),
        "Focal pressure at 70 mm vs drive, target-implied sensitivity (linear)",
        "drive voltage Vp (V)",
        "focal pressure (kPa)",
        &vps,
        &[
            ("focal pressure", BLACK, focal_kpa),
            ("120 kPa", BLUE, constant(120.0)),
            ("500 kPa", RED, constant(500.0)),
            ("1 MPa", plotters::prelude::GREEN, constant(1000.0)),
        ],
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let (out, only) = parse_args();
    fs::create_dir_all(&out)?;
    plots::init_fonts()?;
    println!("output: {}", out.display());
    println!(
        "brain: c = 1546 m/s, f = {} MHz, lambda = {:.4} mm, attenuation = {:.2} Np/m (power {})",
        FREQUENCY_HZ / 1.0e6,
        wavelength_m() * 1.0e3,
        attenuation_np_m(),
        ALPHA_POWER
    );
    let run = |stage: &str| only == "all" || only == stage;
    if run("focus") {
        stage_focus(&out)?;
    }
    let mut gain_70 = f64::NAN;
    if run("gain") || run("vp") {
        gain_70 = stage_gain(&out)?;
    }
    if run("vp") {
        stage_vp(&out, gain_70)?;
    }
    if only == "convergence" {
        stage_convergence(&out)?;
    }
    if run("calibration") {
        stage_calibration(&out)?;
    }
    println!("done");
    Ok(())
}
