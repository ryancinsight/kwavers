//! Replication of the Fullwave 2.5 heterogeneous power-law attenuation result.
//!
//! Fullwave 2.5 (Sode & Pinton, pinton-lab/fullwave25) reports accurate
//! attenuation modelling with **both** the coefficient `α₀` and the exponent `γ`
//! varying spatially, validated over `α₀ = 0.25…0.75` dB·cm⁻¹·MHz⁻ᵞ and
//! `γ = 0.4…1.6`. This example reproduces that claim in kwavers by simulation,
//! not by inspecting the fit:
//!
//! 1. **Homogeneous sweep** — for each `(α₀, γ)` in that envelope, a broadband
//!    pulse propagates through a medium built by
//!    [`ViscoacousticMemorySolver::from_power_law_fields`]. `α(f)` is recovered
//!    from the **spectral ratio** of two downstream sensors,
//!    `α(f) = −ln(|P₂|/|P₁|)/d`, and compared to the prescribed law. The
//!    measurement never consults the fitted spectrum, so a mis-fit shows up as
//!    a discrepancy rather than cancelling out.
//! 2. **Heterogeneous layers** — an abdominal-wall-like stack of fat and muscle
//!    layers that differ in `γ` as well as `α₀`. The transmitted spectrum is
//!    compared to the path-length-weighted law `Σₖ αₖ(f)·Lₖ`, which is the
//!    exact plane-wave prediction and which **no uniform-exponent medium can
//!    reproduce** — the layers' exponents differ, so their sum is not a power
//!    law of any single exponent.
//!
//! ## Measured accuracy
//!
//! Over the whole envelope the simulated `α(f)` matches the prescribed law to
//! **3.4 % worst case, and to 0.5 % across the band interior** — the residual
//! concentrates at 0.6 and 4.6 MHz, the edges where the excitation carries
//! least energy. The heterogeneous fat/muscle stack, where `γ` varies along the
//! propagation path, matches the exact path-weighted prediction to **1.0 %**.
//!
//! This runs on **three** relaxation arms. With the relaxation times optimized
//! rather than log-spaced (`RelaxationTimePlacement::Optimized`), three arms
//! reproduce the fit to 0.16 % analytically, so the residual above is the
//! *measurement*, not the medium: six arms move the worst case only from 3.4 %
//! to 3.0 %. Each arm dropped is one fewer memory field per voxel in the
//! solver, which is the dimension that decides whether a 3-D heterogeneous run
//! fits in memory at all.
//!
//! Two measurement details are load-bearing, both established by experiment
//! rather than assumed (KW-SOL-072):
//!
//! - **The analysis gate must not be tapered.** A Hann taper over the gate
//!   biased the recovered `α` low by 8–19 %, multiplicatively and independently
//!   of sensor separation, because the far-sensor pulse is dispersively
//!   broadened and so is weighted differently by the taper than the near-sensor
//!   pulse. The pulse decays to zero well inside the gate, so a rectangular
//!   gate truncates nothing and needs no taper at all.
//! - **The gate must be centred on the true emission time**, `3·PULSE_WIDTH_S`
//!   after step 0, not on step 0 plus the transit time.
//!
//! The scheme itself was exonerated analytically before the measurement was
//! suspected — see `discrete_dispersion_matches_continuum` in the solver tests.
//!
//! Outputs (under `target/fullwave_attenuation/`):
//! - `attenuation_sweep.png`  — measured vs prescribed `α(f)`, log-log
//! - `attenuation_sweep.csv`  — every measured point
//! - `layered_medium.csv`     — heterogeneous-layer comparison
//!
//! Run: `cargo run --release --example heterogeneous_power_law_attenuation`

use anyhow::{anyhow, Result};
use kwavers_solver::forward::viscoacoustic::ViscoacousticMemorySolver;
use leto::Array3;
use plotters::prelude::*;
use std::f64::consts::TAU;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

const OUT_DIR: &str = "target/fullwave_attenuation";

/// 1 dB = 1/8.685889638 Np; cm⁻¹ → m⁻¹ multiplies by 100.
const NEPER_PER_DB: f64 = 1.0 / 8.685_889_638_065_035;

// ── Grid and acquisition ──────────────────────────────────────────────────────
// 1-D water-like column. `dx` resolves 5 MHz at ~10 points per wavelength, well
// inside the pseudospectral solver's 2-point Nyquist limit.
const N: usize = 1536;
const DX: f64 = 3.0e-5;
const RHO: f64 = 1000.0;
const C0: f64 = 1540.0;
const F_REF: f64 = 1.0e6;
/// Fit band; also the band over which `α(f)` is reported.
const F_MIN: f64 = 0.5e6;
const F_MAX: f64 = 5.0e6;
const N_ARMS: usize = 3;

const SOURCE_INDEX: usize = 220;
const SENSOR_NEAR: usize = 380;
const SENSOR_FAR: usize = 980;
const ABSORBER_CELLS: usize = 128;
/// Boundary damping rate; large enough that a pulse entering the layer decays by
/// many e-folds over its width at the CFL step used here.
const ABSORBER_GAMMA: f64 = 6.0e6;
const STEPS: usize = 9000;

/// Centre frequency of the excitation, placed mid-band on a log scale so the
/// pulse carries usable energy across 0.5–5 MHz.
const PULSE_CENTRE_HZ: f64 = 1.6e6;
/// Gaussian envelope width; `≈0.35/f_c` gives a ~2-cycle broadband pulse.
const PULSE_WIDTH_S: f64 = 0.35 / PULSE_CENTRE_HZ;

/// Half-width, in steps, of the window isolating the direct arrival at each
/// sensor (see [`windowed_magnitude`]).
const GATE_HALF_STEPS: usize = 700;

/// Frequencies at which the spectral ratio is evaluated.
const ANALYSIS_FREQUENCIES: [f64; 8] = [0.6e6, 0.9e6, 1.2e6, 1.6e6, 2.2e6, 3.0e6, 3.8e6, 4.6e6];

/// Envelope Fullwave 2.5 validates.
const ALPHA0_DB: [f64; 3] = [0.25, 0.5, 0.75];
const GAMMAS: [f64; 5] = [0.4, 0.7, 1.0, 1.3, 1.6];

#[derive(Debug, Clone, Copy)]
struct SweepRow {
    alpha0_db: f64,
    gamma: f64,
    frequency_hz: f64,
    prescribed_np_m: f64,
    measured_np_m: f64,
}

impl SweepRow {
    fn relative_error(&self) -> f64 {
        (self.measured_np_m - self.prescribed_np_m).abs() / self.prescribed_np_m
    }
}

fn alpha_np_m(alpha0_db: f64) -> f64 {
    alpha0_db * NEPER_PER_DB * 100.0
}

/// CFL-limited step. The pseudospectral operator resolves up to the Nyquist
/// wavenumber `π/dx`, so stability needs `dt ≤ dx/(π·c)`; a 0.4 safety factor
/// leaves room for the unrelaxed (high-frequency) speed exceeding `c₀`.
fn time_step() -> f64 {
    0.4 * DX / (std::f64::consts::PI * C0)
}

/// Gaussian-modulated sine excitation, zero-padded to `STEPS`.
fn excitation(dt: f64) -> Vec<f64> {
    let t0 = 3.0 * PULSE_WIDTH_S;
    let n_active = ((6.0 * PULSE_WIDTH_S) / dt).ceil() as usize;
    (0..n_active.min(STEPS))
        .map(|k| {
            let t = k as f64 * dt - t0;
            (-(t / PULSE_WIDTH_S).powi(2)).exp() * (TAU * PULSE_CENTRE_HZ * t).sin()
        })
        .collect()
}

/// Single-frequency DFT magnitude of the **direct arrival** at a sensor.
///
/// The damping layer is an absorber, not a PML: its ramp reflects a small
/// residual, and a soft source radiates in both directions. Those late arrivals
/// interfere with the direct pulse and put comb notches in the raw spectrum —
/// at 0.9 MHz the ungated ratio misreports `α` by a factor of three. Gating to
/// the direct arrival is what makes this a plane-wave attenuation measurement
/// at all.
///
/// The gate is **rectangular**, of half-width [`GATE_HALF_STEPS`], centred on
/// the emission time plus the transit `(sensor − source)·dx/c₀`. Rectangular is
/// deliberate: the pulse decays to zero well inside the gate, so there is no
/// truncation to taper away, and a taper would instead weight the dispersively
/// broadened far-sensor pulse differently from the near one — an 8–19 % bias in
/// the recovered `α` (KW-SOL-072). The half-width is wide enough for the pulse
/// plus its dispersive spread and narrower than the earliest contaminant at
/// either sensor (1445 steps at the near sensor, 6725 at the far one).
fn windowed_magnitude(trace: &[f64], sensor_index: usize, frequency_hz: f64, dt: f64) -> f64 {
    let emission = (3.0 * PULSE_WIDTH_S / dt).round() as usize;
    let arrival = emission + ((sensor_index - SOURCE_INDEX) as f64 * DX / C0 / dt).round() as usize;
    let lo = arrival.saturating_sub(GATE_HALF_STEPS);
    let hi = (arrival + GATE_HALF_STEPS).min(trace.len());

    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for (offset, &v) in trace[lo..hi].iter().enumerate() {
        let phase = TAU * frequency_hz * (lo + offset) as f64 * dt;
        re += v * phase.cos();
        im -= v * phase.sin();
    }
    re.hypot(im)
}

/// Build a medium from per-voxel fields, drive it with the broadband pulse, and
/// return the two sensor traces.
fn run_pulse(
    alpha_field: &Array3<f64>,
    gamma_field: &Array3<f64>,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let rho_field = Array3::from_elem([N, 1, 1], RHO);
    let c_field = Array3::from_elem([N, 1, 1], C0);

    let mut solver = ViscoacousticMemorySolver::from_power_law_fields(
        N,
        1,
        1,
        DX,
        1.0,
        1.0,
        dt,
        &rho_field,
        &c_field,
        alpha_field,
        gamma_field,
        F_MIN,
        F_MAX,
        N_ARMS,
        F_REF,
    )?;
    solver.enable_absorbing_layer(ABSORBER_CELLS, ABSORBER_GAMMA);
    solver.add_pressure_source((SOURCE_INDEX, 0, 0), excitation(dt))?;
    let near = solver.add_pressure_sensor((SENSOR_NEAR, 0, 0))?;
    let far = solver.add_pressure_sensor((SENSOR_FAR, 0, 0))?;

    for _ in 0..STEPS {
        solver.step();
    }
    Ok((
        solver.sensor_trace(near).to_vec(),
        solver.sensor_trace(far).to_vec(),
    ))
}

/// Two-sensor spectral amplitudes of one run, at every analysis frequency.
fn sensor_spectra(near: &[f64], far: &[f64], dt: f64) -> Result<Vec<(f64, f64)>> {
    ANALYSIS_FREQUENCIES
        .iter()
        .map(|&f| {
            let a = windowed_magnitude(near, SENSOR_NEAR, f, dt);
            let b = windowed_magnitude(far, SENSOR_FAR, f, dt);
            if a <= 0.0 || b <= 0.0 {
                return Err(anyhow!("no spectral energy at {f:e} Hz"));
            }
            Ok((a, b))
        })
        .collect()
}

/// Measure `α(f)` by the **reference-normalized** two-sensor spectral ratio,
///
/// ```text
///   α(f) = −ln[ (P_far/P_near) / (P_far^ref/P_near^ref) ] / d
/// ```
///
/// where the reference is the identical run in a lossless medium of the same
/// `ρ` and `c₀`. The raw two-sensor ratio carries the source spectrum, the gate
/// transfer function, and the discrete solver's own frequency response, none of
/// which is absorption; taking them out against a lossless reference is the
/// standard insertion-loss measurement and removes a smooth ~10 % frequency-
/// dependent bias that the raw ratio otherwise reports as attenuation.
///
/// Only the medium differs between the two runs, so what survives the double
/// ratio is exactly the absorption introduced between the sensors.
fn measure_attenuation(run: &[(f64, f64)], reference: &[(f64, f64)]) -> Result<Vec<(f64, f64)>> {
    let separation = (SENSOR_FAR - SENSOR_NEAR) as f64 * DX;
    ANALYSIS_FREQUENCIES
        .iter()
        .zip(run)
        .zip(reference)
        .map(|((&f, &(near, far)), &(near_ref, far_ref))| {
            let ratio = (far / near) / (far_ref / near_ref);
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(anyhow!("degenerate spectral ratio at {f:e} Hz"));
            }
            Ok((f, -ratio.ln() / separation))
        })
        .collect()
}

/// Lossless run of the identical geometry — the measurement reference.
fn reference_spectra(dt: f64) -> Result<Vec<(f64, f64)>> {
    let alpha = Array3::<f64>::zeros([N, 1, 1]);
    let gamma = Array3::<f64>::from_elem([N, 1, 1], 1.0);
    let (near, far) = run_pulse(&alpha, &gamma, dt)?;
    sensor_spectra(&near, &far, dt)
}

fn homogeneous_sweep(dt: f64, reference: &[(f64, f64)]) -> Result<Vec<SweepRow>> {
    let mut rows = Vec::new();
    for &alpha0_db in &ALPHA0_DB {
        for &gamma in &GAMMAS {
            let alpha0 = alpha_np_m(alpha0_db);
            let alpha_field = Array3::from_elem([N, 1, 1], alpha0);
            let gamma_field = Array3::from_elem([N, 1, 1], gamma);
            let (near, far) = run_pulse(&alpha_field, &gamma_field, dt)?;
            let spectra = sensor_spectra(&near, &far, dt)?;
            for (frequency_hz, measured_np_m) in measure_attenuation(&spectra, reference)? {
                rows.push(SweepRow {
                    alpha0_db,
                    gamma,
                    frequency_hz,
                    prescribed_np_m: alpha0 * (frequency_hz / F_REF).powf(gamma),
                    measured_np_m,
                });
            }
        }
    }
    Ok(rows)
}

/// One layer of the abdominal-wall stack.
struct Layer {
    name: &'static str,
    cells: usize,
    alpha0_db: f64,
    gamma: f64,
}

/// Fat/muscle stack spanning the two sensors. Fat absorbs weakly with a low
/// exponent, muscle more strongly with a near-linear one (Duck 1990, Ch. 4).
const LAYERS: [Layer; 4] = [
    Layer {
        name: "fat",
        cells: 150,
        alpha0_db: 0.4,
        gamma: 0.6,
    },
    Layer {
        name: "muscle",
        cells: 150,
        alpha0_db: 0.75,
        gamma: 1.1,
    },
    Layer {
        name: "fat",
        cells: 150,
        alpha0_db: 0.4,
        gamma: 0.6,
    },
    Layer {
        name: "muscle",
        cells: 150,
        alpha0_db: 0.75,
        gamma: 1.1,
    },
];

fn layered_medium(dt: f64, reference: &[(f64, f64)]) -> Result<Vec<(f64, f64, f64)>> {
    let mut alpha_field = Array3::<f64>::zeros([N, 1, 1]);
    let mut gamma_field = Array3::<f64>::from_elem([N, 1, 1], 1.0);
    let mut cursor = SENSOR_NEAR;
    for layer in &LAYERS {
        for i in cursor..(cursor + layer.cells).min(N) {
            alpha_field[[i, 0, 0]] = alpha_np_m(layer.alpha0_db);
            gamma_field[[i, 0, 0]] = layer.gamma;
        }
        cursor += layer.cells;
    }
    assert!(
        cursor <= SENSOR_FAR,
        "layer stack must terminate before the far sensor"
    );

    let (near, far) = run_pulse(&alpha_field, &gamma_field, dt)?;
    let spectra = sensor_spectra(&near, &far, dt)?;
    let separation = (SENSOR_FAR - SENSOR_NEAR) as f64 * DX;

    measure_attenuation(&spectra, reference)?
        .into_iter()
        .map(|(f, measured)| {
            // Exact plane-wave prediction: attenuation integrates along the path,
            // so the effective coefficient over the sensor separation is the
            // path-length-weighted mean of each layer's own power law.
            let integrated: f64 = LAYERS
                .iter()
                .map(|l| {
                    alpha_np_m(l.alpha0_db) * (f / F_REF).powf(l.gamma) * (l.cells as f64 * DX)
                })
                .sum();
            (f, integrated / separation, measured)
        })
        .map(Ok)
        .collect()
}

fn write_sweep_csv(path: &Path, rows: &[SweepRow]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "alpha0_db_cm_mhz_gamma,gamma,frequency_hz,prescribed_np_m,measured_np_m,relative_error"
    )?;
    for r in rows {
        writeln!(
            file,
            "{},{},{:e},{:.6},{:.6},{:.6}",
            r.alpha0_db,
            r.gamma,
            r.frequency_hz,
            r.prescribed_np_m,
            r.measured_np_m,
            r.relative_error()
        )?;
    }
    Ok(())
}

fn write_layered_csv(path: &Path, rows: &[(f64, f64, f64)]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "frequency_hz,path_weighted_np_m,measured_np_m,relative_error"
    )?;
    for &(f, predicted, measured) in rows {
        writeln!(
            file,
            "{f:e},{predicted:.6},{measured:.6},{:.6}",
            (measured - predicted).abs() / predicted
        )?;
    }
    Ok(())
}

/// Log-log `α(f)`: prescribed power laws as lines, simulated points as markers.
/// Plotted at the mid `α₀` so the five exponents are legible; the CSV carries
/// every combination.
fn write_plot(path: &Path, rows: &[SweepRow]) -> Result<()> {
    let shown: Vec<&SweepRow> = rows.iter().filter(|r| r.alpha0_db == 0.5).collect();
    let alpha0 = alpha_np_m(0.5);

    let root = BitMapBackend::new(path, (900, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Power-law attenuation, alpha0 = 0.5 dB/cm/MHz^gamma",
            ("sans-serif", 22),
        )
        .margin(16)
        .x_label_area_size(52)
        .y_label_area_size(64)
        .build_cartesian_2d((0.5f64..5.0f64).log_scale(), (1.0f64..120.0f64).log_scale())?;
    chart
        .configure_mesh()
        .x_desc("frequency [MHz]")
        .y_desc("alpha [Np/m]")
        .draw()?;

    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &BLACK];
    for (idx, &gamma) in GAMMAS.iter().enumerate() {
        let colour = palette[idx % palette.len()];
        let curve = (0..=100).map(|k| {
            let f_mhz = 0.5 * (10.0f64).powf(k as f64 / 100.0);
            (f_mhz, alpha0 * f_mhz.powf(gamma))
        });
        chart
            .draw_series(LineSeries::new(curve, colour.stroke_width(2)))?
            .label(format!("gamma = {gamma} (prescribed)"))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], colour.stroke_width(2))
            });

        chart.draw_series(
            shown
                .iter()
                .filter(|r| (r.gamma - gamma).abs() < f64::EPSILON)
                .map(|r| {
                    Circle::new(
                        (r.frequency_hz / 1.0e6, r.measured_np_m),
                        4,
                        colour.filled(),
                    )
                }),
        )?;
    }
    chart
        .configure_series_labels()
        // Upper-left is the only empty quadrant: every series rises to the
        // right, so a right-hand legend hides the high-frequency points.
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

fn main() -> Result<()> {
    let dt = time_step();
    let out_dir = PathBuf::from(OUT_DIR);
    fs::create_dir_all(&out_dir)?;

    println!("Fullwave 2.5 heterogeneous power-law attenuation replication");
    println!(
        "grid {N} x 1 x 1, dx = {DX:e} m, dt = {dt:e} s, {STEPS} steps, \
         sensors {SENSOR_NEAR} -> {SENSOR_FAR}"
    );

    let reference = reference_spectra(dt)?;
    let rows = homogeneous_sweep(dt, &reference)?;
    let sweep_csv = out_dir.join("attenuation_sweep.csv");
    let sweep_png = out_dir.join("attenuation_sweep.png");
    write_sweep_csv(&sweep_csv, &rows)?;
    write_plot(&sweep_png, &rows)?;

    println!("\nhomogeneous sweep — worst relative error per (alpha0, gamma):");
    println!("{:>10} {:>8} {:>14}", "alpha0_db", "gamma", "worst_rel_err");
    let mut worst_overall = 0.0_f64;
    for &alpha0_db in &ALPHA0_DB {
        for &gamma in &GAMMAS {
            let worst = rows
                .iter()
                .filter(|r| r.alpha0_db == alpha0_db && (r.gamma - gamma).abs() < f64::EPSILON)
                .map(SweepRow::relative_error)
                .fold(0.0_f64, f64::max);
            worst_overall = worst_overall.max(worst);
            println!("{alpha0_db:>10} {gamma:>8} {worst:>14.4}");
        }
    }
    println!("worst over the whole envelope: {worst_overall:.4}");

    let layered = layered_medium(dt, &reference)?;
    let layered_csv = out_dir.join("layered_medium.csv");
    write_layered_csv(&layered_csv, &layered)?;

    println!("\nheterogeneous stack along the propagation path:");
    for layer in &LAYERS {
        println!(
            "  {:<7} {:>4} cells   alpha0 = {:.2} dB/cm/MHz^gamma   gamma = {:.2}",
            layer.name, layer.cells, layer.alpha0_db, layer.gamma
        );
    }
    println!("\nrecovered vs the exact path-weighted prediction:");
    println!(
        "{:>12} {:>18} {:>14} {:>12}",
        "freq [MHz]", "path-weighted", "measured", "rel_err"
    );
    for &(f, predicted, measured) in &layered {
        println!(
            "{:>12.2} {predicted:>18.3} {measured:>14.3} {:>12.4}",
            f / 1.0e6,
            (measured - predicted).abs() / predicted
        );
    }

    println!("\npng: {}", sweep_png.display());
    println!("csv: {}", sweep_csv.display());
    println!("csv: {}", layered_csv.display());
    Ok(())
}
