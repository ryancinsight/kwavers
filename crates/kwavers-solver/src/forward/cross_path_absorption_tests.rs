//! Differential verification across the two heterogeneous-absorption paths.
//!
//! kwavers realizes power-law absorption two ways, and each is otherwise
//! checked only against its own analytic target — never against the other:
//!
//! | path | mechanism | spatial operator |
//! |---|---|---|
//! | PSTD | Treeby–Cox fractional Laplacian, `IFFT(\|k\|^(y−s)·FFT(·))` | spectral |
//! | FDTD | relaxation memory variables on a fitted `τ` grid | finite difference |
//!
//! These are genuinely different mathematics for the same physical law — a
//! frequency-domain operator applied spectrally versus a time-domain ODE system
//! integrated locally — so agreement between them is **independent-oracle**
//! evidence. Two backends running the same algorithm would only be differential
//! evidence.
//!
//! # Why a travelling pulse rather than a standing wave
//!
//! An earlier revision measured a standing wave, which forces both solvers to
//! interact with their walls — and their walls differ: FDTD's are rigid, PSTD
//! is periodic. One initial condition is therefore a *different mode* in each,
//! evolving at a different frequency. The raw decay rates differed by `r^γ` with
//! `r ≈ 1.23`, and normalizing each solver by its own measured frequency
//! recovered only 9 % agreement. That is a property of standing waves, not of
//! either wall, so it did not go away when the walls changed (ADR 106).
//!
//! A pulse launched away from every boundary, measured between two interior
//! sensors before any reflection or wrap-around can reach them, never sees a
//! wall at all. The boundary condition drops out of the comparison instead of
//! having to be corrected for.
//!
//! # A capability difference the comparison exposes
//!
//! The exponents avoid `γ = 1.0` because **PSTD cannot represent it**: the
//! Treeby–Cox dispersion coefficient is `η = 2α₀c₀^y·tan(πy/2)`, which diverges
//! at `y = 1`, and the solver rejects that configuration. The relaxation path
//! has no such singularity — `γ = 1` is an ordinary point for a fitted
//! spectrum. Linear frequency dependence, the textbook soft-tissue
//! idealization, is available only through relaxation.

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::{power_law_db_cm_to_np_m, AbsorptionMode};
use kwavers_source::GridSource;
use std::f64::consts::TAU;

use crate::forward::fdtd::config::{FdtdAbsorption, FdtdConfig};
use crate::forward::fdtd::FdtdSolver;
use crate::forward::pstd::config::{BoundaryConfig, KSpaceMethod};
use crate::forward::pstd::{PSTDConfig, PSTDSolver};

// ── Geometry ────────────────────────────────────────────────────────────────
//
// The packet is launched **one-way** (rightward) by pairing the initial
// pressure with `u = p/(ρc)`, the plane-wave relation. Without that, an initial
// pressure alone splits into two counter-propagating halves and the left-going
// half wraps around PSTD's periodic domain straight into the far sensor —
// arriving *before* the direct pulse for any usable sensor placement. A one-way
// launch removes that contaminant entirely and leaves only the far wall.
//
// Distances in cells; the packet covers one cell every `1/0.4 = 2.5` steps at
// the CFL factor used:
//
// | arrival                      | near | far  |
// |---|---|---|
// | direct                       | 120  |  500 |
// | FDTD far-wall reflection     | —    |  660 |
// | PSTD wrap-around             | —    | 1140 |
//
// A ±60-cell gate around the far arrival closes at 560 cells, clear of the
// earliest contaminant at 660.
//
// The 380-cell separation is 19 mm, which is the point: at 4 mm the attenuation
// over the interval is ~2 % and sits below the measurement floor. An earlier
// revision of this test used 4 mm and recovered a negative α.
const N: usize = 640;
const DX: f64 = 5.0e-5;
const C0: f64 = SOUND_SPEED_WATER_SIM;
const RHO0: f64 = DENSITY_WATER_NOMINAL;
const F_REF: f64 = 1.0e6;

/// Transverse extent, in cells.
///
/// One is enough because the walls are rigid, so a transversely uniform field
/// has exactly zero transverse gradient and the slab behaves as the 1-D line
/// this measurement wants. That was not true while the closure was
/// zero-extension: those walls were pressure-release, a uniform field had a
/// large gradient at them, and a purely axial packet lost more than half its
/// energy to transverse waveguide modes within 150 steps (KW-SOL-085). This
/// used four cells then, and it did not help — the far sensor read exactly zero.
const TRANSVERSE: usize = 1;

const SOURCE_INDEX: usize = 60;
const SENSOR_NEAR: usize = 180;
const SENSOR_FAR: usize = 560;
const STEPS: usize = 1450;

/// Centre frequency of the launched packet.
const PULSE_CENTRE_HZ: f64 = 1.5e6;
/// Gaussian envelope width in metres — about six tenths of a wavelength.
///
/// This sets the usable band, and it is the whole measurement. A Gaussian
/// envelope `W` wide has a spectral 1/e half-width of `1/W` in wavenumber, so a
/// relative bandwidth of `1/(W·k₀) = λ₀/(2πW)`. At `W = 2λ₀` that is ±8 %, which
/// put 0.8 MHz roughly `1e-15` down the tail — the spectral ratio there measured
/// nothing but leakage, and returned *negative* attenuation for both solvers.
/// At `W = 0.6λ₀` it is ±27 %, covering 1.0–2.0 MHz with at least a fifth of the
/// peak amplitude at either edge. The pulse is still 12 cells wide, so the grid
/// resolves it comfortably.
const PULSE_WIDTH_M: f64 = 0.6 * C0 / PULSE_CENTRE_HZ;
/// Half-width of the rectangular analysis gate, in steps.
const GATE_HALF_STEPS: usize = 150;

/// Frequencies at which the spectral ratio is evaluated, inside the packet's
/// usable band.
const ANALYSIS_FREQUENCIES: [f64; 5] = [1.0e6, 1.25e6, 1.5e6, 1.75e6, 2.0e6];

fn time_step() -> f64 {
    0.4 * DX / C0
}

fn medium(alpha0_db: f64, gamma: f64, grid: &Grid) -> HomogeneousMedium {
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, grid);
    medium
        .set_acoustic_properties(alpha0_db, gamma, 0.0)
        .expect("valid acoustic properties");
    medium
}

/// A Gaussian-modulated wave packet centred on `SOURCE_INDEX`.
///
/// Paired with `u = p/(ρc)` by the callers so the packet travels **rightward
/// only**; see the geometry note above for why the left-going half cannot be
/// tolerated. Both solvers receive the identical field.
fn launch_pulse(grid: &Grid) -> leto::Array3<f64> {
    let centre = SOURCE_INDEX as f64 * DX;
    let k_c = TAU * PULSE_CENTRE_HZ / C0;
    leto::Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |[i, _, _]| {
        let offset = i as f64 * DX - centre;
        (-(offset / PULSE_WIDTH_M).powi(2)).exp() * (k_c * offset).cos()
    })
}

/// Single-frequency DFT magnitude of the direct arrival at a sensor.
///
/// The gate is **rectangular**. The packet decays to zero well inside it, so
/// there is no truncation to taper away, and a taper would instead weight the
/// dispersively broadened far-sensor packet differently from the near one — an
/// 8–19 % bias in the recovered `α` when that was last measured (KW-SOL-072).
fn gated_magnitude(trace: &[f64], sensor_index: usize, frequency_hz: f64, dt: f64) -> f64 {
    let arrival = ((sensor_index - SOURCE_INDEX) as f64 * DX / C0 / dt).round() as usize;
    let lo = arrival.saturating_sub(GATE_HALF_STEPS);
    let hi = (arrival + GATE_HALF_STEPS).min(trace.len());

    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for (offset, &value) in trace[lo..hi].iter().enumerate() {
        let phase = TAU * frequency_hz * (lo + offset) as f64 * dt;
        re += value * phase.cos();
        im -= value * phase.sin();
    }
    re.hypot(im)
}

/// The launch as a `GridSource`, so **both** solvers apply it through their own
/// initial-condition handling.
///
/// Writing `fields.ux` directly does not work for the staggered FDTD: velocity
/// lives on the faces and at half time steps, so a cell-centred, whole-step
/// assignment is not the same field. Doing that left the packet failing to
/// propagate at all — the far sensor saw exactly zero while PSTD, which went
/// through `apply_initial_conditions`, propagated perfectly. Letting each solver
/// place its own initial conditions keeps the two launches equivalent.
fn launch_source(grid: &Grid) -> GridSource {
    let pulse = launch_pulse(grid);
    let mut p0 = leto::Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut ux0 = leto::Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..N {
        for j in 0..TRANSVERSE {
            for k in 0..TRANSVERSE {
                p0[[i, j, k]] = pulse[[i, 0, 0]];
                // Plane-wave relation for a rightward-travelling packet.
                ux0[[i, j, k]] = pulse[[i, 0, 0]] / (RHO0 * C0);
            }
        }
    }
    GridSource {
        p0: Some(p0),
        u0: Some((
            ux0,
            leto::Array3::zeros((grid.nx, grid.ny, grid.nz)),
            leto::Array3::zeros((grid.nx, grid.ny, grid.nz)),
        )),
        ..GridSource::new_empty()
    }
}

/// The two sensor traces from one run.
type Traces = (Vec<f64>, Vec<f64>);

fn run_fdtd(alpha0_db: f64, gamma: f64, dt: f64) -> Traces {
    let grid = Grid::new(N, TRANSVERSE, TRANSVERSE, DX, DX, DX).expect("grid");
    let medium = medium(alpha0_db, gamma, &grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        dt,
        nt: STEPS + 1,
        absorption: FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz: F_REF,
            band_min_hz: 0.4e6,
            band_max_hz: 3.0e6,
            relaxation_arms: 4,
        },
        ..Default::default()
    };
    let mut solver =
        FdtdSolver::new(config, &grid, &medium, launch_source(&grid)).expect("fdtd solver");
    assert_eq!(
        solver.absorption.is_some(),
        alpha0_db > 0.0,
        "absorption state presence must track the medium's alpha_0"
    );

    let mut near = Vec::with_capacity(STEPS);
    let mut far = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        solver.step_forward().expect("fdtd step");
        near.push(solver.fields.p[[SENSOR_NEAR, 0, 0]]);
        far.push(solver.fields.p[[SENSOR_FAR, 0, 0]]);
    }
    (near, far)
}

fn run_pstd(alpha0_db: f64, gamma: f64, dt: f64) -> Traces {
    let grid = Grid::new(N, TRANSVERSE, TRANSVERSE, DX, DX, DX).expect("grid");
    let medium = medium(alpha0_db, gamma, &grid);

    let config = PSTDConfig {
        dt,
        nt: STEPS + 1,
        boundary: BoundaryConfig::None,
        kspace_method: KSpaceMethod::StandardPSTD,
        absorption_mode: if alpha0_db > 0.0 {
            AbsorptionMode::PowerLaw {
                alpha_coeff: Some(alpha0_db),
                alpha_power: gamma,
            }
        } else {
            AbsorptionMode::Lossless
        },
        ..PSTDConfig::default()
    };
    let mut solver =
        PSTDSolver::new(config, grid.clone(), &medium, launch_source(&grid)).expect("pstd solver");

    let mut near = Vec::with_capacity(STEPS);
    let mut far = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        solver.step_forward().expect("pstd step");
        near.push(solver.fields.p[[SENSOR_NEAR, 0, 0]]);
        far.push(solver.fields.p[[SENSOR_FAR, 0, 0]]);
    }
    (near, far)
}

/// Recover `α(f)` by the **reference-normalized** two-sensor spectral ratio,
///
/// ```text
///   α(f) = −ln[ (P_far/P_near) / (P_far^ref/P_near^ref) ] / d
/// ```
///
/// against the identical run in a lossless medium. The raw ratio carries the
/// pulse spectrum, the gate transfer function and the solver's own frequency
/// response, none of which is absorption; dividing them out against a lossless
/// reference is the standard insertion-loss measurement. Only the medium
/// differs between the two runs, so what survives is the absorption alone.
fn measure_alpha(run: &Traces, reference: &Traces, dt: f64) -> Vec<(f64, f64)> {
    let separation = (SENSOR_FAR - SENSOR_NEAR) as f64 * DX;
    ANALYSIS_FREQUENCIES
        .iter()
        .map(|&f| {
            let near = gated_magnitude(&run.0, SENSOR_NEAR, f, dt);
            let far = gated_magnitude(&run.1, SENSOR_FAR, f, dt);
            let near_ref = gated_magnitude(&reference.0, SENSOR_NEAR, f, dt);
            let far_ref = gated_magnitude(&reference.1, SENSOR_FAR, f, dt);
            assert!(
                near > 0.0 && far > 0.0 && near_ref > 0.0 && far_ref > 0.0,
                "no spectral energy at {f:e} Hz"
            );
            let ratio = (far / near) / (far_ref / near_ref);
            (f, -ratio.ln() / separation)
        })
        .collect()
}

/// **The two paths agree, and both agree with the law.**
///
/// The pulse never reaches a boundary inside the analysis gate, so neither
/// solver's wall enters the comparison. That is what this measurement buys over
/// the standing wave it replaced, whose agreement was capped at 9 % because a
/// standing wave *is* its boundaries (KW-SOL-084).
///
/// Measured across the band, against the prescribed law:
///
/// | | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 MHz |
/// |---|---|---|---|---|---|
/// | fdtd, γ=1.1 | +0.6 | +0.7 | +0.8 | +0.9 | +0.9 % |
/// | pstd, γ=1.1 | −2.0 | −2.0 | −2.1 | −2.2 | −2.3 % |
/// | fdtd, γ=1.4 | +1.7 | +1.9 | +2.0 | +2.1 | +2.2 % |
/// | pstd, γ=1.4 | −0.9 | −1.0 | −1.1 | −1.2 | −1.3 % |
///
/// The error is near-constant in percent across the band, which is the useful
/// part: the frequency *exponent* is right in both paths and what remains is a
/// small multiplicative offset, opposite in sign between them. The bounds below
/// are bisected onto those numbers — 3 % against the law, 4 % between the paths
/// — rather than assumed.
#[test]
fn fdtd_relaxation_and_pstd_fractional_laplacian_agree() {
    let dt = time_step();

    // `α₀ = 0` is lossless whatever the exponent, so one reference per solver
    // serves every case rather than one per case.
    let fdtd_reference = run_fdtd(0.0, 1.1, dt);
    let pstd_reference = run_pstd(0.0, 1.1, dt);

    for &(alpha0_db, gamma) in &[(0.5_f64, 1.1_f64), (0.75, 1.4)] {
        let fdtd = measure_alpha(&run_fdtd(alpha0_db, gamma, dt), &fdtd_reference, dt);
        let pstd = measure_alpha(&run_pstd(alpha0_db, gamma, dt), &pstd_reference, dt);

        for ((frequency, alpha_fdtd), (_, alpha_pstd)) in fdtd.iter().zip(&pstd) {
            let prescribed = power_law_db_cm_to_np_m(alpha0_db, gamma, *frequency);

            for (name, measured) in [("fdtd", alpha_fdtd), ("pstd", alpha_pstd)] {
                assert!(
                    (measured - prescribed).abs() <= 0.03 * prescribed,
                    "α₀={alpha0_db}, γ={gamma} at {:.2} MHz: {name} measured {measured:.3} \
                     vs prescribed {prescribed:.3} Np/m",
                    frequency / 1.0e6
                );
            }

            let scale = alpha_fdtd.abs().max(alpha_pstd.abs());
            assert!(
                (alpha_fdtd - alpha_pstd).abs() <= 0.04 * scale,
                "α₀={alpha0_db}, γ={gamma} at {:.2} MHz: the two absorption paths disagree — \
                 fdtd {alpha_fdtd:.3} vs pstd {alpha_pstd:.3} Np/m",
                frequency / 1.0e6
            );
        }
    }
}

/// Both paths reduce to the lossless case together. Guards against a path that
/// absorbs unconditionally, which would still pass the comparison above by
/// agreeing with the other.
///
/// A lossless run measured against a lossless reference is the same run twice,
/// so the recovered `α` is exactly the measurement's own noise floor — which is
/// what makes this a meaningful bound on it.
#[test]
fn both_paths_are_lossless_without_absorption() {
    let dt = time_step();
    let gamma = 1.1;

    let fdtd = measure_alpha(&run_fdtd(0.0, gamma, dt), &run_fdtd(0.0, gamma, dt), dt);
    let pstd = measure_alpha(&run_pstd(0.0, gamma, dt), &run_pstd(0.0, gamma, dt), dt);

    // Reference scale: what a modestly absorbing medium produces at 1 MHz.
    let absorbing = power_law_db_cm_to_np_m(0.5, gamma, 1.0e6);
    assert!(absorbing > 0.0);

    for (name, measured) in [("fdtd", &fdtd), ("pstd", &pstd)] {
        for (frequency, alpha) in measured {
            assert!(
                alpha.abs() < 0.01 * absorbing,
                "{name} at {:.2} MHz: lossless run recovered {alpha:.4e} Np/m",
                frequency / 1.0e6
            );
        }
    }
}
