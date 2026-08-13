//! Differential verification across the two heterogeneous-absorption paths.
//!
//! kwavers realizes power-law absorption two ways, and until now each was
//! checked only against its own analytic target — never against the other:
//!
//! | path | mechanism | spatial operator |
//! |---|---|---|
//! | PSTD | Treeby–Cox fractional Laplacian, `IFFT(\|k\|^(y−s)·FFT(·))` | spectral |
//! | FDTD | relaxation memory variables on a fitted `τ` grid | finite difference |
//!
//! These are genuinely different mathematics for the same physical law — a
//! frequency-domain operator applied spectrally versus a time-domain ODE system
//! integrated locally — so agreement between them is
//! **independent-oracle** evidence, the strongest tier available here. Two
//! backends running the same algorithm would only be differential evidence.
//!
//! The quantity compared is the realized amplitude decay of a standing wave,
//! which is what both paths exist to produce. Each is also compared against the
//! prescribed `α₀·(f/f_ref)^γ`, so a *shared* error would still surface rather
//! than cancelling between them.
//!
//! # A capability difference the comparison exposes
//!
//! The exponents here avoid `γ = 1.0` because **PSTD cannot represent it**: the
//! Treeby–Cox dispersion coefficient is `η = 2α₀c₀^y·tan(πy/2)`, which diverges
//! at `y = 1`, and the solver rejects that configuration outright. The
//! relaxation path has no such singularity — `γ = 1` is an ordinary point for a
//! fitted spectrum, and the FDTD and viscoacoustic tests exercise it directly.
//! So the two paths are not interchangeable even where both apply: linear
//! frequency dependence, the textbook soft-tissue idealization, is available
//! only through relaxation.

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

const N: usize = 32;
const DX: f64 = 1.0e-4;
const C0: f64 = SOUND_SPEED_WATER_SIM;
const RHO0: f64 = DENSITY_WATER_NOMINAL;
const F_REF: f64 = 1.0e6;
/// Mode 4 of 32 cells: 8 points per wavelength, about 1.9 MHz at 1500 m/s.
const MODE: f64 = 4.0;
const STEPS: usize = 1200;

fn wavenumber() -> f64 {
    TAU * MODE / (N as f64 * DX)
}

fn medium(alpha0_db: f64, gamma: f64, grid: &Grid) -> HomogeneousMedium {
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, grid);
    medium
        .set_acoustic_properties(alpha0_db, gamma, 0.0)
        .expect("valid acoustic properties");
    medium
}

/// `cos(k·x)`, uniform across the transverse extent.
fn standing_wave(grid: &Grid) -> leto::Array3<f64> {
    let k0 = wavenumber();
    leto::Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |[i, _, _]| {
        (k0 * i as f64 * DX).cos()
    })
}

/// Steps averaged at each end of the history.
const WINDOW: usize = 200;

/// Total discrete acoustic energy, `p²/(2ρc²) + ρ|u|²/2`.
///
/// The **total** rather than the pressure alone: a standing wave moves its
/// energy between the two forms twice per period, so a pressure-only measure
/// swings by a factor of five here even averaged over several periods (the
/// window spans a non-integer number of cycles) and buries the decay being
/// measured. The sum is smooth, and in the lossless limit it is the conserved
/// quantity.
fn total_energy(
    pressure: &leto::Array3<f64>,
    ux: &leto::Array3<f64>,
    uy: &leto::Array3<f64>,
    uz: &leto::Array3<f64>,
) -> f64 {
    let bulk = RHO0 * C0 * C0;
    let potential: f64 = pressure.iter().map(|&p| p * p / (2.0 * bulk)).sum();
    let kinetic: f64 = ux
        .iter()
        .zip(uy.iter())
        .zip(uz.iter())
        .map(|((&a, &b), &c)| 0.5 * RHO0 * (a * a + b * b + c * c))
        .sum();
    potential + kinetic
}

/// Spatial attenuation `α` \[Np·m⁻¹] and angular frequency from a history.
///
/// The temporal decay is divided by the *measured* phase speed `ω/k`, not the
/// nominal `c₀`, so numerical dispersion in either solver does not bias the
/// comparison — and so both the absorbing and lossless checks are in the same
/// units as the prescribed law.
///
/// The energy is averaged over a window at each end rather than sampled at two
/// instants. A standing wave exchanges energy between pressure and velocity
/// every half period, so a two-point ratio measures where in that cycle the
/// samples landed, not the decay -- for a lossless run it reports a large
/// spurious rate with an arbitrary sign. Averaging over several periods cancels
/// the exchange and leaves the envelope.
fn decay_and_frequency(energy: &[f64], trace: &[f64], dt: f64) -> (f64, f64) {
    let span = energy.len();
    assert!(
        span > 2 * WINDOW,
        "history must exceed both averaging windows"
    );
    let mean = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
    let early = mean(&energy[..WINDOW]);
    let late = mean(&energy[span - WINDOW..]);
    // Window centres are `span - WINDOW` steps apart.
    let separation = (span - WINDOW) as f64 * dt;
    let decay = -(late / early).ln() / (2.0 * separation);

    let elapsed = span as f64 * dt;
    let crossings = trace.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64;
    let omega = TAU * crossings / 2.0 / elapsed;
    assert!(omega > 0.0, "no oscillation observed in the history");
    let phase_speed = omega / wavenumber();
    (decay / phase_speed, omega)
}

/// Run the FDTD relaxation path and return `(α, ω)`.
fn run_fdtd(alpha0_db: f64, gamma: f64, dt: f64) -> (f64, f64) {
    let grid = Grid::new(N, 4, 4, DX, DX, DX).expect("grid");
    let medium = medium(alpha0_db, gamma, &grid);
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        dt,
        nt: STEPS + 1,
        absorption: FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz: F_REF,
            band_min_hz: 0.2e6,
            band_max_hz: 4.0e6,
            relaxation_arms: 4,
        },
        ..Default::default()
    };
    let mut solver =
        FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).expect("fdtd solver");
    // A lossy configuration must actually allocate the relaxation state; a
    // lossless medium correctly skips it, so the check is conditional rather
    // than unconditional.
    assert_eq!(
        solver.absorption.is_some(),
        alpha0_db > 0.0,
        "absorption state presence must track the medium's alpha_0"
    );

    let initial = standing_wave(&grid);
    for i in 0..N {
        for j in 0..4 {
            for k in 0..4 {
                solver.fields.p[[i, j, k]] = initial[[i, 0, 0]];
            }
        }
    }

    let mut energy = Vec::with_capacity(STEPS);
    let mut trace = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        solver.step_forward().expect("fdtd step");
        energy.push(total_energy(
            &solver.fields.p,
            &solver.fields.ux,
            &solver.fields.uy,
            &solver.fields.uz,
        ));
        trace.push(solver.fields.p[[0, 0, 0]]);
    }
    decay_and_frequency(&energy, &trace, dt)
}

/// Run the PSTD fractional-Laplacian path and return `(α, ω)`.
fn run_pstd(alpha0_db: f64, gamma: f64, dt: f64) -> (f64, f64) {
    let grid = Grid::new(N, 4, 4, DX, DX, DX).expect("grid");
    let medium = medium(alpha0_db, gamma, &grid);

    let mut p0 = leto::Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let initial = standing_wave(&grid);
    for i in 0..N {
        for j in 0..4 {
            for k in 0..4 {
                p0[[i, j, k]] = initial[[i, 0, 0]];
            }
        }
    }
    let source = GridSource {
        p0: Some(p0),
        ..GridSource::new_empty()
    };

    let config = PSTDConfig {
        dt,
        nt: STEPS + 1,
        boundary: BoundaryConfig::None,
        kspace_method: KSpaceMethod::StandardPSTD,
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: alpha0_db,
            alpha_power: gamma,
        },
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, source).expect("pstd solver");

    let mut energy = Vec::with_capacity(STEPS);
    let mut trace = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        solver.step_forward().expect("pstd step");
        energy.push(total_energy(
            &solver.fields.p,
            &solver.fields.ux,
            &solver.fields.uy,
            &solver.fields.uz,
        ));
        trace.push(solver.fields.p[[0, 0, 0]]);
    }
    decay_and_frequency(&energy, &trace, dt)
}

/// **The two paths agree, and both agree with the law.**
///
/// The 2 % bound is measured, not guessed. Tightened by bisection, the pair
/// holds at 2 % and breaks at 0.5 %: the worst single deviation is FDTD against
/// the law at `α₀ = 0.5, γ = 1.1` — 11.361 against 11.213 Np·m⁻¹, or 1.32 %.
/// That is consistent with what the parts contribute (the relaxation fit's own
/// sub-1 % bound at four arms, plus the leapfrog's `0.117·(ω·Δt)²`), and it is
/// two orders below the factor-of-several a genuinely broken path produces.
#[test]
fn fdtd_relaxation_and_pstd_fractional_laplacian_agree() {
    // Well inside both stability limits.
    let dt = 0.15 * DX / C0;

    for &(alpha0_db, gamma) in &[(0.5_f64, 1.1_f64), (0.75, 1.4)] {
        let (alpha_fdtd, omega_fdtd) = run_fdtd(alpha0_db, gamma, dt);
        let (alpha_pstd, omega_pstd) = run_pstd(alpha0_db, gamma, dt);

        assert!(
            alpha_fdtd > 0.0 && alpha_pstd > 0.0,
            "α₀={alpha0_db}, γ={gamma}: both paths must absorb \
             (fdtd {alpha_fdtd:.4e}, pstd {alpha_pstd:.4e})"
        );
        assert!(
            omega_fdtd > 0.0 && omega_pstd > 0.0,
            "α₀={alpha0_db}, γ={gamma}: both paths must oscillate"
        );

        let prescribed_fdtd = power_law_db_cm_to_np_m(alpha0_db, gamma, omega_fdtd / TAU);
        let prescribed_pstd = power_law_db_cm_to_np_m(alpha0_db, gamma, omega_pstd / TAU);

        // Each against the law it was configured with.
        for (name, measured, prescribed) in [
            ("fdtd", alpha_fdtd, prescribed_fdtd),
            ("pstd", alpha_pstd, prescribed_pstd),
        ] {
            assert!(
                (measured - prescribed).abs() <= 0.02 * prescribed,
                "α₀={alpha0_db}, γ={gamma}: {name} measured {measured:.3} vs prescribed \
                 {prescribed:.3} Np/m"
            );
        }

        // And against each other — the independent-oracle comparison. A shared
        // systematic error would survive the checks above but not this one if
        // only one path carried it.
        let scale = alpha_fdtd.max(alpha_pstd);
        assert!(
            (alpha_fdtd - alpha_pstd).abs() <= 0.02 * scale,
            "α₀={alpha0_db}, γ={gamma}: the two absorption paths disagree — \
             fdtd {alpha_fdtd:.3} vs pstd {alpha_pstd:.3} Np/m"
        );
    }
}

/// Both paths reduce to the lossless case together: a medium with zero `α₀`
/// must not decay on either. Guards against a path that absorbs unconditionally
/// (which would still pass the comparison above by agreeing with the other).
#[test]
fn both_paths_are_lossless_without_absorption() {
    let dt = 0.15 * DX / C0;
    let (alpha_fdtd, _) = run_fdtd(0.0, 1.1, dt);
    let (alpha_pstd, _) = run_pstd(0.0, 1.1, dt);

    // The lossless FDTD leapfrog conserves energy (KW-SOL-081), so its measured
    // decay is round-off. PSTD's is likewise bounded by its own conservation.
    let reference = power_law_db_cm_to_np_m(0.5, 1.0, 2.0e6);
    assert!(
        alpha_fdtd.abs() < 0.05 * reference,
        "lossless FDTD decayed at {alpha_fdtd:.4e} Np/m"
    );
    assert!(
        alpha_pstd.abs() < 0.05 * reference,
        "lossless PSTD decayed at {alpha_pstd:.4e} Np/m"
    );
}
