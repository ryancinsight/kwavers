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
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
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
/// Half-wavelengths across the grid. `m = 8` on 32 cells is 8 points per
/// wavelength, about 1.9 MHz at 1500 m/s.
const MODE: f64 = 8.0;
const STEPS: usize = 1200;

/// Wavenumber of the exact discrete Dirichlet eigenmode.
///
/// Zero-extension puts the walls at cell centres `-1` and `N`, so the effective
/// domain is `(N+1)Δx` and the eigenvectors of the composite operator are
/// `sin(πm(i+1)/(N+1))` — the classic Dirichlet tridiagonal eigenvectors. Using
/// `N` instead of `N+1` leaves a small mode mixture that shows up directly as a
/// biased decay estimate.
fn wavenumber() -> f64 {
    std::f64::consts::PI * MODE / ((N as f64 + 1.0) * DX)
}

fn medium(alpha0_db: f64, gamma: f64, grid: &Grid) -> HomogeneousMedium {
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, grid);
    medium
        .set_acoustic_properties(alpha0_db, gamma, 0.0)
        .expect("valid acoustic properties");
    medium
}

/// The exact discrete eigenmode of the zero-extension domain.
///
/// Sine, not cosine: the solvers close their domain with `p = 0` outside, so a
/// cosine leaves a pressure step at the wall and excites a spread of modes,
/// which the single-wavenumber analysis below (dividing the decay by `ω/k`)
/// would then be measuring a mixture of.
fn standing_wave(grid: &Grid) -> leto::Array3<f64> {
    let k0 = wavenumber();
    leto::Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |[i, _, _]| {
        (k0 * (i as f64 + 1.0) * DX).sin()
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

/// Spatial attenuation `α` \[Np·m⁻¹]: the temporal decay divided by each
/// solver's **own** measured phase speed.
///
/// The normalization is load-bearing, not cosmetic. The two solvers close their
/// domains differently — FDTD by zero-extension since KW-SOL-074, PSTD
/// periodically — so the same initial condition is a different mode in each and
/// evolves at a different frequency. Measured here, FDTD runs about 23 % faster
/// than PSTD, which shows up in the raw temporal rates as a ratio of `r^γ`
/// (1.26 at γ = 1.1, 1.32 at γ = 1.4, both giving r ≈ 1.23). Dividing each by
/// its own `ω/k` removes exactly that, leaving the absorption.
///
/// The energy is averaged over a window at each end rather than sampled at two
/// instants: a standing wave exchanges energy between pressure and velocity
/// every half period, so a two-point ratio measures where in that cycle the
/// samples landed.
fn decay_rate(energy: &[f64], trace: &[f64], dt: f64) -> f64 {
    let span = energy.len();
    assert!(
        span > 2 * WINDOW,
        "history must exceed both averaging windows"
    );
    let mean = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
    let early = mean(&energy[..WINDOW]);
    let late = mean(&energy[span - WINDOW..]);
    // Window centres are `span - WINDOW` steps apart.
    let decay = -(late / early).ln() / (2.0 * (span - WINDOW) as f64 * dt);

    let crossings = trace.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64;
    let omega = TAU * crossings / 2.0 / (span as f64 * dt);
    assert!(omega > 0.0, "no oscillation observed in the history");
    decay / (omega / wavenumber())
}

/// Run the FDTD relaxation path and return its realized `α`.
fn run_fdtd(alpha0_db: f64, gamma: f64, dt: f64) -> f64 {
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
        trace.push(solver.fields.p[[N / 2, 0, 0]]);
    }
    decay_rate(&energy, &trace, dt)
}

/// Run the PSTD fractional-Laplacian path and return its realized `α`.
fn run_pstd(alpha0_db: f64, gamma: f64, dt: f64) -> f64 {
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
        trace.push(solver.fields.p[[N / 2, 0, 0]]);
    }
    decay_rate(&energy, &trace, dt)
}

/// **The two paths agree.**
///
/// The independent-oracle comparison: a spectral frequency-domain operator and
/// a local time-domain ODE system, driven identically, must absorb at the same
/// rate. Neither is compared to the law here — each has its own test for that —
/// so this asserts the thing no other test can: that two different mathematics
/// for one physical law land in the same place.
///
/// **Measured agreement is 9.1 %** (22.30 against 20.27 Np·m⁻¹ at the worst
/// point, α₀ = 0.75, γ = 1.4), so the 15 % bound carries margin while staying
/// far tighter than the factor-of-several a broken path gives.
///
/// That is a real loosening from the 1.3 % this test reported before
/// KW-SOL-074. The cause is not a regression in either path but in the
/// comparison: FDTD now closes its domain by zero-extension while PSTD remains
/// periodic, so one initial condition is a different mode in each and the
/// per-solver frequency normalization only partly compensates. A
/// boundary-independent setup — a travelling pulse measured by spectral ratio,
/// as `heterogeneous_power_law_attenuation` does — would restore the tighter
/// figure. Tracked as KW-SOL-084.
#[test]
fn fdtd_relaxation_and_pstd_fractional_laplacian_agree() {
    // Well inside both stability limits.
    let dt = 0.15 * DX / C0;

    for &(alpha0_db, gamma) in &[(0.5_f64, 1.1_f64), (0.75, 1.4)] {
        let fdtd = run_fdtd(alpha0_db, gamma, dt);
        let pstd = run_pstd(alpha0_db, gamma, dt);

        assert!(
            fdtd > 0.0 && pstd > 0.0,
            "α₀={alpha0_db}, γ={gamma}: both paths must absorb (fdtd {fdtd:.4e}, pstd {pstd:.4e})"
        );

        // Both must also scale with α₀ and γ rather than being a fixed damping;
        // that is what makes agreement meaningful rather than coincidental.
        let scale = fdtd.max(pstd);
        assert!(
            (fdtd - pstd).abs() <= 0.15 * scale,
            "α₀={alpha0_db}, γ={gamma}: the two absorption paths disagree — \
             fdtd {fdtd:.4} vs pstd {pstd:.4} Np/m"
        );
    }

    // Doubling α₀ must roughly double both rates: a path that absorbed a fixed
    // amount would agree with the other above yet fail here.
    let weak = run_fdtd(0.375, 1.1, dt);
    let strong = run_fdtd(0.75, 1.1, dt);
    assert!(
        strong > 1.7 * weak,
        "FDTD decay does not track α₀: {weak:.4e} then {strong:.4e}"
    );
    let weak = run_pstd(0.375, 1.1, dt);
    let strong = run_pstd(0.75, 1.1, dt);
    assert!(
        strong > 1.7 * weak,
        "PSTD decay does not track α₀: {weak:.4e} then {strong:.4e}"
    );
}

/// Both paths reduce to the lossless case together: a medium with zero `α₀`
/// must not decay on either. Guards against a path that absorbs
/// unconditionally, which would still pass the comparison above by agreeing
/// with the other.
#[test]
fn both_paths_are_lossless_without_absorption() {
    let dt = 0.15 * DX / C0;
    let fdtd = run_fdtd(0.0, 1.1, dt);
    let pstd = run_pstd(0.0, 1.1, dt);

    // Reference: the rate a modestly absorbing medium produces. A lossless run
    // must sit far below it — what remains is the leapfrog's bounded energy
    // oscillation, which has no preferred sign.
    let absorbing = run_fdtd(0.5, 1.1, dt);
    assert!(absorbing > 0.0);
    assert!(
        fdtd.abs() < 0.05 * absorbing,
        "lossless FDTD decayed at {fdtd:.4e} Np/m against an absorbing {absorbing:.4e}"
    );
    assert!(
        pstd.abs() < 0.05 * absorbing,
        "lossless PSTD decayed at {pstd:.4e} Np/m against an absorbing {absorbing:.4e}"
    );
}
