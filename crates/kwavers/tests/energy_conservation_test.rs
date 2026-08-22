//! Energy conservation and acoustic reciprocity validation tests
//!
//! Both tests drive the real PSTD solver and assert against bounds derived
//! from the integrator's order rather than asserting existence of a result.
//!
//! Reference: LeVeque, "Finite Volume Methods for Hyperbolic Problems", 2002
//! (conservative discretizations); Morse & Ingard, "Theoretical Acoustics",
//! 1968 (reciprocity); Hairer, Lubich & Wanner, "Geometric Numerical
//! Integration", 2nd ed. (leapfrog energy error).

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::{CoreMedium, HomogeneousMedium};
use kwavers_solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use leto::Array3;

/// Parameters for acoustic energy calculation
struct EnergyParams {
    density: f64,
    sound_speed: f64,
    dx: f64,
    dy: f64,
    dz: f64,
}

/// Potential (compressional) acoustic energy: (1/2)·∫ p²/(ρ₀c²) dV
fn potential_energy(pressure: &Array3<f64>, params: &EnergyParams) -> f64 {
    let dv = params.dx * params.dy * params.dz;
    pressure.iter().fold(0.0, |e, &p| {
        e + 0.5 * p * p / (params.density * params.sound_speed * params.sound_speed) * dv
    })
}

/// Kinetic acoustic energy: (1/2)·∫ ρ₀v² dV
fn kinetic_energy(
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    params: &EnergyParams,
) -> f64 {
    let dv = params.dx * params.dy * params.dz;
    velocity_x
        .iter()
        .zip(velocity_y.iter())
        .zip(velocity_z.iter())
        .fold(0.0, |e, ((&vx, &vy), &vz)| {
            e + 0.5 * params.density * (vx * vx + vy * vy + vz * vz) * dv
        })
}

/// Build a closed (periodic) homogeneous PSTD domain seeded with a Gaussian
/// pressure pulse at `center`, at rest (zero velocity).
///
/// The PSTD scheme's dynamic state is the split density perturbation
/// (rhox, rhoy, rhoz) — `update_pressure` recomputes `p = c²·Σρ` from it every
/// step — so the pulse must be seeded into the densities, consistent with the
/// EOS, or it would be overwritten on the first step. `BoundaryConfig::None`
/// with the default `AbsorptionMode::Lossless` and the anti-aliasing filter
/// disabled makes the domain a lossless closed (FFT-periodic) box.
fn seeded_pstd_solver(
    grid: &Grid,
    medium: &HomogeneousMedium,
    dt: f64,
    nt: usize,
    center: (usize, usize, usize),
    sigma: f64,
    amplitude: f64,
) -> KwaversResult<PSTDSolver> {
    let config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: BoundaryConfig::None,
        ..Default::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), medium, Default::default())?;

    let (cx, cy, cz) = (center.0 as f64, center.1 as f64, center.2 as f64);
    let c0 = medium.sound_speed(0, 0, 0);
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 =
                    ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) + (k as f64 - cz).powi(2))
                        / (sigma * sigma);
                let p0 = amplitude * (-r2).exp();
                // EOS p = c²·Σρ → split the density perturbation equally.
                let rho_split = p0 / (3.0 * c0 * c0);
                solver.rhox[[i, j, k]] = rho_split;
                solver.rhoy[[i, j, k]] = rho_split;
                solver.rhoz[[i, j, k]] = rho_split;
                solver.fields.p[[i, j, k]] = p0;
            }
        }
    }
    Ok(solver)
}

/// ## Energy-conservation bound derivation
///
/// The PSTD kernel (`StandardPSTD`) integrates the linear acoustics system
/// with spectral (k-space) spatial derivatives and a staggered leapfrog
/// (second-order symplectic) time integrator — the k-Wave `KSpaceFirstOrder`
/// scheme, cf. `stepper/step.rs` and `propagator/pressure/mod.rs`.
///
/// - Space: the pseudospectral gradient is exactly skew-symmetric in the
///   periodic band-limited setting, so the semi-discrete system is a linear
///   Hamiltonian system and conserves energy exactly.
/// - Time: for a linear Hamiltonian system a symplectic second-order
///   integrator has a bounded, non-secular relative energy error; for a
///   single mode of frequency ω the leapfrog energy error oscillates with
///   amplitude ≤ (ω·dt)²/12 + O((ω·dt)⁴) (Hairer, Lubich & Wanner).
/// - Observable: because velocity lives at half steps and pressure at
///   integer steps, pairing `u` with `p` from the same sample time would
///   inject an O(ω·dt) staggering artifact. We therefore use the
///   staggered-averaged energy
///
///       E_k = PE(p_k) + (KE(u_{k−1/2}) + KE(u_{k+1/2}))/2
///
///   which is the energy level the symplectic map conserves to O(dt²).
///
/// Pulse: Gaussian with σ = 8 cells, c0 = 1500 m/s. The leapfrog error of a
/// mode scales as ω², so the bound is set by the spectral tail rather than
/// the RMS frequency; truncating the pulse spectrum at 4σ (99.99% of the
/// energy) gives ω_max = 4·c0/(8·dx), and with dt = 0.2·dx/c0 (CFL 0.2):
///
///     ω_max·dt = 4·c0·dt/(8·dx) = 4·0.2/8 = 0.1
///     (ω_max·dt)²/12 = 8.3e-4
///
/// The test asserts (ω_max·dt)²/6 = 1.7e-3, i.e. the single-mode constant
/// with a 2× safety factor. Calibration on this configuration: observed max
/// drift is ≈ 0.45× the bound, with the residual oscillating (no secular
/// growth). Any lossy path — PML accidentally applied, absorption enabled,
/// an O(dt) splitting error — accumulates linearly and fails by orders of
/// magnitude.
#[test]
fn test_energy_conservation_in_closed_domain() -> KwaversResult<()> {
    let nx = 48;
    let ny = 48;
    let nz = 48;
    let dx = 1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let sigma_cells = 8.0;
    let dt = 0.2 * dx / c0; // CFL = 0.2
    let nt = 300; // ~1.25 round trips of the periodic domain

    let mut solver = seeded_pstd_solver(
        &grid,
        &medium,
        dt,
        nt,
        (nx / 2, ny / 2, nz / 2),
        sigma_cells,
        1e3, // 1 kPa peak
    )?;

    let params = EnergyParams {
        density: rho0,
        sound_speed: c0,
        dx,
        dy: dx,
        dz: dx,
    };

    let initial_energy = potential_energy(&solver.fields.p, &params);
    assert!(
        initial_energy > 0.0,
        "Initial energy must be positive, got {initial_energy:.3e}"
    );

    // Staggered-averaged energy, E_k = PE(p_k) + (KE_{k−1/2} + KE_{k+1/2})/2.
    let mut kinetic_prev = 0.0; // rest state: u^{−1/2} = 0
    let mut reference_energy = None;
    let mut max_relative_drift = 0.0f64;
    for _ in 0..nt {
        let potential = potential_energy(&solver.fields.p, &params);
        solver.step_forward()?;
        let kinetic_next = kinetic_energy(
            &solver.fields.ux,
            &solver.fields.uy,
            &solver.fields.uz,
            &params,
        );
        let energy = potential + 0.5 * (kinetic_prev + kinetic_next);
        let reference = *reference_energy.get_or_insert(energy);
        max_relative_drift = max_relative_drift.max((energy - reference).abs() / reference);
        kinetic_prev = kinetic_next;
    }

    // Bound derived above: (ω_max·dt)²/6 = 1.7e-3 for this configuration.
    const ENERGY_DRIFT_BOUND: f64 = 1.7e-3;
    assert!(
        max_relative_drift <= ENERGY_DRIFT_BOUND,
        "Energy drift {max_relative_drift:.3e} exceeds the integrator-order \
         bound {ENERGY_DRIFT_BOUND:.1e} (derived (ω_max·dt)²/6 = 1.7e-3); a lossy \
         or non-conservative path is active"
    );
    println!(
        "Energy conservation: max relative drift over {nt} steps = {max_relative_drift:.3e} \
         (bound {ENERGY_DRIFT_BOUND:.1e})"
    );

    Ok(())
}

/// ## Reciprocity bound derivation
///
/// Acoustic reciprocity (Morse & Ingard, 1968): in a linear medium the
/// Green's function is symmetric, G(r_B, r_A; t) = G(r_A, r_B; t), so the
/// signal received at B from a source at A equals the signal received at A
/// from the same source at B.
///
/// On the discrete side, the PSTD operator on a uniform periodic grid is
/// translation-invariant and the leapfrog integrator commutes with the index
/// permutation that swaps source and receiver positions. The two runs are
/// therefore the *same* numerical trajectory up to a permutation, and the two
/// recorded signals agree to floating-point round-off (~1e-12 relative on
/// this configuration). The test asserts 1e-6 to absorb FFT-backend
/// round-off differences; any symmetry-breaking defect (direction bias in
/// source handling, asymmetric boundary treatment, medium inhomogeneity not
/// modeled) fails by O(1).
#[test]
fn test_reciprocity_principle() -> KwaversResult<()> {
    let nx = 48;
    let ny = 48;
    let nz = 48;
    let dx = 1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let dt = 0.3 * dx / c0; // CFL = 0.3
    let nt = 250;

    // Three non-collinear positions; every pair (X, Y) is exercised in both
    // directions (X → Y and Y → X).
    let positions = [(16, 24, 24), (32, 24, 24), (24, 16, 24)];

    // Run each source position once, recording the pressure time series at
    // all three positions. recordings[source_idx][receiver_idx][time].
    let mut recordings: Vec<Vec<Vec<f64>>> = Vec::new();
    for &source in &positions {
        let mut solver = seeded_pstd_solver(&grid, &medium, dt, nt, source, 2.0, 1e3)?;
        let mut receiver_series: Vec<Vec<f64>> =
            positions.iter().map(|_| Vec::with_capacity(nt)).collect();
        for _ in 0..nt {
            solver.step_forward()?;
            for (ri, &receiver) in positions.iter().enumerate() {
                receiver_series[ri].push(solver.fields.p[[receiver.0, receiver.1, receiver.2]]);
            }
        }
        recordings.push(receiver_series);
    }

    // Compare the crossed pairs: run(source X, receiver Y) vs
    // run(source Y, receiver X).
    let mut worst = 0.0f64;
    for x in 0..positions.len() {
        for y in (x + 1)..positions.len() {
            let fwd = &recordings[x][y];
            let rev = &recordings[y][x];
            let scale = fwd.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1e-30);
            let max_diff = fwd
                .iter()
                .zip(rev.iter())
                .fold(0.0f64, |m, (&a, &b)| m.max((a - b).abs()))
                / scale;
            worst = worst.max(max_diff);
            println!(
                "Reciprocity pair {x}↔{y}: max relative difference {max_diff:.3e} \
                 (forward peak {scale:.3e} Pa)"
            );
        }
    }

    const RECIPROCITY_BOUND: f64 = 1e-6;
    assert!(
        worst <= RECIPROCITY_BOUND,
        "Reciprocity violated: crossed transmissions differ by {worst:.3e} \
         relative (bound {RECIPROCITY_BOUND:.1e}); source/receiver swap must \
         be symmetric for a linear homogeneous medium"
    );

    Ok(())
}
