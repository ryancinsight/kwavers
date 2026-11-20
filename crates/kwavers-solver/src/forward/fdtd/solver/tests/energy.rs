use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;
use leto_ops::Axis;

/// **Lossless leapfrog conserves discrete energy (KW-SOL-081 regression).**
///
/// The Yee leapfrog is symplectic when the pressure update's divergence is the
/// negative adjoint of the velocity update's gradient. It then has a conserved
/// discrete energy
///
/// ```text
///   E = sum_i [ p_i^2 / (2 rho c^2) + rho |u_i|^2 / 2 ] dV
/// ```
///
/// which oscillates slightly (velocity sits at half steps, so the naive sum
/// samples a shadow Hamiltonian) but does not drift.
///
/// Before the adjoint closure this ran away: the divergence used a one-sided
/// difference at the low face instead of the zero-flux one, breaking
/// `D = -G^T`, and pressure energy alone grew by a factor of 8.8e4 over two
/// thousand steps. The bound below would fail by four orders of magnitude on
/// that scheme, so this test pins the fix rather than merely exercising it.
#[test]
fn lossless_staggered_leapfrog_conserves_energy() {
    use std::f64::consts::TAU;

    const N: usize = 64;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const STEPS: usize = 2000;

    let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
    // Lossless: the conserved quantity only exists without absorption.
    medium.set_acoustic_properties(0.0, 1.0, 0.0).unwrap();

    let dt = 0.15 * DX / C0;
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        dt,
        nt: STEPS + 1,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    let k0 = TAU * 8.0 / (N as f64 * DX);
    for i in 0..N {
        let value = (k0 * i as f64 * DX).cos();
        for j in 0..4 {
            for k in 0..4 {
                solver.fields.p[[i, j, k]] = value;
            }
        }
    }

    let energy = |solver: &FdtdSolver| -> f64 {
        let bulk = RHO0 * C0 * C0;
        let potential: f64 = solver.fields.p.iter().map(|&p| p * p / (2.0 * bulk)).sum();
        let kinetic: f64 = solver
            .fields
            .ux
            .iter()
            .zip(solver.fields.uy.iter())
            .zip(solver.fields.uz.iter())
            .map(|((&ux, &uy), &uz)| 0.5 * RHO0 * (ux * ux + uy * uy + uz * uz))
            .sum();
        potential + kinetic
    };

    let initial = energy(&solver);
    assert!(initial > 0.0);

    let mut lowest = f64::INFINITY;
    let mut highest = 0.0_f64;
    for _ in 0..STEPS {
        solver.step_forward().unwrap();
        let current = energy(&solver) / initial;
        lowest = lowest.min(current);
        highest = highest.max(current);
    }

    assert!(
        highest < 1.1 && lowest > 0.9,
        "discrete energy drifted to [{lowest:.4}, {highest:.4}] of its initial value \
         over {STEPS} steps; a symplectic leapfrog holds it near 1"
    );
}

/// The same conservation on the collocated (central-difference) branch, whose
/// one-sided boundary rows broke skew-symmetry the same way the staggered
/// divergence's did.
#[test]
fn lossless_collocated_leapfrog_conserves_energy() {
    use std::f64::consts::TAU;

    const N: usize = 64;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const STEPS: usize = 2000;

    let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
    medium.set_acoustic_properties(0.0, 1.0, 0.0).unwrap();

    let dt = 0.15 * DX / C0;
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: false,
        dt,
        nt: STEPS + 1,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    let k0 = TAU * 8.0 / (N as f64 * DX);
    for i in 0..N {
        let value = (k0 * i as f64 * DX).cos();
        for j in 0..4 {
            for k in 0..4 {
                solver.fields.p[[i, j, k]] = value;
            }
        }
    }

    let energy = |solver: &FdtdSolver| -> f64 {
        let bulk = RHO0 * C0 * C0;
        let potential: f64 = solver.fields.p.iter().map(|&p| p * p / (2.0 * bulk)).sum();
        let kinetic: f64 = solver
            .fields
            .ux
            .iter()
            .zip(solver.fields.uy.iter())
            .zip(solver.fields.uz.iter())
            .map(|((&ux, &uy), &uz)| 0.5 * RHO0 * (ux * ux + uy * uy + uz * uz))
            .sum();
        potential + kinetic
    };

    let initial = energy(&solver);
    assert!(initial > 0.0);
    let mut lowest = f64::INFINITY;
    let mut highest = 0.0_f64;
    for _ in 0..STEPS {
        solver.step_forward().unwrap();
        let current = energy(&solver) / initial;
        lowest = lowest.min(current);
        highest = highest.max(current);
    }

    assert!(
        highest < 1.1 && lowest > 0.9,
        "collocated discrete energy drifted to [{lowest:.4}, {highest:.4}] over {STEPS} \
         steps; the skew-symmetric closure holds it near 1 (the one-sided closure \
         reached 1.3e4)"
    );
}

/// **Eighth order runs, and every order conserves energy.**
///
/// Orders 4-8 became reachable on the staggered path when it moved onto
/// `leto_ops::StaggeredLeapfrog3D` (KW-SOL-074); eighth order is what Fullwave 2.5
/// uses. Conservation must hold at each, since the gradient/divergence pair is
/// adjoint at every order by construction.
#[test]
fn every_staggered_order_conserves_energy() {
    use std::f64::consts::TAU;

    const N: usize = 48;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const STEPS: usize = 1000;

    for order in [2usize, 4, 6, 8] {
        let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
        let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
        medium.set_acoustic_properties(0.0, 1.1, 0.0).unwrap();

        // Inside the derived staggered limit for every order tested.
        let dt = 0.1 * DX / C0;
        let config = FdtdConfig {
            spatial_order: order,
            staggered_grid: true,
            dt,
            nt: STEPS + 1,
            ..Default::default()
        };
        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

        // The exact discrete Neumann mode of the rigid-walled domain (ADR 106):
        // cell centres at `(i+½)Δ`, walls at `0` and `NΔ`, so `cos(k(i+½)Δ)`
        // with `k = mπ/(NΔ)` satisfies `∂p/∂n = 0` at both. It was a `sin` mode
        // while the walls were pressure-release.
        let k0 = std::f64::consts::PI * 8.0 / (N as f64 * DX);
        for i in 0..N {
            let value = (k0 * (i as f64 + 0.5) * DX).cos();
            for j in 0..4 {
                for k in 0..4 {
                    solver.fields.p[[i, j, k]] = value;
                }
            }
        }

        let energy = |solver: &FdtdSolver| -> f64 {
            let bulk = RHO0 * C0 * C0;
            let potential: f64 = solver.fields.p.iter().map(|&p| p * p / (2.0 * bulk)).sum();
            let kinetic: f64 = solver
                .fields
                .ux
                .iter()
                .zip(solver.fields.uy.iter())
                .zip(solver.fields.uz.iter())
                .map(|((&a, &b), &c)| 0.5 * RHO0 * (a * a + b * b + c * c))
                .sum();
            potential + kinetic
        };

        let initial = energy(&solver);
        assert!(initial > 0.0, "order {order}: zero initial energy");
        let mut lowest = f64::INFINITY;
        let mut highest = 0.0_f64;
        for _ in 0..STEPS {
            solver.step_forward().unwrap();
            let ratio = energy(&solver) / initial;
            lowest = lowest.min(ratio);
            highest = highest.max(ratio);
        }
        assert!(
            highest < 1.1 && lowest > 0.9,
            "order {order}: energy drifted to [{lowest:.4}, {highest:.4}]"
        );
        let _ = TAU;
    }
}

/// Eighth order is rejected on the collocated path, whose CFL table stops at
/// six — a configuration the solver cannot size a stable step for must fail
/// loudly rather than run.
#[test]
fn collocated_path_rejects_eighth_order() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let config = FdtdConfig {
        spatial_order: 8,
        staggered_grid: false,
        dt: 1e-8,
        nt: 4,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let staggered = FdtdConfig {
        staggered_grid: true,
        ..config
    };
    assert!(staggered.validate().is_ok());
    let _ = grid;
}

/// **The collocated path conserves energy, and it conserves it in the norm the
/// operator carries.**
///
/// This is the payoff of ADR 107, tested where it matters — a full solver run
/// rather than the operator in isolation. The summation-by-parts property gives
/// `d/dt ‖E‖_H = −(p_{N−1}u_{N−1} − p₀u₀)` per axis, so conservation needs both
/// halves: the operator *and* the rigid wall holding the wall-normal velocity at
/// zero. Removing either breaks it.
///
/// The weights are load-bearing. `H` is the trapezoidal rule — half at the end
/// points, one inside — so an unweighted sum measures a quantity the scheme has
/// no reason to preserve. The test asserts against `norm_weight` for that
/// reason, not as a formality.
#[test]
fn the_collocated_path_conserves_the_weighted_energy() {
    const N: usize = 24;
    const STEPS: usize = 1200;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.2 / (3.0_f64).sqrt() * dx / c0;

    let grid = Grid::new(N, N, N, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: false,
        enable_nonlinear: false,
        dt,
        nt: STEPS + 1,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    let centre = (N / 2) as f64;
    let sigma = 3.0_f64;
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                let r2 = (i as f64 - centre).powi(2)
                    + (j as f64 - centre).powi(2)
                    + (k as f64 - centre).powi(2);
                solver.fields.p[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    let weighted_energy = |solver: &FdtdSolver| -> f64 {
        let op = &solver.conservative_operator;
        let mut total = 0.0;
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    // Separable quadrature: the product of the three axes'
                    // weights is the cell's weight in the 3-D norm.
                    let weight = op.norm_weight(Axis::X, i)
                        * op.norm_weight(Axis::Y, j)
                        * op.norm_weight(Axis::Z, k);
                    let p = solver.fields.p[[i, j, k]];
                    let kinetic = solver.fields.ux[[i, j, k]].powi(2)
                        + solver.fields.uy[[i, j, k]].powi(2)
                        + solver.fields.uz[[i, j, k]].powi(2);
                    total += weight * (p * p / (2.0 * rho0 * c0 * c0) + rho0 * kinetic / 2.0);
                }
            }
        }
        total
    };

    let initial = weighted_energy(&solver);
    for _ in 0..STEPS {
        solver.step_forward().expect("collocated step");
    }
    let final_energy = weighted_energy(&solver);
    let drift = (final_energy - initial).abs() / initial;
    assert!(
        drift < 0.05,
        "weighted energy drifted {:.3} % over {STEPS} steps",
        100.0 * drift
    );
}

/// A transversely uniform field stays uniform on the collocated path.
///
/// The defect KW-SOL-086 fixed, at solver level: under the previous
/// zero-extension closure the walls were pressure-release, so a field uniform
/// across a thin axis had a large gradient there and the axis developed
/// transverse motion out of nothing. Here the launch has no transverse
/// structure and must acquire none.
#[test]
fn the_collocated_path_leaves_a_thin_axis_inert() {
    const N: usize = 48;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.2 / (3.0_f64).sqrt() * dx / c0;

    let grid = Grid::new(N, 4, 4, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: false,
        enable_nonlinear: false,
        dt,
        nt: 401,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    for i in 0..N {
        let value = (-((i as f64 - 24.0) / 4.0).powi(2)).exp();
        for j in 0..4 {
            for k in 0..4 {
                solver.fields.p[[i, j, k]] = value;
            }
        }
    }

    for _ in 0..400 {
        solver.step_forward().expect("collocated step");
    }

    let axial: f64 = solver.fields.ux.iter().map(|v| v * v).sum();
    let transverse: f64 = solver.fields.uy.iter().map(|v| v * v).sum::<f64>()
        + solver.fields.uz.iter().map(|v| v * v).sum::<f64>();
    assert!(axial > 0.0, "the packet must actually be propagating");
    assert!(
        transverse <= 1e-12 * axial,
        "a uniform transverse profile developed motion: transverse/axial = {:.3e}",
        transverse / axial
    );
}
