//! FDTD/PSTD solver comparison on one acoustic initial-value problem.
//!
//! This example runs three real solvers on the same homogeneous, lossless
//! Gaussian pressure pulse:
//! - classical staggered-grid FDTD,
//! - FDTD with spectral k-space derivative correction,
//! - standard PSTD.
//!
//! ## Theorem
//! For the linear acoustic system
//! ```text
//! rho0 * du/dt = -grad(p)
//! dp/dt        = -rho0 * c0^2 * div(u),
//! ```
//! all three discretizations approximate the same Cauchy problem when given the
//! same `p(t=0)` and leapfrog-compatible `u(t=-dt/2)`.
//!
//! ## Alignment Contract
//! `KSpaceCorrectionMode::Spectral` replaces finite-difference derivatives with
//! the same half-cell shifted spectral derivative family used by PSTD. In a
//! short propagation window where the pulse does not reach the boundary, its
//! pressure field should align more closely with PSTD than classical FDTD in
//! relative L2 error and correlation.
//!
//! ## References
//! - Courant, Friedrichs & Lewy (1928). Math. Ann. 100, 32-74.
//! - Yee (1966). IEEE Trans. Antennas Propag. 14(3), 302-307.
//! - Virieux (1986). Geophysics 51(4), 889-901.
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Geophysics 63(6), 2082-2089.

use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::{GridSource, SourceMode};
use kwavers::solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use kwavers::solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod};
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers::solver::interface::solver::Solver;
use kwavers::KwaversResult;
use ndarray::{Array3, ArrayView3};
use std::time::{Duration, Instant};

const NX: usize = 16;
const DX: f64 = 0.5e-3;
const C0: f64 = 1500.0;
const RHO0: f64 = 1000.0;
const CFL: f64 = 0.25;
const NT: usize = 8;
const P0_AMPLITUDE: f64 = 1.0;
const SIGMA_CELLS: f64 = 2.0;

fn main() -> KwaversResult<()> {
    let grid = Grid::new(NX, NX, NX, DX, DX, DX)?;
    let medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
    let dt = CFL * DX / C0;

    println!("FDTD/PSTD pressure-field comparison");
    println!(
        "grid: {NX}^3, dx: {:.3} mm, dt: {:.3} ns, steps: {NT}",
        DX * 1e3,
        dt * 1e9
    );
    println!("problem: homogeneous lossless Gaussian IVP, zero initial dp/dt");
    println!();

    let fdtd = run_fdtd("FDTD", KSpaceCorrectionMode::None, &grid, &medium, dt, NT)?;
    let fdtd_kspace = run_fdtd(
        "FDTD+k-space",
        KSpaceCorrectionMode::Spectral,
        &grid,
        &medium,
        dt,
        NT,
    )?;
    let pstd = run_pstd(&grid, &medium, dt, NT)?;

    let fdtd_vs_pstd = compare_fields(
        "FDTD",
        Solver::pressure_field(&fdtd.solver).view(),
        "PSTD",
        Solver::pressure_field(&pstd.solver).view(),
    );
    let kspace_vs_pstd = compare_fields(
        "FDTD+k-space",
        Solver::pressure_field(&fdtd_kspace.solver).view(),
        "PSTD",
        Solver::pressure_field(&pstd.solver).view(),
    );
    let fdtd_vs_kspace = compare_fields(
        "FDTD",
        Solver::pressure_field(&fdtd.solver).view(),
        "FDTD+k-space",
        Solver::pressure_field(&fdtd_kspace.solver).view(),
    );

    print_run_summary(&[fdtd.summary(), fdtd_kspace.summary(), pstd.summary()]);
    print_comparison_table(&[fdtd_vs_pstd, kspace_vs_pstd, fdtd_vs_kspace]);

    let improvement = fdtd_vs_pstd.relative_l2 / kspace_vs_pstd.relative_l2.max(1.0e-12);
    println!();
    if kspace_vs_pstd.relative_l2 < 1.0e-12 {
        println!("alignment: k-space FDTD matches PSTD to machine precision on this fixture");
        println!("relative-L2 improvement lower bound vs classical FDTD = {improvement:.3e}x");
    } else {
        println!(
            "alignment: k-space FDTD relative-L2 improvement vs PSTD = {:.3}x",
            improvement
        );
    }
    println!("interpretation: values > 1 mean the spectral FDTD derivative path is closer to PSTD");

    Ok(())
}

struct FdtdRun {
    name: &'static str,
    solver: FdtdSolver,
    elapsed: Duration,
}

struct PstdRun {
    name: &'static str,
    solver: PSTDSolver,
    elapsed: Duration,
}

#[derive(Clone, Copy)]
struct RunSummary {
    name: &'static str,
    elapsed: Duration,
    energy_l2: f64,
    peak_abs: f64,
}

impl FdtdRun {
    fn summary(&self) -> RunSummary {
        summarize_field(
            self.name,
            Solver::pressure_field(&self.solver).view(),
            self.elapsed,
        )
    }
}

impl PstdRun {
    fn summary(&self) -> RunSummary {
        summarize_field(
            self.name,
            Solver::pressure_field(&self.solver).view(),
            self.elapsed,
        )
    }
}

#[derive(Clone, Copy)]
struct Comparison {
    lhs: &'static str,
    rhs: &'static str,
    relative_l2: f64,
    normalized_max_abs: f64,
    correlation: f64,
    energy_ratio: f64,
    centroid_shift_cells: f64,
}

fn run_fdtd(
    name: &'static str,
    kspace_correction: KSpaceCorrectionMode,
    grid: &Grid,
    medium: &HomogeneousMedium,
    dt: f64,
    nt: usize,
) -> KwaversResult<FdtdRun> {
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: CFL,
        kspace_correction,
        dt,
        nt,
        ..FdtdConfig::default()
    };
    let source = gaussian_initial_source(grid, dt);
    let mut solver = FdtdSolver::new(config, grid, medium, source)?;

    let start = Instant::now();
    for _ in 0..nt {
        solver.step_forward()?;
    }

    Ok(FdtdRun {
        name,
        solver,
        elapsed: start.elapsed(),
    })
}

fn run_pstd(grid: &Grid, medium: &HomogeneousMedium, dt: f64, nt: usize) -> KwaversResult<PstdRun> {
    let config = PSTDConfig {
        dt,
        nt,
        boundary: BoundaryConfig::None,
        kspace_method: KSpaceMethod::StandardPSTD,
        ..PSTDConfig::default()
    };
    let source = gaussian_initial_source(grid, dt);
    let mut solver = PSTDSolver::new(config, grid.clone(), medium, source)?;

    let start = Instant::now();
    for _ in 0..nt {
        solver.step_forward()?;
    }

    Ok(PstdRun {
        name: "PSTD",
        solver,
        elapsed: start.elapsed(),
    })
}

fn gaussian_initial_source(grid: &Grid, dt: f64) -> GridSource {
    let mut p0 = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut ux0 = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut uy0 = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut uz0 = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

    let center = (
        0.5 * (grid.nx as f64 - 1.0) * grid.dx,
        0.5 * (grid.ny as f64 - 1.0) * grid.dy,
        0.5 * (grid.nz as f64 - 1.0) * grid.dz,
    );
    let sigma = SIGMA_CELLS * grid.dx;

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                p0[[i, j, k]] = gaussian_pressure(x, y, z, center, sigma);

                if i + 1 < grid.nx {
                    let x_half = (i as f64 + 0.5) * grid.dx;
                    ux0[[i, j, k]] = 0.5 * dt / RHO0 * gaussian_dpdx(x_half, y, z, center, sigma);
                }
                if j + 1 < grid.ny {
                    let y_half = (j as f64 + 0.5) * grid.dy;
                    uy0[[i, j, k]] = 0.5 * dt / RHO0 * gaussian_dpdy(x, y_half, z, center, sigma);
                }
                if k + 1 < grid.nz {
                    let z_half = (k as f64 + 0.5) * grid.dz;
                    uz0[[i, j, k]] = 0.5 * dt / RHO0 * gaussian_dpdz(x, y, z_half, center, sigma);
                }
            }
        }
    }

    GridSource {
        p0: Some(p0),
        u0: Some((ux0, uy0, uz0)),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    }
}

fn gaussian_pressure(x: f64, y: f64, z: f64, center: (f64, f64, f64), sigma: f64) -> f64 {
    let dx = x - center.0;
    let dy = y - center.1;
    let dz = z - center.2;
    let r2 = dx * dx + dy * dy + dz * dz;
    P0_AMPLITUDE * (-r2 / (2.0 * sigma * sigma)).exp()
}

fn gaussian_dpdx(x: f64, y: f64, z: f64, center: (f64, f64, f64), sigma: f64) -> f64 {
    -(x - center.0) / (sigma * sigma) * gaussian_pressure(x, y, z, center, sigma)
}

fn gaussian_dpdy(x: f64, y: f64, z: f64, center: (f64, f64, f64), sigma: f64) -> f64 {
    -(y - center.1) / (sigma * sigma) * gaussian_pressure(x, y, z, center, sigma)
}

fn gaussian_dpdz(x: f64, y: f64, z: f64, center: (f64, f64, f64), sigma: f64) -> f64 {
    -(z - center.2) / (sigma * sigma) * gaussian_pressure(x, y, z, center, sigma)
}

fn summarize_field(
    name: &'static str,
    field: ArrayView3<'_, f64>,
    elapsed: Duration,
) -> RunSummary {
    let mut energy_sq: f64 = 0.0;
    let mut peak_abs: f64 = 0.0;
    for &value in field.iter() {
        assert!(
            value.is_finite(),
            "{name} pressure field contains a non-finite value"
        );
        energy_sq += value * value;
        peak_abs = peak_abs.max(value.abs());
    }
    RunSummary {
        name,
        elapsed,
        energy_l2: energy_sq.sqrt(),
        peak_abs,
    }
}

fn compare_fields(
    lhs: &'static str,
    a: ArrayView3<'_, f64>,
    rhs: &'static str,
    b: ArrayView3<'_, f64>,
) -> Comparison {
    assert_eq!(a.dim(), b.dim(), "comparison fields must share dimensions");

    let mut diff_sq: f64 = 0.0;
    let mut a_sq: f64 = 0.0;
    let mut b_sq: f64 = 0.0;
    let mut max_abs: f64 = 0.0;
    let mut sum_a: f64 = 0.0;
    let mut sum_b: f64 = 0.0;
    let mut sum_ab: f64 = 0.0;
    let mut max_signal: f64 = 0.0;

    for (&av, &bv) in a.iter().zip(b.iter()) {
        assert!(
            av.is_finite(),
            "{lhs} pressure field contains a non-finite value"
        );
        assert!(
            bv.is_finite(),
            "{rhs} pressure field contains a non-finite value"
        );
        let diff = av - bv;
        diff_sq += diff * diff;
        a_sq += av * av;
        b_sq += bv * bv;
        max_abs = max_abs.max(diff.abs());
        sum_a += av;
        sum_b += bv;
        sum_ab += av * bv;
        max_signal = max_signal.max(av.abs()).max(bv.abs());
    }

    let n = a.len() as f64;
    let denom_l2 = a_sq.sqrt().max(b_sq.sqrt()).max(f64::EPSILON);
    let var_a = (a_sq - sum_a * sum_a / n).max(0.0);
    let var_b = (b_sq - sum_b * sum_b / n).max(0.0);
    let cov = sum_ab - sum_a * sum_b / n;
    let correlation = if var_a > 0.0 && var_b > 0.0 {
        cov / (var_a * var_b).sqrt()
    } else {
        1.0
    };

    let relative_l2 = diff_sq.sqrt() / denom_l2;

    Comparison {
        lhs,
        rhs,
        relative_l2,
        normalized_max_abs: max_abs / max_signal.max(f64::EPSILON),
        correlation,
        energy_ratio: a_sq.sqrt() / b_sq.sqrt().max(f64::EPSILON),
        centroid_shift_cells: centroid_shift_cells(a, b),
    }
}

fn centroid_shift_cells(a: ArrayView3<'_, f64>, b: ArrayView3<'_, f64>) -> f64 {
    let ca = energy_centroid_cells(a);
    let cb = energy_centroid_cells(b);
    ((ca.0 - cb.0).powi(2) + (ca.1 - cb.1).powi(2) + (ca.2 - cb.2).powi(2)).sqrt()
}

fn energy_centroid_cells(field: ArrayView3<'_, f64>) -> (f64, f64, f64) {
    let mut weight_sum = 0.0;
    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut z_sum = 0.0;

    for ((i, j, k), &value) in field.indexed_iter() {
        let weight = value * value;
        weight_sum += weight;
        x_sum += weight * i as f64;
        y_sum += weight * j as f64;
        z_sum += weight * k as f64;
    }

    if weight_sum > f64::EPSILON {
        (x_sum / weight_sum, y_sum / weight_sum, z_sum / weight_sum)
    } else {
        (0.0, 0.0, 0.0)
    }
}

fn print_run_summary(runs: &[RunSummary]) {
    println!("solver run summary");
    println!(
        "{:<14} {:>12} {:>16} {:>16}",
        "solver", "time_ms", "||p||_2", "max|p|"
    );
    for run in runs {
        println!(
            "{:<14} {:>12.3} {:>16.6e} {:>16.6e}",
            run.name,
            run.elapsed.as_secs_f64() * 1e3,
            run.energy_l2,
            run.peak_abs
        );
    }
    println!();
}

fn print_comparison_table(comparisons: &[Comparison]) {
    println!("pairwise final-pressure metrics");
    println!(
        "{:<27} {:>12} {:>12} {:>12} {:>12} {:>14}",
        "pair", "rel_l2", "max_rel", "corr", "energy", "centroid_dx"
    );
    for comparison in comparisons {
        println!(
            "{:<27} {:>12.5e} {:>12.5e} {:>12.6} {:>12.6} {:>14.6}",
            format!("{} vs {}", comparison.lhs, comparison.rhs),
            comparison.relative_l2,
            comparison.normalized_max_abs,
            comparison.correlation,
            comparison.energy_ratio,
            comparison.centroid_shift_cells
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_initial_source_is_finite_and_centered() {
        let grid = Grid::new(NX, NX, NX, DX, DX, DX).unwrap();
        let source = gaussian_initial_source(&grid, CFL * DX / C0);
        let p0 = source.p0.as_ref().expect("p0 must be present");
        let centroid = energy_centroid_cells(p0.view());
        let expected = 0.5 * (NX as f64 - 1.0);

        assert!(p0.iter().all(|value| value.is_finite()));
        assert!((centroid.0 - expected).abs() < 0.05);
        assert!((centroid.1 - expected).abs() < 0.05);
        assert!((centroid.2 - expected).abs() < 0.05);
    }

    #[test]
    fn comparison_metrics_detect_identical_fields() {
        let field = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
            (i as f64 + 2.0 * j as f64 - k as f64).sin()
        });
        let metrics = compare_fields("a", field.view(), "b", field.view());

        assert_eq!(metrics.relative_l2, 0.0);
        assert_eq!(metrics.normalized_max_abs, 0.0);
        assert!((metrics.correlation - 1.0).abs() < 1e-12);
        assert!((metrics.energy_ratio - 1.0).abs() < 1e-12);
        assert_eq!(metrics.centroid_shift_cells, 0.0);
    }

    #[test]
    fn pressure_source_signal_not_allocated_for_ivp_only_case() {
        let grid = Grid::new(NX, NX, NX, DX, DX, DX).unwrap();
        let source = gaussian_initial_source(&grid, CFL * DX / C0);

        assert!(source.p0.is_some());
        assert!(source.u0.is_some());
        assert!(source.p_signal.is_none());
        assert!(source.p_mask.is_none());
        assert!(source.u_signal.is_none());
        assert!(source.u_mask.is_none());
    }
}
