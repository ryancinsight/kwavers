//! Shared 1-D acoustic DG/FDTD/PSTD comparison fixture for examples.
#![allow(dead_code)] // Shared example fixture: each example target uses a subset.
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::HomogeneousMedium;
use kwavers_domain::source::{GridSource, SourceMode};
use kwavers_solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use kwavers_solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod};
use kwavers_solver::forward::pstd::dg::dg_solver::acoustic::AcousticDg1DWorkspace;
use kwavers_solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
use kwavers_solver::forward::pstd::dg::{DGConfig, DGSolver};
use kwavers_solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers_solver::interface::solver::Solver;
use kwavers_core::error::KwaversResult;
use ndarray::{Array1, Array3};
use std::f64::consts::PI;
use std::sync::Arc;
pub const ELEMENTS: usize = 12;
pub const POLYNOMIAL_ORDER: usize = 2;
pub const SOUND_SPEED: f64 = 1.0;
pub const DENSITY: f64 = 1.0;
pub const DT: f64 = 0.005;
pub const STEPS: usize = 40;
pub const EMBEDDED_NY: usize = 4;
pub const EMBEDDED_NZ: usize = 4;
const GAUSSIAN_SIGMA: f64 = 3.0;
#[path = "lines.rs"]
mod lines;
#[path = "sampling.rs"]
mod sampling;
pub use lines::NamedLine;
use lines::{absolute_error, center_line, dg_line, uniform_line};
pub use sampling::CommonGaussianMatrix;
#[derive(Debug, Clone, Copy)]
pub struct NativeAcousticDiagnostic {
    pub pressure_relative_l2: f64,
    pub velocity_relative_l2: f64,
    pub pressure_characteristic_l2: f64,
    pub velocity_characteristic_l2: f64,
    pub pressure_mass_error: f64,
    pub velocity_mass_error: f64,
    pub energy_ratio: f64,
}
#[derive(Debug, Clone, Copy)]
pub struct EmbeddedGaussianMatrix {
    pub dg_exact_l2: f64,
    pub fdtd_exact_l2: f64,
    pub kspace_exact_l2: f64,
    pub pstd_exact_l2: f64,
    pub fdtd_pstd_l2: f64,
    pub kspace_pstd_l2: f64,
    pub dg_pressure_mass_error: f64,
}
#[derive(Debug, Clone)]
pub struct EmbeddedGaussianSeries {
    pub matrix: EmbeddedGaussianMatrix,
    pub common_matrix: CommonGaussianMatrix,
    pub pressure_lines: Vec<NamedLine>,
    pub error_lines: Vec<NamedLine>,
    pub common_pressure_lines: Vec<NamedLine>,
    pub common_error_lines: Vec<NamedLine>,
}
pub fn run_native_acoustic_diagnostic() -> KwaversResult<NativeAcousticDiagnostic> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let wavenumber = 2.0 * PI / domain_length();
    let final_time = STEPS as f64 * DT;
    let (mut pressure, mut velocity) = initial_standing_wave(&xi_nodes, wavenumber);
    let initial_pressure_mass = weighted_mass(&pressure, &weights);
    let initial_velocity_mass = weighted_mass(&velocity, &weights);
    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, grid)?;
    let mut workspace = AcousticDg1DWorkspace::new(pressure.dim());
    for _ in 0..STEPS {
        solver.step_acoustic_1d_ssp_rk3(
            &mut pressure,
            &mut velocity,
            DENSITY,
            DT,
            &mut workspace,
        )?;
    }

    let (exact_pressure, exact_velocity) =
        exact_standing_wave(&xi_nodes, wavenumber, SOUND_SPEED * final_time);
    let (characteristic_pressure, characteristic_velocity) =
        characteristic_reference(&xi_nodes, wavenumber)?;

    Ok(NativeAcousticDiagnostic {
        pressure_relative_l2: relative_l2(&pressure, &exact_pressure, &weights),
        velocity_relative_l2: relative_l2(&velocity, &exact_velocity, &weights),
        pressure_characteristic_l2: relative_l2(&pressure, &characteristic_pressure, &weights),
        velocity_characteristic_l2: relative_l2(&velocity, &characteristic_velocity, &weights),
        pressure_mass_error: (weighted_mass(&pressure, &weights) - initial_pressure_mass).abs(),
        velocity_mass_error: (weighted_mass(&velocity, &weights) - initial_velocity_mass).abs(),
        energy_ratio: acoustic_energy(&pressure, &velocity, &weights)
            / acoustic_energy(&exact_pressure, &exact_velocity, &weights).max(f64::EPSILON),
    })
}

pub fn run_embedded_gaussian_solver_matrix() -> KwaversResult<EmbeddedGaussianMatrix> {
    Ok(run_embedded_gaussian_series()?.matrix)
}

pub fn run_embedded_gaussian_series() -> KwaversResult<EmbeddedGaussianSeries> {
    let (dg_pressure, dg_mass_error, xi_nodes, weights) = run_dg_gaussian()?;
    let fdtd_line = run_fdtd_gaussian(KSpaceCorrectionMode::None)?;
    let kspace_line = run_fdtd_gaussian(KSpaceCorrectionMode::Spectral)?;
    let pstd_line = run_pstd_gaussian()?;
    let final_time = STEPS as f64 * DT;
    let exact_uniform = exact_uniform_gaussian_line(final_time);
    let exact_dg = exact_dg_gaussian_pressure(&xi_nodes, final_time);
    let common_samples = sampling::common_gaussian_samples(
        &dg_pressure,
        &xi_nodes,
        &fdtd_line,
        &kspace_line,
        &pstd_line,
    )?;

    let matrix = EmbeddedGaussianMatrix {
        dg_exact_l2: relative_l2(&dg_pressure, &exact_dg, &weights),
        fdtd_exact_l2: relative_l2_line(&fdtd_line, &exact_uniform),
        kspace_exact_l2: relative_l2_line(&kspace_line, &exact_uniform),
        pstd_exact_l2: relative_l2_line(&pstd_line, &exact_uniform),
        fdtd_pstd_l2: relative_l2_line(&fdtd_line, &pstd_line),
        kspace_pstd_l2: relative_l2_line(&kspace_line, &pstd_line),
        dg_pressure_mass_error: dg_mass_error,
    };

    let dg_exact = dg_line(&exact_dg, &xi_nodes);
    let uniform_exact = uniform_line(&exact_uniform);
    let pressure_lines = vec![
        NamedLine {
            name: "exact",
            samples: uniform_exact.clone(),
        },
        NamedLine {
            name: "DG",
            samples: dg_line(&dg_pressure, &xi_nodes),
        },
        NamedLine {
            name: "FDTD",
            samples: uniform_line(&fdtd_line),
        },
        NamedLine {
            name: "FDTD+k-space",
            samples: uniform_line(&kspace_line),
        },
        NamedLine {
            name: "PSTD",
            samples: uniform_line(&pstd_line),
        },
    ];
    let error_lines = vec![
        NamedLine {
            name: "DG error",
            samples: absolute_error(&pressure_lines[1].samples, &dg_exact),
        },
        NamedLine {
            name: "FDTD error",
            samples: absolute_error(&pressure_lines[2].samples, &uniform_exact),
        },
        NamedLine {
            name: "FDTD+k-space error",
            samples: absolute_error(&pressure_lines[3].samples, &uniform_exact),
        },
        NamedLine {
            name: "PSTD error",
            samples: absolute_error(&pressure_lines[4].samples, &uniform_exact),
        },
    ];
    Ok(EmbeddedGaussianSeries {
        matrix,
        common_matrix: common_samples.matrix,
        pressure_lines,
        error_lines,
        common_pressure_lines: common_samples.pressure_lines,
        common_error_lines: common_samples.error_lines,
    })
}

pub fn print_solver_matrix(matrix: &EmbeddedGaussianMatrix) {
    println!("{:<28} {:>16.6e}", "DG vs exact", matrix.dg_exact_l2);
    println!("{:<28} {:>16.6e}", "FDTD vs exact", matrix.fdtd_exact_l2);
    println!(
        "{:<28} {:>16.6e}",
        "FDTD+k-space vs exact", matrix.kspace_exact_l2
    );
    println!("{:<28} {:>16.6e}", "PSTD vs exact", matrix.pstd_exact_l2);
    println!("{:<28} {:>16.6e}", "FDTD vs PSTD", matrix.fdtd_pstd_l2);
    println!(
        "{:<28} {:>16.6e}",
        "FDTD+k-space vs PSTD", matrix.kspace_pstd_l2
    );
    println!(
        "{:<28} {:>16.6e}",
        "DG pressure mass error", matrix.dg_pressure_mass_error
    );
}

pub fn print_common_solver_matrix(matrix: &CommonGaussianMatrix) {
    sampling::print_common_solver_matrix(matrix);
}

pub fn embedded_grid() -> KwaversResult<Grid> {
    Ok(Grid::new(
        2 * ELEMENTS,
        EMBEDDED_NY,
        EMBEDDED_NZ,
        1.0,
        1.0,
        1.0,
    )?)
}

pub fn gaussian_embedded_source(grid: &Grid) -> GridSource {
    let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut ux = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let uy = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let uz = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        let x = i as f64;
        let ux_x = x + 0.5;
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                pressure[(i, j, k)] = gaussian_profile(x);
                ux[(i, j, k)] = 0.5 * DT / DENSITY * gaussian_derivative(ux_x);
            }
        }
    }
    GridSource {
        p0: Some(pressure),
        u0: Some((ux, uy, uz)),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    }
}

fn run_dg_gaussian() -> KwaversResult<(Array3<f64>, f64, Array1<f64>, Array1<f64>)> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let mut pressure = Array3::zeros((ELEMENTS, n_nodes, 1));
    let mut velocity = Array3::zeros((ELEMENTS, n_nodes, 1));
    for elem in 0..ELEMENTS {
        for node in 0..n_nodes {
            pressure[(elem, node, 0)] = gaussian_profile(physical_coordinate(elem, xi_nodes[node]));
        }
    }
    let initial_mass = weighted_mass(&pressure, &weights);

    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, grid)?;
    let mut workspace = AcousticDg1DWorkspace::new(pressure.dim());
    for _ in 0..STEPS {
        solver.step_acoustic_1d_ssp_rk3(
            &mut pressure,
            &mut velocity,
            DENSITY,
            DT,
            &mut workspace,
        )?;
    }
    let mass_error = (weighted_mass(&pressure, &weights) - initial_mass).abs();
    Ok((pressure, mass_error, xi_nodes, weights))
}

fn run_fdtd_gaussian(kspace_correction: KSpaceCorrectionMode) -> KwaversResult<Array1<f64>> {
    let grid = embedded_grid()?;
    let medium = HomogeneousMedium::new(DENSITY, SOUND_SPEED, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: DT * SOUND_SPEED,
        kspace_correction,
        dt: DT,
        nt: STEPS,
        ..FdtdConfig::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, gaussian_embedded_source(&grid))?;
    for _ in 0..STEPS {
        solver.step_forward()?;
    }
    Ok(center_line(Solver::pressure_field(&solver)))
}

fn run_pstd_gaussian() -> KwaversResult<Array1<f64>> {
    let grid = embedded_grid()?;
    let medium = HomogeneousMedium::new(DENSITY, SOUND_SPEED, 0.0, 0.0, &grid);
    let config = PSTDConfig {
        dt: DT,
        nt: STEPS,
        boundary: BoundaryConfig::None,
        kspace_method: KSpaceMethod::StandardPSTD,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(
        config,
        grid.clone(),
        &medium,
        gaussian_embedded_source(&grid),
    )?;
    for _ in 0..STEPS {
        solver.step_forward()?;
    }
    Ok(center_line(Solver::pressure_field(&solver)))
}

fn characteristic_reference(
    xi_nodes: &Array1<f64>,
    wavenumber: f64,
) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
    let w_plus = evolve_characteristic(xi_nodes, |x| (wavenumber * x).sin())?;
    let reflected_minus =
        evolve_characteristic(xi_nodes, |x| (wavenumber * (domain_length() - x)).sin())?;
    let w_minus = reflect_coefficients(&reflected_minus);
    Ok(pressure_velocity_from_characteristics(&w_plus, &w_minus))
}

fn evolve_characteristic(
    xi_nodes: &Array1<f64>,
    initial: impl Fn(f64) -> f64,
) -> KwaversResult<Array3<f64>> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let mut solver = DGSolver::new(config, grid)?;
    solver.initialize_modal_coefficients(ELEMENTS, 1);
    {
        let coeffs = solver.modal_coefficients_mut().expect("coefficients");
        for elem in 0..ELEMENTS {
            for node in 0..xi_nodes.len() {
                coeffs[(elem, node, 0)] = initial(physical_coordinate(elem, xi_nodes[node]));
            }
        }
    }
    let mut ignored_grid_field = Array3::zeros((ELEMENTS * n_nodes, 1, 1));
    for _ in 0..STEPS {
        solver.solve_step(&mut ignored_grid_field, DT)?;
    }
    Ok(solver.modal_coefficients().expect("coefficients").clone())
}

fn initial_standing_wave(xi_nodes: &Array1<f64>, k: f64) -> (Array3<f64>, Array3<f64>) {
    let mut pressure = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    let velocity = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            pressure[(elem, node, 0)] = (k * physical_coordinate(elem, xi_nodes[node])).sin();
        }
    }
    (pressure, velocity)
}

fn exact_standing_wave(
    xi_nodes: &Array1<f64>,
    k: f64,
    displacement: f64,
) -> (Array3<f64>, Array3<f64>) {
    let mut pressure = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    let mut velocity = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            let phase = k * physical_coordinate(elem, xi_nodes[node]);
            pressure[(elem, node, 0)] = phase.sin() * (k * displacement).cos();
            velocity[(elem, node, 0)] =
                -phase.cos() * (k * displacement).sin() / (DENSITY * SOUND_SPEED);
        }
    }
    (pressure, velocity)
}

fn reflect_coefficients(coeffs: &Array3<f64>) -> Array3<f64> {
    let mut reflected = Array3::zeros(coeffs.raw_dim());
    let n_nodes = coeffs.dim().1;
    for elem in 0..ELEMENTS {
        for node in 0..n_nodes {
            reflected[(elem, node, 0)] = coeffs[(ELEMENTS - 1 - elem, n_nodes - 1 - node, 0)];
        }
    }
    reflected
}

fn pressure_velocity_from_characteristics(
    w_plus: &Array3<f64>,
    w_minus: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>) {
    (
        0.5 * (w_plus + w_minus),
        (w_plus - w_minus) / (2.0 * DENSITY * SOUND_SPEED),
    )
}
pub fn physical_coordinate(elem: usize, xi: f64) -> f64 {
    2.0 * elem as f64 + xi + 1.0
}
pub fn domain_length() -> f64 {
    2.0 * ELEMENTS as f64
}
fn gaussian_center() -> f64 {
    0.5 * domain_length()
}
pub fn gaussian_profile(x: f64) -> f64 {
    let scaled = (x - gaussian_center()) / GAUSSIAN_SIGMA;
    (-0.5 * scaled * scaled).exp()
}
pub fn gaussian_derivative(x: f64) -> f64 {
    -(x - gaussian_center()) / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA) * gaussian_profile(x)
}
pub fn exact_gaussian_pressure(x: f64, time: f64) -> f64 {
    0.5 * (gaussian_profile(x - SOUND_SPEED * time) + gaussian_profile(x + SOUND_SPEED * time))
}
fn exact_uniform_gaussian_line(time: f64) -> Array1<f64> {
    Array1::from_shape_fn(2 * ELEMENTS, |i| exact_gaussian_pressure(i as f64, time))
}
fn exact_dg_gaussian_pressure(xi_nodes: &Array1<f64>, time: f64) -> Array3<f64> {
    let mut pressure = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            pressure[(elem, node, 0)] =
                exact_gaussian_pressure(physical_coordinate(elem, xi_nodes[node]), time);
        }
    }
    pressure
}

pub fn weighted_mass(values: &Array3<f64>, weights: &Array1<f64>) -> f64 {
    let mut mass = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            mass += weights[node] * values[(elem, node, 0)];
        }
    }
    mass
}

pub fn relative_l2(actual: &Array3<f64>, expected: &Array3<f64>, weights: &Array1<f64>) -> f64 {
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            let diff = actual[(elem, node, 0)] - expected[(elem, node, 0)];
            diff_sq += weights[node] * diff * diff;
            expected_sq += weights[node] * expected[(elem, node, 0)] * expected[(elem, node, 0)];
        }
    }
    diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON)
}

fn relative_l2_line(actual: &Array1<f64>, expected: &Array1<f64>) -> f64 {
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        let diff = actual - expected;
        diff_sq += diff * diff;
        expected_sq += expected * expected;
    }
    diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON)
}

fn acoustic_energy(pressure: &Array3<f64>, velocity: &Array3<f64>, weights: &Array1<f64>) -> f64 {
    let mut energy = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            let p = pressure[(elem, node, 0)];
            let u = velocity[(elem, node, 0)];
            energy += weights[node]
                * (p * p / (2.0 * DENSITY * SOUND_SPEED * SOUND_SPEED) + 0.5 * DENSITY * u * u);
        }
    }
    energy
}
