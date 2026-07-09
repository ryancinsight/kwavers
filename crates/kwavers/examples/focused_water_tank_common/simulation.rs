use super::physics;
use super::AxialField;
use super::SolverField;
use anyhow::Result;
use kwavers_boundary::cpml::CPMLConfig;
use kwavers_grid::Grid;
use kwavers_solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use kwavers_solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use kwavers_solver::forward::pstd::dg::dg_solver::acoustic::{
    AcousticDg1DWorkspace, AcousticDgTensorWorkspace, ACOUSTIC_PRESSURE_VAR,
};
use kwavers_solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
use kwavers_solver::forward::pstd::dg::{
    DGConfig, DGSolver, DgBoundaryCondition, DgCpmlAxis, DgCpmlConfig, DgCpmlMemoryWorkspace,
    DgCpmlProfiles,
};
use kwavers_solver::forward::pstd::PSTDSolver;
use kwavers_solver::interface::solver::Solver;
use leto::Array1;
use leto::{
    Array2,
    Array3,
};
use std::sync::Arc;
use std::time::Instant;

const DG_POLYNOMIAL_ORDER: usize = 2;
const DG_ELEMENTS: usize = physics::NY;
const DG_SUBSTEPS_PER_STEP: usize = 2;
const DG_SOURCE_SIGMA_CELLS: f64 = 0.75;
const DG_TENSOR_POLYNOMIAL_ORDER: usize = 3;
const DG_TENSOR_SUBSTEPS_PER_STEP: usize = 2;

pub fn run_solver_fields() -> Result<Vec<SolverField>> {
    Ok(vec![
        run_fdtd()?,
        run_pstd()?,
        run_dg_tensor_field("DG-2D", 1)?,
        run_dg_tensor_field("DG-3D", physics::NZ)?,
        run_dg_tensor_field_with_cpml("DG-3D-CPML", physics::NZ)?,
    ])
}

pub fn run_dg_axial_field() -> Result<AxialField> {
    let n_nodes = DG_POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let grid = Arc::new(Grid::new(DG_ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: DG_POLYNOMIAL_ORDER,
        sound_speed: physics::C0 / physics::DX,
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, grid)?;
    let mut pressure = Array3::<f64>::zeros((DG_ELEMENTS, n_nodes, 1));
    let mut velocity = Array3::<f64>::zeros((DG_ELEMENTS, n_nodes, 1));
    let mut workspace = AcousticDg1DWorkspace::new(pressure.dim());
    let source_weights = dg_source_weights(&xi_nodes, &weights);
    let density = physics::RHO0 * physics::DX;
    let dt_sub = physics::DT / DG_SUBSTEPS_PER_STEP as f64;

    let start = Instant::now();
    let mut peak = vec![0.0; physics::NY];
    for step in 0..physics::NT {
        for substep in 0..DG_SUBSTEPS_PER_STEP {
            let t = (step * DG_SUBSTEPS_PER_STEP + substep) as f64 * dt_sub;
            solver.step_acoustic_1d_ssp_rk3(
                &mut pressure,
                &mut velocity,
                density,
                dt_sub,
                &mut workspace,
            )?;
            apply_dg_source(&mut pressure, &source_weights, t, dt_sub);
        }
        if (physics::GATE_START..=physics::GATE_END).contains(&step) {
            for (y, value) in peak.iter_mut().enumerate() {
                *value = f64::max(
                    *value,
                    sample_dg_pressure(&pressure, &xi_nodes, y as f64).abs(),
                );
            }
        }
    }

    Ok(AxialField {
        name: "DG-1D axial",
        normalized_peak: normalize_line(&peak),
        elapsed: start.elapsed(),
    })
}

fn run_fdtd() -> Result<SolverField> {
    let grid = physics::grid()?;
    let medium = physics::medium(&grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: physics::CFL,
        kspace_correction: KSpaceCorrectionMode::None,
        dt: physics::DT,
        nt: physics::NT,
        ..FdtdConfig::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, physics::focused_source())?;
    solver.enable_cpml(cpml_config(), physics::DT, physics::C0)?;

    let start = Instant::now();
    let mut peak = Array2::<f64>::zeros((physics::NX, physics::NY));
    for step in 0..physics::NT {
        solver.step_forward()?;
        if (physics::GATE_START..=physics::GATE_END).contains(&step) {
            update_peak("FDTD", &mut peak, Solver::pressure_field(&solver));
        }
    }

    Ok(SolverField {
        name: "FDTD",
        normalized_peak: super::metrics::normalize_map(&peak),
        elapsed: start.elapsed(),
    })
}

fn run_pstd() -> Result<SolverField> {
    let grid = physics::grid()?;
    let medium = physics::medium(&grid);
    let config = PSTDConfig {
        dt: physics::DT,
        nt: physics::NT,
        boundary: BoundaryConfig::CPML(cpml_config()),
        pml_inside: true,
        smooth_sources: false,
        kspace_method: KSpaceMethod::StandardPSTD,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid, &medium, physics::focused_source())?;

    let start = Instant::now();
    let mut peak = Array2::<f64>::zeros((physics::NX, physics::NY));
    for step in 0..physics::NT {
        solver.step_forward()?;
        if (physics::GATE_START..=physics::GATE_END).contains(&step) {
            update_peak("PSTD", &mut peak, solver.pressure_field());
        }
    }

    Ok(SolverField {
        name: "PSTD",
        normalized_peak: super::metrics::normalize_map(&peak),
        elapsed: start.elapsed(),
    })
}

fn run_dg_tensor_field(name: &'static str, nz: usize) -> Result<SolverField> {
    let grid = Arc::new(Grid::new(
        physics::NX,
        physics::NY,
        nz,
        physics::DX,
        physics::DX,
        physics::DX,
    )?);
    let config = DGConfig {
        polynomial_order: DG_TENSOR_POLYNOMIAL_ORDER,
        sound_speed: physics::C0,
        boundary_conditions: [
            DgBoundaryCondition::AbsorbingCharacteristic,
            DgBoundaryCondition::AbsorbingCharacteristic,
            DgBoundaryCondition::Periodic,
        ],
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, Arc::clone(&grid))?;
    let mut state = Array3::<f64>::zeros(solver.acoustic_tensor_state_shape()?);
    let mut workspace = AcousticDgTensorWorkspace::new(state.dim());
    let sources = dg_tensor_sources(&solver, nz)?;
    let dt_sub = physics::DT / DG_TENSOR_SUBSTEPS_PER_STEP as f64;
    let mut pressure = Array3::<f64>::zeros((physics::NX, physics::NY, nz));

    let start = Instant::now();
    let mut peak = Array2::<f64>::zeros((physics::NX, physics::NY));
    for step in 0..physics::NT {
        for substep in 0..DG_TENSOR_SUBSTEPS_PER_STEP {
            let t = (step * DG_TENSOR_SUBSTEPS_PER_STEP + substep) as f64 * dt_sub;
            solver.step_acoustic_tensor_ssp_rk3_with_source(
                &mut state,
                physics::RHO0,
                dt_sub,
                &mut workspace,
                t,
                |stage_t, rhs| apply_dg_tensor_source_rhs(rhs, &sources, stage_t),
            )?;
        }
        if (physics::GATE_START..=physics::GATE_END).contains(&step) {
            solver.project_acoustic_tensor_pressure_to_uniform_grid(&state, &mut pressure)?;
            update_peak(name, &mut peak, &pressure);
        }
    }

    Ok(SolverField {
        name,
        normalized_peak: super::metrics::normalize_map(&peak),
        elapsed: start.elapsed(),
    })
}

fn run_dg_tensor_field_with_cpml(name: &'static str, nz: usize) -> Result<SolverField> {
    // Non-slab finite 3-D water-tank closure with CPML in x and y. The z axis
    // has only 1 DG element (nz == 4 grid points, n_nodes == 4), too thin for
    // a graded CPML strip, so it keeps the absorbing-characteristic exterior
    // state — exact for normally incident plane waves on the source-invariant
    // z direction.
    let grid = Arc::new(Grid::new(
        physics::NX,
        physics::NY,
        nz,
        physics::DX,
        physics::DX,
        physics::DX,
    )?);
    let n_nodes = DG_TENSOR_POLYNOMIAL_ORDER + 1;
    let pml_elements = physics::PML.div_ceil(n_nodes);
    let cpml_cfg = DgCpmlConfig::with_axes(
        DgCpmlAxis::standard(pml_elements),
        DgCpmlAxis::standard(pml_elements),
        DgCpmlAxis::DISABLED,
    );
    let config = DGConfig {
        polynomial_order: DG_TENSOR_POLYNOMIAL_ORDER,
        sound_speed: physics::C0,
        // x, y: absorbing characteristic + CPML layer (finite open boundary).
        // z: periodic (NZ == 4 grid points → 1 DG element along z, effectively
        // a thin slab; absorbing characteristic on the single element would
        // dampen the z-invariant mode the focused-source produces, distorting
        // the focal field. Periodic z matches the existing DG-3D reference run
        // and the FDTD/PSTD slab convention.)
        boundary_conditions: [
            DgBoundaryCondition::AbsorbingCharacteristic,
            DgBoundaryCondition::AbsorbingCharacteristic,
            DgBoundaryCondition::Periodic,
        ],
        cpml: Some(cpml_cfg),
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, Arc::clone(&grid))?;
    let state_shape = solver.acoustic_tensor_state_shape()?;
    let mut state = Array3::<f64>::zeros(state_shape);
    let mut workspace = AcousticDgTensorWorkspace::new(state_shape);
    let mut memory = DgCpmlMemoryWorkspace::new(state_shape.0, state_shape.1);
    let profiles = DgCpmlProfiles::new(
        &cpml_cfg,
        physics::C0,
        [
            physics::NX / n_nodes,
            physics::NY / n_nodes,
            (nz / n_nodes).max(1),
        ],
        n_nodes,
        [
            n_nodes as f64 * physics::DX,
            n_nodes as f64 * physics::DX,
            n_nodes as f64 * physics::DX,
        ],
    )?;
    let sources = dg_tensor_sources(&solver, nz)?;
    let dt_sub = physics::DT / DG_TENSOR_SUBSTEPS_PER_STEP as f64;
    let mut pressure = Array3::<f64>::zeros((physics::NX, physics::NY, nz));

    let start = Instant::now();
    let mut peak = Array2::<f64>::zeros((physics::NX, physics::NY));
    for step in 0..physics::NT {
        for substep in 0..DG_TENSOR_SUBSTEPS_PER_STEP {
            let t = (step * DG_TENSOR_SUBSTEPS_PER_STEP + substep) as f64 * dt_sub;
            solver.step_acoustic_tensor_ssp_rk3_with_cpml_and_source(
                &mut state,
                physics::RHO0,
                dt_sub,
                &mut workspace,
                &mut memory,
                &profiles,
                t,
                |stage_t, rhs| apply_dg_tensor_source_rhs(rhs, &sources, stage_t),
            )?;
        }
        if (physics::GATE_START..=physics::GATE_END).contains(&step) {
            solver.project_acoustic_tensor_pressure_to_uniform_grid(&state, &mut pressure)?;
            update_peak(name, &mut peak, &pressure);
        }
    }

    Ok(SolverField {
        name,
        normalized_peak: super::metrics::normalize_map(&peak),
        elapsed: start.elapsed(),
    })
}

trait PressureGrid3 {
    fn nz(&self) -> usize;
    fn at(&self, i: usize, j: usize, k: usize) -> f64;
}

impl PressureGrid3 for leto::Array3<f64> {
    fn nz(&self) -> usize { self.shape()[2] }
    fn at(&self, i: usize, j: usize, k: usize) -> f64 { self[[i, j, k]] }
}

impl PressureGrid3 for leto::Array3<f64> {
    fn nz(&self) -> usize { self.shape()[2] }
    fn at(&self, i: usize, j: usize, k: usize) -> f64 { self[[i, j, k]] }
}

fn update_peak<P: PressureGrid3>(name: &'static str, peak: &mut Array2<f64>, pressure: &P) {
    let z = physics::FOCUS_Z.min(pressure.nz() - 1);
    for i in 0..physics::NX {
        for j in 0..physics::NY {
            let value = pressure.at(i, j, z);
            assert!(
                value.is_finite(),
                "{name} pressure field contains non-finite values"
            );
            peak[[i, j]] = peak[[i, j]].max(value.abs());
        }
    }
}

fn cpml_config() -> CPMLConfig {
    CPMLConfig::with_per_dimension_thickness(physics::PML, physics::PML, 1).with_alpha(2.0)
}

#[derive(Debug, Clone, Copy)]
struct DgTensorSource {
    elem: usize,
    node: usize,
    element: physics::Element,
    weight: f64,
}

fn dg_tensor_sources(solver: &DGSolver, nz: usize) -> Result<Vec<DgTensorSource>> {
    let elements = physics::elements();
    let apodization_sum = elements
        .iter()
        .map(|element| element.apodization)
        .sum::<f64>()
        .max(f64::EPSILON);
    let mut sources = Vec::new();
    let z_weight = (nz as f64 * apodization_sum).recip();
    for z in 0..nz {
        for &element in &elements {
            for source in solver.acoustic_tensor_cell_source_weights([element.x, element.y, z])? {
                sources.push(DgTensorSource {
                    elem: source.elem,
                    node: source.node,
                    element,
                    weight: z_weight * source.weight,
                });
            }
        }
    }
    Ok(sources)
}

fn apply_dg_tensor_source_rhs(rhs: &mut Array3<f64>, sources: &[DgTensorSource], t: f64) {
    for source in sources {
        rhs[(source.elem, source.node, ACOUSTIC_PRESSURE_VAR)] +=
            source.weight * physics::OMEGA * physics::element_pressure_at(&source.element, t);
    }
}

fn dg_source_weights(xi_nodes: &Array1<f64>, weights: &Array1<f64>) -> Vec<(usize, usize, f64)> {
    let mut entries = Vec::new();
    let mut integral = 0.0;
    for elem in 0..DG_ELEMENTS {
        for node in 0..xi_nodes.len() {
            let coordinate = dg_coordinate(elem, xi_nodes[node]);
            let scaled = (coordinate - physics::SOURCE_Y as f64) / DG_SOURCE_SIGMA_CELLS;
            let value = (-0.5 * scaled * scaled).exp();
            if value > 1.0e-8 {
                integral += weights[node] * value;
                entries.push((elem, node, value));
            }
        }
    }
    let norm = integral.max(f64::EPSILON);
    for (elem, node, value) in &mut entries {
        *value *= weights[*node] / norm;
        debug_assert!(*elem < DG_ELEMENTS);
    }
    entries
}

fn apply_dg_source(
    pressure: &mut Array3<f64>,
    source_weights: &[(usize, usize, f64)],
    t: f64,
    dt: f64,
) {
    let drive = physics::OMEGA * physics::focused_aperture_pressure_at(t);
    for &(elem, node, weight) in source_weights {
        pressure[(elem, node, 0)] += dt * drive * weight;
    }
}

fn sample_dg_pressure(pressure: &Array3<f64>, xi_nodes: &Array1<f64>, coordinate: f64) -> f64 {
    if coordinate <= 0.0 {
        return pressure[(0, 0, 0)];
    }
    let max_coordinate = 2.0 * DG_ELEMENTS as f64;
    if coordinate >= max_coordinate {
        return pressure[(DG_ELEMENTS - 1, xi_nodes.len() - 1, 0)];
    }
    let elem = ((coordinate / 2.0).floor() as usize).min(DG_ELEMENTS - 1);
    if coordinate.fract() == 0.0 && elem > 0 && (coordinate as usize).is_multiple_of(2) {
        let left = lagrange_value(pressure, xi_nodes, elem - 1, 1.0);
        let right = lagrange_value(pressure, xi_nodes, elem, -1.0);
        0.5 * (left + right)
    } else {
        let xi = coordinate - 2.0 * elem as f64 - 1.0;
        lagrange_value(pressure, xi_nodes, elem, xi)
    }
}

fn lagrange_value(pressure: &Array3<f64>, xi_nodes: &Array1<f64>, elem: usize, xi: f64) -> f64 {
    let mut value = 0.0;
    for node in 0..xi_nodes.len() {
        let mut basis = 1.0;
        for other in 0..xi_nodes.len() {
            if other != node {
                basis *= (xi - xi_nodes[other]) / (xi_nodes[node] - xi_nodes[other]);
            }
        }
        value += basis * pressure[(elem, node, 0)];
    }
    value
}

fn dg_coordinate(elem: usize, xi: f64) -> f64 {
    2.0 * elem as f64 + xi + 1.0
}

fn normalize_line(values: &[f64]) -> Vec<f64> {
    let max_value = values
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    values.iter().map(|value| value / max_value).collect()
}
