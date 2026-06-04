use super::{validate_diffusion_regime, OpticalForwardModel, OpticalSolveResult, OpticalWorkspace};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::GridDimensions;
use kwavers_imaging::photoacoustic::{
    IlluminationGeometry, OpticalModel, PhotoacousticScenario,
};
use kwavers_medium::optical_map::OpticalPropertyMap;
use kwavers_medium::properties::OpticalPropertyData;
use kwavers_solver::forward::optical::diffusion::{
    DiffusionBoundaryCondition, DiffusionBoundaryConditions, DiffusionSolver, DiffusionSolverConfig,
};
use ndarray::Array3;

/// Diffusion-approximation optical solver for the canonical photoacoustic vertical.
///
/// # Governing Equation
/// The absorbed optical fluence is approximated by the diffusion model
/// `-∇·(D∇Φ) + μ_a Φ = q`, with diffusion coefficient
/// `D = 1 / (3(μ_a + μ_s'))`.
///
/// # Numerical Realization
/// This implementation delegates the linear solve to the retained diffusion
/// solver and keeps stage-local source and fluence buffers in
/// [`OpticalWorkspace`].
///
/// # Validity Regime
/// The diffusion approximation is admitted only when `μ_s' > μ_a`; invalid
/// regimes are rejected before any solve is attempted.
#[derive(Debug, Default)]
pub struct DiffusionOpticalSolver;

impl DiffusionOpticalSolver {
    fn source_term(
        scenario: &PhotoacousticScenario,
        dimensions: GridDimensions,
        workspace: &mut OpticalWorkspace,
    ) -> KwaversResult<()> {
        workspace.source.fill(0.0);
        let amplitude = scenario.config.incident_fluence_j_m2 / scenario.grid.cell_volume();
        let origin = match scenario.config.illumination {
            IlluminationGeometry::PencilBeam { origin_m, .. } => origin_m,
            IlluminationGeometry::IsotropicPoint { origin_m } => origin_m,
        };
        let (i, j, k) = scenario
            .grid
            .coordinates_to_indices(origin[0], origin[1], origin[2])
            .ok_or_else(|| {
                KwaversError::InvalidInput(
                    "illumination origin lies outside the computational grid".to_owned(),
                )
            })?;
        workspace.source[[
            i.min(dimensions.nx - 1),
            j.min(dimensions.ny - 1),
            k.min(dimensions.nz - 1),
        ]] = amplitude;
        Ok(())
    }
}

impl OpticalForwardModel for DiffusionOpticalSolver {
    fn model_kind(&self) -> OpticalModel {
        OpticalModel::Diffusion
    }

    fn solve(&self, scenario: &PhotoacousticScenario) -> KwaversResult<OpticalSolveResult> {
        validate_diffusion_regime(scenario)?;
        let mut workspace = OpticalWorkspace::new(scenario.optical_map.dimensions);
        let optical_properties = optical_map_to_array(&scenario.optical_map);
        let config = DiffusionSolverConfig {
            max_iterations: 10_000,
            tolerance: 1e-8,
            boundary_parameter: 2.0,
            boundary_conditions: Some(DiffusionBoundaryConditions {
                x_min: DiffusionBoundaryCondition::ZeroFlux,
                x_max: DiffusionBoundaryCondition::ZeroFlux,
                y_min: DiffusionBoundaryCondition::ZeroFlux,
                y_max: DiffusionBoundaryCondition::ZeroFlux,
                z_min: DiffusionBoundaryCondition::Extrapolated { a: 2.0 },
                z_max: DiffusionBoundaryCondition::ZeroFlux,
            }),
            verbose: false,
        };
        let solver = DiffusionSolver::new(scenario.grid.clone(), optical_properties, config)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        Self::source_term(scenario, scenario.optical_map.dimensions, &mut workspace)?;
        workspace.fluence = solver
            .solve(&workspace.source)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        Ok(OpticalSolveResult {
            model: OpticalModel::Diffusion,
            fluence: workspace.fluence,
        })
    }
}

fn optical_map_to_array(optical_map: &OpticalPropertyMap) -> Array3<OpticalPropertyData> {
    let dims = optical_map.dimensions;
    Array3::from_shape_fn((dims.nx, dims.ny, dims.nz), |(i, j, k)| {
        optical_map
            .get_properties(i, j, k)
            .expect("optical map dimensions are internally consistent")
    })
}
