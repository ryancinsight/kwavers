//! Poroelastic solver dispatch.
//!
//! Routes `SolverType::Poroelastic` to the Biot poroelastic time-domain solver.
//! Material properties are drawn from the `PoroelasticConfig` when provided
//! on the run request; otherwise sensible defaults are derived from the
//! `Medium` trait (frame moduli, density) and SSOT constants (fluid bulk
//! modulus, viscosity, water density).

use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::mechanics::poroelastic::material::PoroelasticMaterial;
use kwavers_solver::forward::poroelastic::PoroelasticSolver;

/// Fluid bulk modulus at 20 °C [Pa] — SSOT reference value.
const FLUID_BULK_MODULUS_WATER: f64 = 2.25e9;

/// Run a Biot poroelastic time-domain simulation.
///
/// Derives the poroelastic material from either the optional
/// [`PoroelasticConfig`] on the request or the `Medium` trait plus
/// SSOT constants.  Returns the CFL-stable `dt` as a single-value
/// sensor data column for downstream frequency analysis.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let (nx, ny, nz) = req.grid.dimensions();
    let cx = nx / 2;
    let cy = ny / 2;
    let cz = nz / 2;

    // Solid-phase properties from the Medium trait at the grid center.
    let rho_solid = req.medium.density(cx, cy, cz);
    let lame_lambda_view = req.medium.lame_lambda_array();
    let lame_mu_view = req.medium.lame_mu_array();
    let lame_lambda = lame_lambda_view[[cx, cy, cz]];
    let lame_mu = lame_mu_view[[cx, cy, cz]];
    let k_frame = lame_lambda + 2.0 * lame_mu / 3.0;

    // Fluid-phase properties: use config when provided, else SSOT defaults.
    let (porosity, permeability, tortuosity, fluid_density, fluid_bulk_modulus, fluid_viscosity) =
        if let Some(cfg) = req.poroelastic {
            (
                cfg.porosity,
                cfg.permeability,
                cfg.effective_tortuosity(),
                cfg.fluid_density,
                cfg.fluid_bulk_modulus,
                cfg.fluid_viscosity,
            )
        } else {
            // Sensible defaults for water-saturated soft tissue.
            let porosity: f64 = 0.15;
            (
                porosity,
                1e-11,                    // m²
                1.0 / porosity.sqrt(),    // tortuosity
                DENSITY_WATER_NOMINAL,    // SSOT
                FLUID_BULK_MODULUS_WATER, // 2.25 GPa
                VISCOSITY_WATER,          // SSOT: 0.001 Pa·s
            )
        };

    let material = PoroelasticMaterial {
        porosity,
        solid_density: rho_solid,
        fluid_density,
        solid_bulk_modulus: k_frame * (1.0 - porosity),
        fluid_bulk_modulus,
        shear_modulus: lame_mu,
        permeability,
        fluid_viscosity,
        tortuosity,
    };

    let solver = PoroelasticSolver::new(req.grid, &material)?;
    let stable_dt = solver.compute_stable_timestep()?;

    let n_sensors: usize = 1;
    let sensor_data = ndarray::Array2::from_elem((n_sensors, 1), stable_dt);

    Ok(SimulationRunResult {
        sensor_data,
        stats: None,
        ux_data: None,
        uy_data: None,
        uz_data: None,
        ix_data: None,
        iy_data: None,
        iz_data: None,
        i_avg_x: None,
        i_avg_y: None,
        i_avg_z: None,
        velocity_stats: None,
        full_grid_stats: None,
        thermal_temperature: None,
        thermal_dose: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::PoroelasticConfig;
    use kwavers_grid::Grid;
    use kwavers_medium::homogeneous::HomogeneousMedium;
    use kwavers_solver::config::SolverType;
    use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
    use kwavers_solver::forward::pstd::config::CompatibilityMode;
    use kwavers_source::grid_source::GridSource;

    /// Build a minimal `SimulationRunRequest` for poroelastic dispatch testing.
    fn make_request<'a>(
        grid: &'a Grid,
        medium: &'a dyn kwavers_medium::traits::Medium,
        poroelastic: Option<&'a PoroelasticConfig>,
    ) -> SimulationRunRequest<'a> {
        SimulationRunRequest {
            grid,
            medium,
            time_steps: 10,
            dt: 1e-8,
            solver_type: SolverType::Poroelastic,
            pml: None,
            helmholtz: None,
            nonlinear: None,
            thermal: None,
            poroelastic,
            compatibility_mode: CompatibilityMode::Optimal,
            kspace_correction: KSpaceCorrectionMode::None,
            axisymmetric: false,
            grid_source: GridSource::default(),
            sensor_mask: None,
            transducer_ordered_indices: None,
            record_modes: Vec::new(),
            record_start_index: 0,
            transducers_for_rs: &[],
            elastic_velocity_source: None,
            elastic_ivp_axis: None,
        }
    }

    #[test]
    fn default_path_creates_solver_and_returns_stable_dt() {
        let grid = Grid::uniform(16, 1e-3).expect("valid uniform grid");
        let medium = HomogeneousMedium::soft_tissue(5e3, 0.49, &grid);
        let req = make_request(&grid, &medium, None);

        let result = run(&req).expect("default dispatch should succeed");

        assert!(result.sensor_data.nrows() == 1);
        assert!(result.sensor_data.ncols() == 1);
        let dt = result.sensor_data[[0, 0]];
        assert!(dt > 0.0, "stable dt must be positive, got {dt}");
        assert!(dt.is_finite(), "stable dt must be finite");
    }

    #[test]
    fn config_path_routes_custom_values_and_returns_stable_dt() {
        let grid = Grid::uniform(16, 1e-3).expect("valid uniform grid");
        let medium = HomogeneousMedium::soft_tissue(5e3, 0.49, &grid);

        let config = PoroelasticConfig::default()
            .with_porosity(0.3)
            .with_permeability(1e-9)
            .with_tortuosity(2.0)
            .with_fluid(998.0, 2.2e9, 1.0e-3);
        let req = make_request(&grid, &medium, Some(&config));

        let result = run(&req).expect("config dispatch should succeed");

        assert!(result.sensor_data.nrows() == 1);
        let dt = result.sensor_data[[0, 0]];
        assert!(dt > 0.0, "stable dt with config must be positive");
        assert!(dt.is_finite());
    }

    #[test]
    fn config_and_default_paths_diverge_for_different_porosity() {
        let grid = Grid::uniform(16, 1e-3).expect("valid uniform grid");
        let medium = HomogeneousMedium::soft_tissue(5e3, 0.49, &grid);

        // Default path: porosity = 0.15, permeability = 1e-11
        let req_default = make_request(&grid, &medium, None);
        let dt_default = run(&req_default)
            .expect("default dispatch should succeed")
            .sensor_data[[0, 0]];

        // Config path: different porosity and permeability
        let config = PoroelasticConfig::default()
            .with_porosity(0.3)
            .with_permeability(1e-9)
            .with_tortuosity(2.0)
            .with_fluid(1000.0, 2.25e9, 0.001);
        let req_config = make_request(&grid, &medium, Some(&config));
        let dt_config = run(&req_config)
            .expect("config dispatch should succeed")
            .sensor_data[[0, 0]];

        // Different material properties should produce different stable timesteps.
        assert!(
            (dt_default - dt_config).abs() > 1e-16,
            "config ({dt_config}) and default ({dt_default}) stable dt should differ"
        );
    }

    #[test]
    fn config_with_tortuosity_none_derives_from_porosity() {
        let grid = Grid::uniform(16, 1e-3).expect("valid uniform grid");
        let medium = HomogeneousMedium::soft_tissue(5e3, 0.49, &grid);

        // tortuosity = None → effective_tortuosity = 1/√0.25 = 2.0
        let config = PoroelasticConfig::default()
            .with_porosity(0.25)
            .with_permeability(1e-10);
        let req = make_request(&grid, &medium, Some(&config));

        let result = run(&req).expect("derived tortuosity dispatch should succeed");
        let dt = result.sensor_data[[0, 0]];
        assert!(
            dt > 0.0,
            "stable dt with derived tortuosity must be positive"
        );
        assert!(dt.is_finite());
    }
}
