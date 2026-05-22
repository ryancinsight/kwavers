use crate::domain::imaging::photoacoustic::PhotoacousticScenario;

/// Validation descriptor for acoustic propagation.
#[derive(Debug, Clone)]
pub struct AcousticValidationCase {
    pub name: &'static str,
    pub cfl_limit: f64,
}

#[must_use]
pub fn compute_time_step(scenario: &PhotoacousticScenario) -> f64 {
    let speed_of_sound = scenario.config.acoustic.speed_of_sound_m_s;
    let min_h = scenario.grid.dx.min(scenario.grid.dy).min(scenario.grid.dz);
    scenario.config.acoustic.cfl_factor * min_h / speed_of_sound
}

#[cfg(test)]
mod tests {
    use super::compute_time_step;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use crate::core::constants::thermodynamic::SPECIFIC_HEAT_WATER_37C;
    use crate::domain::grid::{Grid, GridDimensions};
    use crate::domain::imaging::photoacoustic::{
        IlluminationGeometry, MonteCarloModelConfig, OpticalModel, PhotoacousticAcousticConfig,
        PhotoacousticScenario, PhotoacousticSolverConfig, ThermoelasticProperties,
    };
    use crate::domain::medium::optical_map::OpticalPropertyMap;
    use crate::domain::medium::properties::OpticalPropertyData;

    #[test]
    fn computes_cfl_time_step_from_minimum_spacing() {
        let grid = Grid::new(8, 8, 8, 1e-3, 2e-3, 3e-3).unwrap();
        let dims = GridDimensions::from_grid(&grid);
        let optical_map = OpticalPropertyMap::homogeneous(
            &OpticalPropertyData {
                absorption_coefficient: 0.5,
                scattering_coefficient: 120.0,
                anisotropy: 0.9,
                refractive_index: 1.4,
            },
            dims,
        );
        let scenario = PhotoacousticScenario::new(
            grid,
            750.0,
            optical_map,
            PhotoacousticSolverConfig {
                optical_model: OpticalModel::Diffusion,
                illumination: IlluminationGeometry::IsotropicPoint {
                    origin_m: [0.003, 0.003, 0.003],
                },
                pulse_duration_s: 8e-9,
                incident_fluence_j_m2: 10.0,
                acoustic: PhotoacousticAcousticConfig {
                    speed_of_sound_m_s: SOUND_SPEED_WATER_SIM,
                    cfl_factor: 0.25,
                    num_time_steps: 8,
                    snapshot_interval: 2,
                },
                thermoelastic: ThermoelasticProperties {
                    density_kg_m3: DENSITY_WATER_NOMINAL,
                    sound_speed_m_s: SOUND_SPEED_WATER_SIM,
                    specific_heat_j_kgk: SPECIFIC_HEAT_WATER_37C,
                    thermal_conductivity_w_mk: 0.6,
                },
                monte_carlo: MonteCarloModelConfig::default(),
            },
            vec![[0.002, 0.002, 0.006]],
        )
        .unwrap();

        let dt = compute_time_step(&scenario);
        assert!((dt - (0.25e-3 / SOUND_SPEED_WATER_SIM)).abs() < 1e-15);
    }
}
