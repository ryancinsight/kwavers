use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::imaging::photoacoustic::PhotoacousticScenario;

/// Deterministic validation case descriptor for the optical stage.
#[derive(Debug, Clone)]
pub struct OpticalValidationCase {
    pub name: &'static str,
    pub wavelength_nm: f64,
    pub expected_regime: &'static str,
}

/// Validate the diffusion approximation regime.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn validate_diffusion_regime(scenario: &PhotoacousticScenario) -> KwaversResult<()> {
    let props = scenario
        .optical_map
        .get_properties(
            scenario.grid.nx / 2,
            scenario.grid.ny / 2,
            scenario.grid.nz / 2,
        )
        .ok_or_else(|| {
            KwaversError::InvalidInput("failed to sample optical properties".to_owned())
        })?;
    let mu_a = props.absorption_coefficient;
    let mu_s_prime = props.reduced_scattering();
    if mu_s_prime <= mu_a {
        return Err(KwaversError::InvalidInput(format!(
            "diffusion model invalid for mu_s' <= mu_a (mu_s'={mu_s_prime:.6}, mu_a={mu_a:.6})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_diffusion_regime;
    use kwavers_grid::{Grid, GridDimensions};
    use kwavers_domain::imaging::photoacoustic::{
        IlluminationGeometry, MonteCarloModelConfig, OpticalModel, PhotoacousticAcousticConfig,
        PhotoacousticScenario, PhotoacousticSolverConfig, ThermoelasticProperties,
    };
    use kwavers_domain::medium::optical_map::OpticalPropertyMap;
    use kwavers_domain::medium::properties::OpticalPropertyData;

    #[test]
    fn rejects_invalid_diffusion_regime() {
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let dims = GridDimensions::from_grid(&grid);
        let optical_map = OpticalPropertyMap::homogeneous(
            &OpticalPropertyData {
                absorption_coefficient: 2.0,
                scattering_coefficient: 0.5,
                anisotropy: 0.1,
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
                    speed_of_sound_m_s: kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM,
                    cfl_factor: 0.2,
                    num_time_steps: 8,
                    snapshot_interval: 2,
                },
                thermoelastic: ThermoelasticProperties {
                    density_kg_m3: kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL,
                    sound_speed_m_s: kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM,
                    specific_heat_j_kgk:
                        kwavers_core::constants::thermodynamic::SPECIFIC_HEAT_WATER_37C,
                    thermal_conductivity_w_mk:
                        kwavers_core::constants::thermodynamic::THERMAL_CONDUCTIVITY_WATER_37C,
                },
                monte_carlo: MonteCarloModelConfig::default(),
            },
            vec![[0.002, 0.002, 0.006]],
        )
        .unwrap();

        assert!(validate_diffusion_regime(&scenario).is_err());
    }
}
