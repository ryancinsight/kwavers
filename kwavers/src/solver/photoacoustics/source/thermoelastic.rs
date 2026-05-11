use super::{validate_source_generation, SourceWorkspace};
use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{InitialPressure, PhotoacousticScenario};
use crate::physics::photoacoustics::thermoelasticity::GrueneisenModel;
use crate::physics::photoacoustics::PhotoacousticGoverningEquations;
use ndarray::Array3;

/// Canonical thermoelastic source model with `p0 = Gamma * mu_a * Phi`.
///
/// # Governing Equation
/// Under stress and thermal confinement, the retained source relation is
/// `p0 = Γ μ_a Φ`, where `Γ` is the Gruneisen parameter, `μ_a` the absorption
/// coefficient, and `Φ` the absorbed optical fluence.
///
/// # Numerical Realization
/// The implementation evaluates the governing relation cell-by-cell over the
/// optical map and stores the result in a stage-local [`SourceWorkspace`].
#[derive(Debug, Default)]
pub struct PhotoacousticSourceModel;

impl PhotoacousticSourceModel {
    /// Compute initial pressure.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `optical map dimensions are internally consistent`.
    ///
    pub fn compute_initial_pressure(
        &self,
        scenario: &PhotoacousticScenario,
        fluence: &Array3<f64>,
    ) -> KwaversResult<InitialPressure> {
        let dims = scenario.optical_map.dimensions;
        let mut workspace = SourceWorkspace::new(dims);
        let mut max_pressure = 0.0_f64;

        for ((i, j, k), cell_pressure) in workspace.pressure.indexed_iter_mut() {
            let props = scenario
                .optical_map
                .get_properties(i, j, k)
                .expect("optical map dimensions are internally consistent");
            // Use tissue-specific GrueneisenModel at body temperature (37 °C).
            // The scenario's thermoelastic properties define confinement geometry;
            // Γ(T) is evaluated via the canonical temperature-dependent model.
            let gruneisen = GrueneisenModel::soft_tissue();
            let report = PhotoacousticGoverningEquations::initial_pressure(
                props.absorption_coefficient,
                fluence[[i, j, k]],
                scenario.config.pulse_duration_s,
                scenario.config.thermoelastic,
                &gruneisen,
                37.0,
            );
            *cell_pressure = report.initial_pressure_pa;
            max_pressure = max_pressure.max(report.initial_pressure_pa);
        }

        let initial_pressure = InitialPressure {
            pressure: workspace.pressure,
            max_pressure,
            fluence: fluence.clone(),
        };
        validate_source_generation(scenario, &initial_pressure)?;
        Ok(initial_pressure)
    }
}
