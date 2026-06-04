use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::error::KwaversResult;
use kwavers_imaging::photoacoustic::{
    PhotoacousticScenario, PhotoacousticSimulation, PhotoacousticValidationReport,
};
use kwavers_physics::photoacoustics::thermoelasticity::GrueneisenModel;
use kwavers_physics::photoacoustics::ConfinementAssessment;
/// Validate photoacoustic simulation.
/// # Errors
/// - Propagates invalid confinement-domain parameters.
///
/// # Panics
/// - Panics if `optical map dimensions are internally consistent`.
///
pub fn validate_photoacoustic_simulation(
    scenario: &PhotoacousticScenario,
    simulation: &PhotoacousticSimulation,
) -> KwaversResult<PhotoacousticValidationReport> {
    let center = scenario
        .optical_map
        .get_properties(
            scenario.grid.nx / 2,
            scenario.grid.ny / 2,
            scenario.grid.nz / 2,
        )
        .expect("optical map dimensions are internally consistent");
    let confinement = ConfinementAssessment::evaluate(
        center.absorption_coefficient,
        scenario.config.pulse_duration_s,
        scenario.config.thermoelastic,
    )?;
    let pressure_balance_error = simulation
        .optical_fluence
        .iter()
        .zip(simulation.initial_pressure.pressure.iter())
        .map(|(fluence, pressure)| {
            // Use canonical GrueneisenModel at body temperature for comparison.
            // The simulation was generated with the same constant-Γ assumption.
            let gamma = GrueneisenModel::water().evaluate(BODY_TEMPERATURE_C);
            let expected = gamma * center.absorption_coefficient * *fluence;
            (expected - *pressure).abs()
        })
        .sum::<f64>();
    let denominator = simulation
        .initial_pressure
        .max_pressure
        .max(f64::MIN_POSITIVE)
        * simulation.initial_pressure.pressure.len() as f64;

    Ok(PhotoacousticValidationReport {
        optical_model: format!("{:?}", scenario.config.optical_model),
        wavelength_nm: scenario.wavelength_nm,
        stress_confined: confinement.stress_confined,
        thermal_confined: confinement.thermal_confined,
        total_optical_energy: simulation.optical_fluence.sum() * scenario.grid.cell_volume(),
        max_initial_pressure: simulation.initial_pressure.max_pressure,
        relative_pressure_balance_error: pressure_balance_error / denominator,
    })
}
