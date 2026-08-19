//! Bounded single-bubble sonoluminescence simulation.
//!
//! The example runs the integrated Keller–Miksis/thermodynamics/emission path
//! on one cell and reports the typed blackbody and bremsstrahlung components.
//! Cherenkov output is queried separately through the arbitrary-unit spectrum;
//! it is not added to the dimensioned power-density field.

use kwavers_physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use kwavers_physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use kwavers_physics::optics::sonoluminescence::{EmissionParameters, IntegratedSonoluminescence};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bubble_params = BubbleParameters {
        r0: 10e-6,
        t0: 300.0,
        initial_gas_pressure: 101_325.0,
        ..Default::default()
    };
    let emission_params = EmissionParameters {
        use_cherenkov: true,
        ..Default::default()
    };
    let bubble_model = KellerMiksisModel::new(bubble_params.clone());
    let mut simulation =
        IntegratedSonoluminescence::new([1, 1, 1], bubble_params.clone(), emission_params);
    simulation.set_acoustic_pressure(leto::Array3::from_elem([1, 1, 1], 1.0e5));

    const STEPS: usize = 8;
    const DT: f64 = 5.0e-9;
    for step in 0..STEPS {
        let time = step as f64 * DT;
        simulation.simulate_step(DT, time, &bubble_params, &bubble_model)?;

        let temperature = simulation.temperature_field[[0, 0, 0]];
        let radius = simulation.radius_field[[0, 0, 0]];
        let charge_density = simulation.charge_density_field[[0, 0, 0]];
        let components =
            simulation
                .emission
                .components_at_point(temperature, radius, charge_density);
        println!(
            "step={step} temperature={temperature:.3} K radius={radius:.6e} m blackbody={:.6e} W/m³ bremsstrahlung={:.6e} W/m³",
            components.blackbody().into_base(),
            components.bremsstrahlung().into_base(),
        );
    }

    let spectrum = simulation.emission.calculate_spectrum_at_point(
        simulation.temperature_field[[0, 0, 0]],
        simulation.pressure_field[[0, 0, 0]],
        simulation.radius_field[[0, 0, 0]],
        simulation.particle_velocity_field[[0, 0, 0]],
        simulation.charge_density_field[[0, 0, 0]],
        simulation.compression_field[[0, 0, 0]],
    );
    println!(
        "spectral samples={} arbitrary-unit total={:.6e} peak={:.6e} m",
        spectrum.intensities.len(),
        spectrum.total_intensity(),
        spectrum.peak_wavelength(),
    );

    Ok(())
}
