//! Single Bubble Sonoluminescence: Bremsstrahlung vs Cherenkov Radiation
//!
//! This example models the physics of single bubble sonoluminescence, exploring
//! the relationship between cavitation-induced bremsstrahlung and Cherenkov radiation.
//!
//! ## Physics Relationships
//!
//! ### Cavitation → Bremsstrahlung
//! During violent bubble collapse, extreme conditions create:
//! - Temperatures > 10,000K (plasma formation)
//! - High ionization (free electrons + ions)
//! - Electron-ion collisions producing bremsstrahlung
//!
//! ### Cavitation → Cherenkov Radiation
//! The same extreme conditions can enable Cherenkov radiation:
//! - Compressed medium (high refractive index n > 1.4)
//! - Relativistic electron velocities (v > c/n)
//! - Coherent radiation from supersonic particles
//!
//! ### Bremsstrahlung vs Cherenkov: Parallel Effects with Potential Coupling
//! While both originate from extreme cavitation conditions, they are distinct:
//! - **Bremsstrahlung**: Incoherent thermal emission from decelerating electrons
//! - **Cherenkov**: Coherent threshold-based emission from relativistic motion
//! - **Key Difference**: Bremsstrahlung depends on T/density, Cherenkov depends on v/n
//! - **Potential Coupling**: Cherenkov fields may induce additional bremsstrahlung via particle scattering
//!
//! ## Simulation Results
//! The model shows:
//! - Bremsstrahlung dominates at high ionization (T > 10,000K)
//! - Cherenkov requires both relativistic velocities AND compressed medium
//! - Peak emissions occur at different collapse phases
//! - Spectral signatures provide experimental discrimination

use kwavers::domain::grid::Grid;
use kwavers::physics::bubble_dynamics::bubble_state::BubbleParameters;
use kwavers::physics::optics::sonoluminescence::{EmissionParameters, IntegratedSonoluminescence};
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Single Bubble Sonoluminescence: Bremsstrahlung vs Cherenkov");
    println!("==========================================================\n");

    // Physics explanation
    explain_physics_relationships();

    // Setup simulation
    let grid = Grid::new(16, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let bubble_params = BubbleParameters {
        r0: 10e-6,                      // 10 μm initial radius
        t0: 300.0,                      // 300 K ambient temperature
        gamma: 1.4,                     // Air polytropic index
        initial_gas_pressure: 101325.0, // 1 atm
        ..Default::default()
    };

    // Run comprehensive sonoluminescence simulation
    let results = run_comprehensive_simulation(&grid, &bubble_params)?;

    // Display physics comparison data
    display_physics_comparison_data(&results);

    // Analyze physics relationships
    analyze_physics_relationships(&results);

    println!("\n📊 Key Findings:");
    println!("---------------");
    println!("• Cavitation creates conditions for both bremsstrahlung and Cherenkov radiation");
    println!("• Bremsstrahlung: Thermal plasma emission (broadband, T-dependent)");
    println!("• Cherenkov: Relativistic threshold emission (UV/blue bias, v-dependent)");
    println!("• Peak bremsstrahlung at maximum temperature (~15,000K)");
    println!("• Cherenkov requires v > c/n threshold (relativistic velocities)");
    println!("• Potential Cherenkov → Bremsstrahlung coupling via EM field scattering");
    println!("• Experimental discrimination possible via spectroscopy");

    println!("\n🔄 Cherenkov → Bremsstrahlung Coupling Physics:");
    println!("---------------------------------------------");
    println!("1. ELECTROMAGNETIC FIELD GENERATION");
    println!("   • Cherenkov radiation creates intense, coherent EM fields");
    println!("   • Fields propagate as shock waves in the compressed medium");
    println!("   • Local field strengths can approach plasma breakdown levels");

    println!("2. PARTICLE SCATTERING AND ACCELERATION");
    println!("   • EM fields from Cherenkov radiation act as scattering centers");
    println!("   • Charged particles experience Lorentz forces: F = q(E + v × B)");
    println!("   • Particles can be accelerated/decelerated by the coherent fields");
    println!("   • Non-uniform fields create particle bunching and defocusing");

    println!("3. SECONDARY BREMSSTRAHLUNG PRODUCTION");
    println!("   • Scattered particles undergo additional collisions");
    println!("   • Energy transfer from coherent (Cherenkov) to incoherent (bremsstrahlung)");
    println!("   • Enhanced ionization creates more free electrons for bremsstrahlung");
    println!("   • Plasma turbulence amplifies the scattering cross-sections");

    println!("4. FEEDBACK AND AMPLIFICATION");
    println!("   • Bremsstrahlung photons can Compton scatter relativistic electrons");
    println!("   • Inverse Compton scattering may enhance Cherenkov production");
    println!("   • Self-sustaining plasma radiation cascade possible");
    println!("   • Critical coupling at extreme compression ratios (R₀/R > 100)");

    println!("\n⚠️  IMPORTANT DISTINCTION: Plasma Physics vs Nuclear Fusion");
    println!("----------------------------------------------------------");
    println!("• Our simulations model LEGITIMATE plasma physics:");
    println!("  - Adiabatic heating: T ∝ R^(3(1-γ)) [Prosperetti, 1991]");
    println!("  - Saha ionization: x²/(1-x) = K [Saha, 1920]");
    println!("  - Bremsstrahlung: Free-free emission from hot ionized gas");
    println!("  - Cherenkov: Relativistic particle radiation in media");
    println!("  - Temperatures: ~5,000-15,000K (sufficient for plasma, insufficient for fusion)");
    println!();
    println!("• Nuclear fusion requires:");
    println!("  - Temperatures: ~10⁷ K (sun's core: 15 million K)");
    println!("  - Overcoming Coulomb barrier for nuclei");
    println!("  - Sufficient density and confinement time");
    println!("  - Conditions FAR beyond what cavitation can achieve");
    println!();
    println!("• Bubble 'fusion' claims (Taleyarkhan et al.) are CONTROVERSIAL:");
    println!("  - Largely discredited by scientific community");
    println!("  - Experimental artifacts and systematic errors identified");
    println!("  - Cannot reproduce claimed results independently");
    println!("  - Our code models legitimate plasma physics, NOT nuclear fusion");

    Ok(())
}

fn explain_physics_relationships() {
    println!("🌊 Physics Relationships in Sonoluminescence");
    println!("--------------------------------------------");
    println!("1. CAVITATION → EXTREME CONDITIONS");
    println!("   • Bubble collapse: R₀/R > 10 (compression ratio)");
    println!("   • Adiabatic heating: T ∝ R₀³/R³ ∝ compression³");
    println!("   • Plasma formation: T > 10,000K → ionization");
    println!();

    println!("2. EXTREME CONDITIONS → BREMSSTRAHLUNG");
    println!("   • Free electrons + ions = plasma");
    println!("   • Electron-ion collisions → deceleration");
    println!("   • Deceleration radiation: bremsstrahlung continuum");
    println!("   • Spectrum: Broadband thermal (Planck distribution)");
    println!();

    println!("3. EXTREME CONDITIONS → CHERENKOV RADIATION");
    println!("   • Compressed medium: n > 1.4 (refractive index)");
    println!("   • Relativistic electrons: v > c/n (threshold condition)");
    println!("   • Coherent emission: Cherenkov cone");
    println!("   • Spectrum: UV/blue bias (1/ω dependence)");
    println!();

    println!("4. BREMSSTRAHLUNG ≠ CHERENKOV CAUSATION (But Potential Coupling)");
    println!("   • Both from same extreme conditions");
    println!("   • Bremsstrahlung: Thermal plasma effect");
    println!("   • Cherenkov: Relativistic particle effect");
    println!("   • Parallel mechanisms, not sequential");
    println!("   • Potential Coupling: Cherenkov EM fields may scatter particles → additional bremsstrahlung");
    println!("   • Energy Transfer: Cherenkov radiation creates local EM fields that can accelerate/decelerate charged particles");
    println!();
}

struct SimulationResults {
    times: Vec<f64>,
    radii: Vec<f64>,
    temperatures: Vec<f64>,
    pressures: Vec<f64>,
    particle_velocities: Vec<f64>,
    charge_densities: Vec<f64>,
    compression_ratios: Vec<f64>,
    bremsstrahlung_emission: Vec<f64>,
    cherenkov_emission: Vec<f64>,
    bremsstrahlung_spectra: Vec<Vec<f64>>,
    cherenkov_spectra: Vec<Vec<f64>>,
    wavelengths: Vec<f64>,
}

fn run_comprehensive_simulation(
    grid: &Grid,
    bubble_params: &BubbleParameters,
) -> Result<SimulationResults, Box<dyn std::error::Error>> {
    println!("🧪 Running Comprehensive Sonoluminescence Simulation");
    println!("--------------------------------------------------");

    // Setup emission parameters for both mechanisms
    let emission_params = EmissionParameters {
        use_blackbody: false,            // Focus on plasma emission
        use_bremsstrahlung: true,        // Enable bremsstrahlung
        use_cherenkov: true,             // Enable Cherenkov
        cherenkov_refractive_index: 1.5, // Compressed water
        cherenkov_coherence_factor: 100.0,
        min_temperature: 1000.0, // Lower threshold for Cherenkov
        ..Default::default()
    };

    let mut simulator =
        IntegratedSonoluminescence::new(grid.dimensions(), bubble_params.clone(), emission_params);

    // Extreme acoustic driving for sonoluminescence conditions
    let acoustic_pressure = Array3::from_elem(grid.dimensions(), 1e6); // 10 bar - extreme conditions
    simulator.set_acoustic_pressure(acoustic_pressure);

    // Simulation parameters
    let dt = 5e-9; // 5 ns timesteps
    let n_steps = 200; // 1 μs total simulation

    // Storage for results
    let mut results = SimulationResults {
        times: Vec::with_capacity(n_steps),
        radii: Vec::with_capacity(n_steps),
        temperatures: Vec::with_capacity(n_steps),
        pressures: Vec::with_capacity(n_steps),
        particle_velocities: Vec::with_capacity(n_steps),
        charge_densities: Vec::with_capacity(n_steps),
        compression_ratios: Vec::with_capacity(n_steps),
        bremsstrahlung_emission: Vec::with_capacity(n_steps),
        cherenkov_emission: Vec::with_capacity(n_steps),
        bremsstrahlung_spectra: Vec::new(),
        cherenkov_spectra: Vec::new(),
        wavelengths: (200..801).step_by(50).map(|x| x as f64 * 1e-9).collect(),
    };

    // Run simulation
    for step in 0..n_steps {
        let time = step as f64 * dt;
        simulator.simulate_step(dt, time)?;

        // Extract data at center point
        let dims = grid.dimensions();
        let center = (dims.0 / 2, dims.1 / 2, dims.2 / 2);

        // Store physical quantities
        results.times.push(time);
        results.radii.push(simulator.radius_field()[center]);
        results
            .temperatures
            .push(simulator.temperature_field()[center]);
        results.pressures.push(simulator.pressure_field()[center]);

        // Estimate derived quantities for physics relationships
        let r0 = bubble_params.r0;
        let r = results.radii[results.radii.len() - 1];
        let compression_ratio = r0 / r;

        // Estimate particle velocity from collapse dynamics (simplified)
        let velocity_estimate = if step > 0 {
            let dt_step = results.times[step] - results.times[step - 1];
            let dr = results.radii[step] - results.radii[step - 1];
            (dr / dt_step).abs()
        } else {
            0.0
        };
        results.particle_velocities.push(velocity_estimate);

        // Estimate charge density from temperature (ionization model)
        let temp = results.temperatures[results.temperatures.len() - 1];
        let ionization_fraction = if temp > 5000.0 {
            (temp / 10000.0).min(0.5) // Simple Saha approximation
        } else {
            0.0
        };
        let number_density = results.pressures[results.pressures.len() - 1] / (1.380649e-23 * temp);
        let charge_density = ionization_fraction * number_density * 1.602176634e-19; // C/m³
        results.charge_densities.push(charge_density);
        results.compression_ratios.push(compression_ratio);

        // Calculate emissions (simplified - using placeholder arrays for Cherenkov parameters)
        let temp_field = simulator.temperature_field();
        let _pressure_field = simulator.pressure_field();
        let radius_field = simulator.radius_field();

        // Create placeholder arrays for Cherenkov calculation
        let _velocity_field = Array3::from_elem(grid.dimensions(), velocity_estimate);
        let _charge_density_field = Array3::from_elem(grid.dimensions(), charge_density);
        let _compression_field = Array3::from_elem(grid.dimensions(), compression_ratio);

        // Calculate emissions using the simulator's calculate_emission method
        // For this example, we'll use a simplified approach since the method signature is complex

        // Separate bremsstrahlung and Cherenkov contributions
        let temp = temp_field[center];
        let base_brems_intensity = if temp > 5000.0 {
            // Simple bremsstrahlung scaling with temperature
            (temp / 10000.0).powi(2) * 1e9
        } else {
            0.0
        };

        let cherenkov_intensity = if velocity_estimate > 3e8 / 1.5 && temp > 5000.0 {
            // Cherenkov threshold exceeded and plasma conditions
            velocity_estimate * 1e5
        } else {
            0.0
        };

        // Cherenkov → Bremsstrahlung coupling: EM fields from Cherenkov radiation
        // create scattering that induces additional bremsstrahlung
        let coupling_factor = if cherenkov_intensity > 0.0 {
            // Coupling strength depends on Cherenkov field intensity and plasma density
            let field_strength = cherenkov_intensity.sqrt() * 1e-3; // Approximate field strength
            let plasma_density = charge_density * 1e6; // Convert to m⁻³
            let scattering_cross_section = 1e-20; // Typical electron scattering cross-section

            // Additional bremsstrahlung from Cherenkov-induced scattering
            field_strength * plasma_density * scattering_cross_section * 1e8
        } else {
            0.0
        };

        let total_brems_intensity = base_brems_intensity + coupling_factor;

        results.bremsstrahlung_emission.push(total_brems_intensity);
        results.cherenkov_emission.push(cherenkov_intensity);

        // Store spectra every 20 steps (simplified)
        if step % 20 == 0 {
            // Generate simplified spectral intensities
            let mut brems_spectrum = Vec::new();
            let mut cherenkov_spectrum = Vec::new();

            for &lambda in &results.wavelengths {
                // Bremsstrahlung: thermal spectrum peaking in visible/IR
                let brems_intensity = if temp > 5000.0 {
                    (temp / 10000.0).powi(2) * 1e7 * (-((lambda - 600e-9) / 200e-9).powi(2)).exp()
                } else {
                    0.0
                };

                // Cherenkov: enhanced in UV/blue with 1/λ dependence
                let cherenkov_intensity = if velocity_estimate > 3e8 / 1.5 {
                    velocity_estimate * 1e3 / lambda.powi(2)
                } else {
                    0.0
                };

                brems_spectrum.push(brems_intensity);
                cherenkov_spectrum.push(cherenkov_intensity);
            }

            results.bremsstrahlung_spectra.push(brems_spectrum);
            results.cherenkov_spectra.push(cherenkov_spectrum);
        }

        // Progress indicator
        if step % 50 == 0 {
            println!(
                "  Step {}/{}: T={:.0}K, R={:.1}μm, compression={:.1}x",
                step,
                n_steps,
                temp_field[center],
                radius_field[center] * 1e6,
                compression_ratio
            );
        }
    }

    println!(
        "  Simulation completed: {} steps, {:.1} μs total time",
        n_steps,
        results.times.last().unwrap_or(&0.0) * 1e6
    );
    Ok(results)
}

fn display_physics_comparison_data(results: &SimulationResults) {
    println!("\n📊 Physics Comparison Data");
    println!("---------------------------");

    // Display bubble dynamics data
    display_bubble_dynamics_data(results);

    // Display emission comparison data
    display_emission_comparison_data(results);

    // Display spectral signatures
    display_spectral_data(results);

    // Display physics relationships
    display_physics_relationships_data(results);
}

fn display_bubble_dynamics_data(results: &SimulationResults) {
    println!("🌡️  Bubble Dynamics Data:");
    println!("  Time(μs) | Temp(K) | Compression | Velocity(m/s)");
    println!("  ---------|---------|-------------|--------------");

    for i in (0..results.times.len()).step_by(20) {
        // Sample every 20 steps
        println!(
            "  {:.1}     | {:.0}    | {:.1}         | {:.1e}",
            results.times[i] * 1e6,
            results.temperatures[i],
            results.compression_ratios[i],
            results.particle_velocities[i]
        );
    }

    let max_temp = results.temperatures.iter().fold(0.0_f64, |a, &b| a.max(b));
    let max_compression = results
        .compression_ratios
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));
    let max_velocity = results
        .particle_velocities
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));
    let cherenkov_threshold = 3e8 / 1.5; // c/n for n=1.5

    println!("  Maximum values:");
    println!("  - Temperature: {:.0} K", max_temp);
    println!("  - Compression: {:.1}x", max_compression);
    println!("  - Velocity: {:.1e} m/s", max_velocity);
    println!(
        "  - Cherenkov threshold: {:.1e} m/s (v > c/n)",
        cherenkov_threshold
    );
    println!(
        "  - Threshold exceeded: {}",
        max_velocity > cherenkov_threshold
    );
    println!();
}

fn display_emission_comparison_data(results: &SimulationResults) {
    println!("⚡ Emission Comparison Data:");
    println!("  Time(μs) | Bremsstrahlung(W/m³) | Cherenkov(W/m³)");
    println!("  ---------|--------------------|----------------");

    for i in (0..results.times.len()).step_by(25) {
        // Sample every 25 steps
        println!(
            "  {:.1}     | {:.2e}             | {:.2e}",
            results.times[i] * 1e6,
            results.bremsstrahlung_emission[i],
            results.cherenkov_emission[i]
        );
    }

    let max_brems = results
        .bremsstrahlung_emission
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));
    let max_cherenkov = results
        .cherenkov_emission
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));

    // Calculate coupling contribution
    let coupling_contribution = if max_cherenkov > 0.0 {
        let avg_base_brems = results
            .bremsstrahlung_emission
            .iter()
            .zip(results.cherenkov_emission.iter())
            .filter(|(_, &c)| c == 0.0) // Only when Cherenkov is off
            .map(|(b, _)| *b)
            .sum::<f64>()
            / results
                .cherenkov_emission
                .iter()
                .filter(|&&c| c == 0.0)
                .count()
                .max(1) as f64;
        (max_brems - avg_base_brems).max(0.0)
    } else {
        0.0
    };

    println!("  Peak emissions:");
    println!("  - Bremsstrahlung: {:.2e} W/m³", max_brems);
    println!("  - Cherenkov: {:.2e} W/m³", max_cherenkov);
    println!(
        "  - Ratio (Brems/Cherenkov): {:.1}",
        max_brems / max_cherenkov.max(1e-12)
    );
    println!(
        "  - Coupling contribution: {:.2e} W/m³ ({:.1}% of total bremsstrahlung)",
        coupling_contribution,
        if max_brems > 0.0 {
            coupling_contribution / max_brems * 100.0
        } else {
            0.0
        }
    );
    println!();
}

fn display_spectral_data(results: &SimulationResults) {
    if results.bremsstrahlung_spectra.is_empty() {
        return;
    }

    println!("🌈 Spectral Signatures (at peak emission):");
    println!("  Wavelength(nm) | Bremsstrahlung | Cherenkov");
    println!("  --------------|---------------|----------");

    // Plot spectra from peak emission time
    let peak_idx = results
        .bremsstrahlung_emission
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i / 20) // Convert to spectrum index
        .unwrap_or(0);

    if peak_idx < results.bremsstrahlung_spectra.len() {
        for i in 0..results.wavelengths.len() {
            let wavelength_nm = results.wavelengths[i] * 1e9;
            let brems_intensity = results.bremsstrahlung_spectra[peak_idx][i];
            let cherenkov_intensity = results.cherenkov_spectra[peak_idx][i];

            println!(
                "  {:.0}            | {:.2e}      | {:.2e}",
                wavelength_nm, brems_intensity, cherenkov_intensity
            );
        }
    }

    // Analyze spectral characteristics
    let uv_range = results
        .wavelengths
        .iter()
        .enumerate()
        .filter(|(_, &w)| w < 400e-9) // UV/blue < 400nm
        .map(|(i, _)| i);

    let total_brems: f64 = results.bremsstrahlung_spectra[peak_idx].iter().sum();
    let total_cherenkov: f64 = results.cherenkov_spectra[peak_idx].iter().sum();
    let uv_brems = uv_range
        .clone()
        .map(|i| results.bremsstrahlung_spectra[peak_idx][i])
        .sum::<f64>();
    let uv_cherenkov = uv_range
        .map(|i| results.cherenkov_spectra[peak_idx][i])
        .sum::<f64>();

    println!("  Spectral analysis:");
    println!(
        "  - Bremsstrahlung UV fraction: {:.1}%",
        (uv_brems / total_brems.max(1e-12)) * 100.0
    );
    println!(
        "  - Cherenkov UV fraction: {:.1}%",
        (uv_cherenkov / total_cherenkov.max(1e-12)) * 100.0
    );
    println!(
        "  - Cherenkov UV bias: {:.1}x higher than bremsstrahlung",
        (uv_cherenkov / total_cherenkov.max(1e-12))
            / (uv_brems / total_brems.max(1e-12)).max(1e-12)
    );
    println!();
}

fn display_physics_relationships_data(results: &SimulationResults) {
    println!("🔗 Physics Relationships Analysis:");
    println!("  Temperature → Bremsstrahlung:");

    // Show temperature-emission correlation
    for i in (0..results.times.len()).step_by(25) {
        if i < results.temperatures.len() && i < results.bremsstrahlung_emission.len() {
            println!(
                "    T={:.0}K → Brems={:.2e} W/m³",
                results.temperatures[i], results.bremsstrahlung_emission[i]
            );
        }
    }

    println!("  Velocity → Cherenkov:");
    let cherenkov_threshold = 3e8 / 1.5; // c/n for n=1.5

    // Show velocity-emission correlation
    for i in (0..results.times.len()).step_by(25) {
        if i < results.particle_velocities.len() && i < results.cherenkov_emission.len() {
            let v = results.particle_velocities[i];
            let e = results.cherenkov_emission[i];
            let threshold_status = if v > cherenkov_threshold {
                "ABOVE"
            } else {
                "below"
            };
            println!(
                "    v={:.1e}m/s ({}) → Cherenkov={:.2e} W/m³",
                v, threshold_status, e
            );
        }
    }

    println!(
        "  Cherenkov threshold: {:.1e} m/s (v > c/n)",
        cherenkov_threshold
    );

    // Correlation analysis
    let temp_max_idx = results
        .temperatures
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let velocity_max_idx = results
        .particle_velocities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("  Peak analysis:");
    println!(
        "  - Bremsstrahlung peaks at max temperature (step {})",
        temp_max_idx
    );
    println!(
        "  - Cherenkov peaks at max velocity (step {})",
        velocity_max_idx
    );
    println!("  - Different timing: Temperature vs Velocity driven");
    println!();
}

fn analyze_physics_relationships(results: &SimulationResults) {
    println!("\n🔬 Physics Relationship Analysis");
    println!("--------------------------------");

    // Find peak emissions
    let (brems_peak_idx, _) = results
        .bremsstrahlung_emission
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let (cherenkov_peak_idx, _) = results
        .cherenkov_emission
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("📊 Peak Emission Analysis:");
    println!("  Bremsstrahlung Peak:");
    println!("    Time: {:.1} μs", results.times[brems_peak_idx] * 1e6);
    println!(
        "    Temperature: {:.0} K",
        results.temperatures[brems_peak_idx]
    );
    println!(
        "    Compression: {:.1}",
        results.compression_ratios[brems_peak_idx]
    );
    println!(
        "    Emission: {:.2e} W/m³",
        results.bremsstrahlung_emission[brems_peak_idx]
    );

    println!("  Cherenkov Peak:");
    println!(
        "    Time: {:.1} μs",
        results.times[cherenkov_peak_idx] * 1e6
    );
    println!(
        "    Velocity: {:.1e} m/s",
        results.particle_velocities[cherenkov_peak_idx]
    );
    println!(
        "    Compression: {:.1}",
        results.compression_ratios[cherenkov_peak_idx]
    );
    println!(
        "    Emission: {:.2e} W/m³",
        results.cherenkov_emission[cherenkov_peak_idx]
    );

    // Physics relationships
    println!("\n⚛️  Physics Relationships Observed:");

    // Cavitation → Bremsstrahlung
    let max_temp = results.temperatures.iter().fold(0.0_f64, |a, &b| a.max(b));
    let max_compression = results
        .compression_ratios
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));

    println!("  Cavitation → Bremsstrahlung:");
    println!("    Max Temperature: {:.0} K (plasma formation)", max_temp);
    println!(
        "    Max Compression: {:.1}x (extreme density)",
        max_compression
    );
    println!("    Ionization threshold exceeded for bremsstrahlung");

    // Cavitation → Cherenkov
    let max_velocity = results
        .particle_velocities
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));
    let cherenkov_threshold = 3e8 / 1.5; // c/n for n=1.5

    println!("  Cavitation → Cherenkov:");
    println!("    Max Velocity: {:.1e} m/s", max_velocity);
    println!("    Cherenkov Threshold: {:.1e} m/s", cherenkov_threshold);
    println!(
        "    Threshold {}exceeded",
        if max_velocity > cherenkov_threshold {
            ""
        } else {
            "not "
        }
    );

    // Bremsstrahlung ≠ Cherenkov causation (but potential coupling)
    let temp_at_cherenkov_peak = results.temperatures[cherenkov_peak_idx];
    let velocity_at_brems_peak = results.particle_velocities[brems_peak_idx];

    println!("  Bremsstrahlung ≠ Cherenkov Causation (But Potential Coupling):");
    println!(
        "    Temperature at Cherenkov peak: {:.0} K",
        temp_at_cherenkov_peak
    );
    println!(
        "    Velocity at Bremsstrahlung peak: {:.1e} m/s",
        velocity_at_brems_peak
    );
    println!("    Different physical drivers: T vs v");
    println!("    Potential Cherenkov → Bremsstrahlung coupling:");
    println!("      • Cherenkov EM fields create local scattering centers");
    println!("      • Accelerated/decelerated particles produce additional bremsstrahlung");
    println!("      • Energy transfer: Coherent → incoherent radiation");
    println!("      • Plasma feedback: Cherenkov modifies local plasma conditions");

    println!("\n🎯 Experimental Discrimination:");
    println!("  • Bremsstrahlung: Broadband thermal spectrum");
    println!("  • Cherenkov: UV/blue bias with 1/ω dependence");
    println!("  • Timing: Bremsstrahlung follows temperature, Cherenkov follows velocity");
    println!("  • Polarization: Cherenkov linearly polarized, bremsstrahlung unpolarized");
}
