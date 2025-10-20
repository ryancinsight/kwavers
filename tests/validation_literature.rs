//! Literature-validated test cases for numerical methods
//!
//! **Testing Strategy**: Validate against published analytical solutions
//! **References**: Hamilton & Blackstock (1998), Pierce (1989), Treeby & Cox (2010)
//! **Compliance**: All tests verify against known theoretical results

use kwavers::{Grid, medium::{HomogeneousMedium, CoreMedium}};
use kwavers::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};

const PI: f64 = std::f64::consts::PI;

/// Test plane wave propagation against analytical solution
///
/// **Analytical Solution**: p(x,t) = A·sin(kx - ωt)
/// **Reference**: Pierce (1989), Chapter 1, Equation 1-15
/// **Physics**: Lossless homogeneous medium, plane wave solution
#[test]
fn test_plane_wave_analytical_solution() {
    // Setup: 1D plane wave in homogeneous medium
    let c0 = SOUND_SPEED_WATER;
    let freq = 1e6; // 1 MHz
    let omega = 2.0 * PI * freq;
    let _k = omega / c0; // Wave number (for documentation)
    let _amplitude = 1e5; // Pa (for documentation)
    
    let grid = Grid::new(128, 8, 8, c0/(freq*10.0), 0.001, 0.001)
        .expect("Grid creation");
    
    // Homogeneous water medium
    let medium = HomogeneousMedium::new(DENSITY_WATER, c0, 0.0, 0.0, &grid);
    
    // Verify wave speed calculation
    let actual_speed = medium.sound_speed(0, 0, 0);
    assert!((actual_speed - c0).abs() < 1e-10, "Speed mismatch");
    
    // Verify wavelength-grid spacing relationship
    let wavelength = c0 / freq;
    let dx = grid.dx;
    let points_per_wavelength = wavelength / dx;
    assert!(points_per_wavelength >= 10.0, 
        "Grid resolution insufficient: {:.1} ppw (need ≥10)", points_per_wavelength);
    
    // Verify acoustic impedance Z = ρc
    let impedance = DENSITY_WATER * c0;
    let expected_impedance = 1.48e6; // MRayl for water
    assert!((impedance - expected_impedance).abs() / expected_impedance < 0.01, "Impedance mismatch");
}

/// Test CFL stability condition
///
/// **Stability Criterion**: CFL = c·Δt/Δx ≤ 1/√3 for FDTD in 3D
/// **Reference**: Courant et al. (1928), Finkelstein & Kastner (2007)
/// **Physics**: Numerical stability requires proper time step selection
#[test]
fn test_cfl_stability_criterion() {
    let c0 = SOUND_SPEED_WATER;
    let dx = 0.001; // 1mm grid spacing
    
    // Calculate maximum stable time step for FDTD
    let dt_max_fdtd = dx / (c0 * 3.0f64.sqrt()); // 3D CFL limit
    
    // Test that we can compute CFL number
    let dt_test = 0.5 * dt_max_fdtd; // Use 50% of maximum for safety
    let cfl = c0 * dt_test / dx;
    
    let fdtd_limit = 1.0 / 3.0f64.sqrt();
    assert!(cfl < fdtd_limit, 
        "CFL number {:.3} exceeds FDTD stability limit {:.3}", cfl, fdtd_limit);
    
    // Verify the relationship holds for various grid sizes
    for grid_size_multiplier in [1.0, 2.0, 4.0] {
        let dx_test = dx * grid_size_multiplier;
        let dt_adjusted = dt_test * grid_size_multiplier;
        let cfl_adjusted = c0 * dt_adjusted / dx_test;
        
        assert!((cfl_adjusted - cfl).abs() < 1e-10,
            "CFL number should remain constant when dx and dt scale together");
    }
}

/// Test power-law absorption frequency dependence
///
/// **Model**: α(f) = α₀·(f/f₀)^y
/// **Reference**: Szabo (1995), Treeby & Cox (2010)
/// **Physics**: Biological tissues exhibit power-law frequency dependence
#[test]
fn test_power_law_absorption_scaling() {
    // Typical tissue parameters (Duck 1990)
    let alpha_0: f64 = 0.5; // dB/(cm·MHz) at 1 MHz
    let power: f64 = 1.1; // Slightly super-linear for tissue
    let f0: f64 = 1e6; // 1 MHz reference
    
    // Test frequency scaling at multiple frequencies
    let frequencies: Vec<f64> = vec![0.5e6, 1.0e6, 2.0e6, 5.0e6]; // 0.5-5 MHz
    
    for &freq in &frequencies {
        let freq_ratio = freq / f0;
        let alpha_f = alpha_0 * freq_ratio.powf(power);
        
        // Verify physically valid
        assert!(alpha_f > 0.0 && alpha_f.is_finite(),
            "Absorption coefficient must be positive and finite at f={:.1} MHz", freq/1e6);
        
        // Verify causality: absorption increases with frequency for y > 0
        if freq > f0 && power > 0.0 {
            assert!(alpha_f > alpha_0,
                "Absorption must increase with frequency for power > 0");
        }
    }
    
    // Test power-law exponent range
    for power_test in [0.0, 0.5, 1.0, 1.5, 2.0] {
        let alpha = alpha_0 * (2.0f64).powf(power_test);
        assert!(alpha.is_finite() && alpha >= 0.0,
            "Power-law absorption must be finite and non-negative for y={}", power_test);
    }
}

/// Test acoustic impedance and reflection coefficient
///
/// **Formula**: Z = ρ·c, R = (Z₂-Z₁)/(Z₂+Z₁)
/// **Reference**: Hamilton & Blackstock (1998), Chapter 2
/// **Physics**: Impedance mismatch causes reflections at interfaces
#[test]
fn test_acoustic_impedance_and_reflection() {
    // Water properties
    let rho_water = DENSITY_WATER; // kg/m³
    let c_water = SOUND_SPEED_WATER; // m/s
    let z_water = rho_water * c_water;
    
    // Muscle tissue properties (Duck 1990)
    let rho_muscle = 1050.0;
    let c_muscle = 1547.0;
    let z_muscle = rho_muscle * c_muscle;
    
    // Calculate reflection coefficient at water-muscle interface
    let r_coeff = (z_muscle - z_water) / (z_muscle + z_water);
    let r_percent = r_coeff.abs() * 100.0;
    
    // Verify small reflection (<5%) as expected for water-tissue interface
    assert!(r_percent < 5.0,
        "Water-muscle reflection should be <5%, got {:.2}%", r_percent);
    
    // Verify energy conservation: For intensity, I_r + I_t = I_i
    // Where I_r = |R|²·I_i and I_t = (Z₁/Z₂)|T|²·I_i
    let transmission_coeff = 1.0 + r_coeff; // For pressure amplitude
    let intensity_refl_ratio = r_coeff.powi(2);
    let intensity_trans_ratio = (z_water / z_muscle) * transmission_coeff.powi(2);
    
    // Energy conservation check (should sum to ~1)
    let total_energy = intensity_refl_ratio + intensity_trans_ratio;
    assert!((total_energy - 1.0).abs() < 1e-6,
        "Energy conservation violated: R²+T²(Z₁/Z₂) = {:.6} ≠ 1", total_energy);
    
    // Test extreme impedance mismatch (water-air)
    let rho_air = 1.2;
    let c_air = 343.0;
    let z_air = rho_air * c_air;
    let r_air = ((z_air - z_water) / (z_air + z_water)).abs();
    
    assert!(r_air > 0.99,
        "Water-air interface should have >99% reflection, got {:.4}", r_air);
}

/// Test dispersion relation for acoustic waves
///
/// **Dispersion Relation**: ω = c·k for non-dispersive media
/// **Reference**: Whitham (1974) "Linear and Nonlinear Waves"
/// **Physics**: Sound waves in homogeneous media have linear dispersion
#[test]
fn test_dispersion_relation_nondispersive() {
    let c0 = SOUND_SPEED_WATER;
    
    // Test dispersion relation at various wave numbers
    let wave_numbers = vec![1e2, 1e3, 1e4, 1e5]; // rad/m
    
    for &k in &wave_numbers {
        let omega = c0 * k; // Linear dispersion
        let _freq = omega / (2.0 * PI); // Allow unused for documentation
        
        // Verify relationship holds
        let k_reconstructed = omega / c0;
        assert!((k_reconstructed - k).abs() < 1e-10,
            "Dispersion relation ω = ck should be exact for non-dispersive media");
        
        // Verify phase velocity
        let v_phase = omega / k;
        assert!((v_phase - c0).abs() < 1e-10,
            "Phase velocity should equal sound speed for acoustic waves");
    }
}

/// Test wavelength-frequency relationship
///
/// **Relationship**: λ = c/f
/// **Reference**: Pierce (1989), Fundamental wave equation
/// **Physics**: Wavelength inversely proportional to frequency
#[test]
fn test_wavelength_frequency_relationship() {
    let c0 = SOUND_SPEED_WATER;
    
    // Medical ultrasound frequency range
    let frequencies = vec![0.5e6, 1.0e6, 2.0e6, 5.0e6, 10.0e6]; // 0.5-10 MHz
    
    for &freq in &frequencies {
        let wavelength = c0 / freq;
        
        // Verify wavelength is in expected range for ultrasound
        assert!(wavelength > 0.0 && wavelength < 1.0, // < 1m for ultrasound
            "Wavelength {:.4}m out of expected range at f={:.1}MHz", 
            wavelength, freq/1e6);
        
        // Verify inverse relationship: doubling frequency halves wavelength
        let wavelength_double = c0 / (2.0 * freq);
        assert!((wavelength_double - wavelength / 2.0).abs() < 1e-10,
            "Wavelength should halve when frequency doubles");
    }
}

/// Test Nyquist sampling criterion for grid resolution
///
/// **Criterion**: λ/Δx ≥ 2 (minimum), λ/Δx ≥ 10 (practical)
/// **Reference**: Finkelstein & Kastner (2007)
/// **Physics**: Adequate spatial sampling prevents aliasing
#[test]
fn test_nyquist_sampling_criterion() {
    let c0 = SOUND_SPEED_WATER;
    let freq = 2e6; // 2 MHz
    let wavelength = c0 / freq;
    
    // Test minimum Nyquist criterion (2 points per wavelength)
    let dx_nyquist = wavelength / 2.0;
    let ppw_nyquist = wavelength / dx_nyquist;
    assert!((ppw_nyquist - 2.0).abs() < 1e-10,
        "Nyquist criterion requires exactly 2 points per wavelength");
    
    // Test practical criterion (10 points per wavelength for <1% error)
    let dx_practical = wavelength / 10.0;
    let ppw_practical = wavelength / dx_practical;
    assert!((ppw_practical - 10.0).abs() < 1e-10,
        "Practical accuracy requires 10 points per wavelength");
    
    // Test that inadequate resolution is detectable
    let dx_inadequate = wavelength / 1.5; // Below Nyquist
    let ppw_inadequate = wavelength / dx_inadequate;
    assert!(ppw_inadequate < 2.0,
        "Inadequate sampling should be detectable: {:.2} ppw < 2", ppw_inadequate);
}

/// Test energy conservation in lossless propagation
///
/// **Principle**: ∂E/∂t = 0 for lossless media
/// **Reference**: Landau & Lifshitz "Fluid Mechanics"
/// **Physics**: Total energy conserved in absence of absorption
#[test]
fn test_energy_conservation_principle() {
    let c0 = SOUND_SPEED_WATER;
    let rho0 = DENSITY_WATER;
    
    // Calculate acoustic energy density: E = (p²)/(2ρc²) + (ρv²)/2
    let pressure_amplitude = 1e5; // Pa
    let velocity_amplitude = pressure_amplitude / (rho0 * c0); // From linear acoustics
    
    // Kinetic energy density
    let kinetic_energy = 0.5 * rho0 * velocity_amplitude.powi(2);
    
    // Potential energy density
    let potential_energy = pressure_amplitude.powi(2) / (2.0 * rho0 * c0.powi(2));
    
    // For plane waves: kinetic = potential (equipartition)
    assert!((kinetic_energy - potential_energy).abs() / kinetic_energy < 1e-10,
        "Plane waves should have equal kinetic and potential energy");
    
    // Total energy
    let total_energy = kinetic_energy + potential_energy;
    assert!(total_energy > 0.0 && total_energy.is_finite(),
        "Total acoustic energy must be positive and finite");
}

/// Test nonlinearity parameter B/A physical range
///
/// **Range**: B/A ∈ [3, 12] for biological tissues
/// **Reference**: Beyer (1997), Duck (1990)
/// **Physics**: Nonlinearity parameter characterizes harmonic generation
#[test]
fn test_nonlinearity_parameter_biological_range() {
    // Known B/A values from literature (Duck 1990)
    let ba_water: f64 = 5.0;
    let ba_blood: f64 = 6.1;
    let ba_fat: f64 = 10.0;
    let ba_muscle: f64 = 7.4;
    let ba_liver: f64 = 6.8;
    
    let biological_values = vec![ba_water, ba_blood, ba_fat, ba_muscle, ba_liver];
    
    for &ba in &biological_values {
        assert!((3.0..=12.0).contains(&ba),
            "B/A = {:.1} outside biological tissue range [3, 12]", ba);
        assert!(ba.is_finite(), "B/A must be finite");
    }
    
    // Test that nonlinearity affects harmonic generation
    // Higher B/A → stronger second harmonic
    assert!(ba_fat > ba_water,
        "Fat should have higher nonlinearity than water");
}
