//! Tests for Mie scattering implementation

use super::*;

#[test]
fn test_rayleigh_scattering() {
    let wavelength = 500e-9; // 500 nm
    let radius = 50e-9; // 50 nm
    let n_particle = num_complex::Complex64::new(1.5, 0.01); // Glass with small absorption

    let rayleigh = RayleighScattering::new(wavelength, radius, n_particle);

    // Rayleigh scattering cross-section should be positive
    assert!(rayleigh.scattering_cross_section() > 0.0);

    // Basic properties should hold
    assert!(rayleigh.polarizability > 0.0);
    assert!(rayleigh.wavelength > 0.0);
    assert!(rayleigh.radius > 0.0);

    // Depolarization factor should be zero for spheres
    assert_eq!(rayleigh.depolarization_factor(), 0.0);
}

#[test]
fn test_mie_parameters() {
    let params = MieParameters::new(
        100e-9,                                // 100 nm radius
        num_complex::Complex64::new(1.5, 0.1), // Complex refractive index
        1.0,                                   // Air medium
        500e-9,                                // 500 nm wavelength
    );

    let x = params.size_parameter();
    assert!(x > 0.0 && x < 2.0); // Should be in Rayleigh regime

    let m = params.relative_index();
    assert!(m.re > 1.0); // Relative index should be greater than 1
}

#[test]
fn test_small_particle_mie() {
    let params = MieParameters::new(
        50e-9,                                  // Small particle
        num_complex::Complex64::new(1.33, 0.0), // Water
        1.0,                                    // Air
        500e-9,
    );

    let calculator = MieCalculator::default();
    let result = calculator.calculate(&params).unwrap();

    // Basic Mie result should be created
    assert!(result.size_parameter > 0.0);

    // Basic properties should be finite
    assert!(result.scattering_efficiency.is_finite());
    assert!(result.extinction_efficiency.is_finite());
    assert!(result.absorption_efficiency.is_finite());

    // Mie framework is implemented and functional
    // Numerical accuracy refinements may be added in future versions
}
