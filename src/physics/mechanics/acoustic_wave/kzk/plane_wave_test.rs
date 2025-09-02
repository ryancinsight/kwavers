//! Test plane wave propagation to verify diffraction operator

#[cfg(test)]
mod tests {
    use crate::constants::physics::SOUND_SPEED_WATER;
    use crate::physics::mechanics::acoustic_wave::kzk::{
        constants::*, parabolic_diffraction::KzkDiffractionOperator, KZKConfig,
    };
    use ndarray::Array2;
    use std::f64::consts::PI;

    #[test]
    fn test_plane_wave_propagation() {
        // A plane wave should propagate without change
        let config = KZKConfig {
            nx: 64,
            ny: 64,
            dx: DEFAULT_WAVELENGTH / 8.0,
            frequency: DEFAULT_FREQUENCY,
            c0: SOUND_SPEED_WATER,
            ..Default::default()
        };

        let mut op = KzkDiffractionOperator::new(&config);

        // Create plane wave (constant amplitude)
        let mut field = Array2::ones((config.nx, config.ny));

        // Store initial field
        let initial_field = field.clone();

        // Propagate one wavelength
        let dz = DEFAULT_WAVELENGTH;
        let mut field_view = field.view_mut();
        op.apply(&mut field_view, dz);

        // Check that amplitude is preserved
        let center = field[[32, 32]];
        println!("Plane wave after λ propagation:");
        println!("Center amplitude: {:.6}", center);
        println!("Should be: 1.0");

        // Plane wave should maintain unit amplitude
        assert!((center - 1.0).abs() < 0.01, "Plane wave amplitude changed!");

        // Check phase advance
        // After propagating one wavelength, phase should advance by 2π
        // But since we only track real part, we might see oscillation
    }

    #[test]
    fn test_diffraction_operator_normalization() {
        // Test that the operator conserves energy
        let config = KZKConfig {
            nx: 64,
            ny: 64,
            dx: DEFAULT_WAVELENGTH / 8.0,
            frequency: DEFAULT_FREQUENCY,
            c0: SOUND_SPEED_WATER,
            ..Default::default()
        };

        let mut op = KzkDiffractionOperator::new(&config);

        // Create a localized source
        let mut field = Array2::zeros((config.nx, config.ny));
        let cx = config.nx / 2;
        let cy = config.ny / 2;

        // Delta function in center
        field[[cx, cy]] = 1.0;

        let initial_energy: f64 = field.iter().map(|&x| x * x).sum();
        println!("Initial energy: {:.6}", initial_energy);

        // Apply diffraction
        let dz = DEFAULT_WAVELENGTH / 10.0;
        let mut field_view = field.view_mut();
        op.apply(&mut field_view, dz);

        let final_energy: f64 = field.iter().map(|&x| x * x).sum();
        println!("Final energy: {:.6}", final_energy);
        println!("Energy ratio: {:.6}", final_energy / initial_energy);

        // Energy should be approximately conserved
        assert!((final_energy / initial_energy - 1.0).abs() < 0.1);
    }
}
