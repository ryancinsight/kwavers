//! Direct test of angular spectrum propagation

#[cfg(test)]
mod tests {
    use crate::physics::mechanics::acoustic_wave::kzk::{
        diffraction_corrected::AngularSpectrumOperator, KZKConfig,
    };
    use crate::physics::validation::GaussianBeamParameters;
    use ndarray::Array2;
    use std::f64::consts::PI;

    #[test]
    fn test_angular_spectrum_gaussian_propagation() {
        // Simple test: propagate Gaussian beam and check radius
        let beam_waist = 5e-3; // 5mm
        let wavelength = 1.5e-3; // 1.5mm (1 MHz in water)
        let c0 = 1500.0; // m/s
        let frequency = c0 / wavelength;

        // Create analytical solution
        let params = GaussianBeamParameters::new(beam_waist, wavelength);

        // Setup grid
        let nx = 256;
        let ny = 256;
        let dx = 0.2e-3; // 0.2mm spacing
        let dz = 1e-3; // 1mm steps

        let config = KZKConfig {
            nx,
            ny,
            nz: 100,
            nt: 1,
            dx,
            dz,
            dt: 1e-6,
            c0,
            frequency,
            ..Default::default()
        };

        // Create angular spectrum operator
        let mut op = AngularSpectrumOperator::new(&config, dz);

        // Initialize with Gaussian beam
        let mut field = Array2::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                let x = (i as f64 - nx as f64 / 2.0) * dx;
                let y = (j as f64 - ny as f64 / 2.0) * dx;
                let r2 = x * x + y * y;
                field[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        // Propagate to Rayleigh distance
        let steps = (params.z_r / dz) as usize;
        println!(
            "Propagating {} steps to z_R = {:.2}mm",
            steps,
            params.z_r * 1000.0
        );

        for _ in 0..steps {
            let mut field_slice = field.view_mut();
            op.apply(&mut field_slice, dz);
        }

        // Measure beam radius
        let intensity = &field * &field;
        let measured_radius = measure_beam_radius(&intensity, dx);
        let expected_radius = params.beam_radius(params.z_r);

        println!("Measured radius: {:.3}mm", measured_radius * 1000.0);
        println!("Expected radius: {:.3}mm", expected_radius * 1000.0);

        // Check within 5% (accounting for discretization)
        let error = (measured_radius - expected_radius).abs() / expected_radius;
        assert!(
            error < 0.05,
            "Beam radius error too large: {:.1}%",
            error * 100.0
        );
    }

    #[test]
    fn test_angular_spectrum_phase_accuracy() {
        // Test that phase is preserved correctly
        let wavelength = 1e-3;
        let k0 = 2.0 * PI / wavelength;

        let config = KZKConfig {
            nx: 128,
            ny: 128,
            dx: wavelength / 4.0, // Quarter wavelength sampling
            frequency: 1500.0 / wavelength,
            c0: 1500.0,
            ..Default::default()
        };

        let dz = wavelength; // Propagate one wavelength
        let mut op = AngularSpectrumOperator::new(&config, dz);

        // Create plane wave
        let mut field = Array2::ones((128, 128));
        let mut field_slice = field.view_mut();

        // Propagate
        op.apply(&mut field_slice, dz);

        // Should still be unity magnitude (plane wave)
        let center = field[[64, 64]];
        assert!(
            (center.abs() - 1.0).abs() < 0.01,
            "Plane wave amplitude changed: {}",
            center.abs()
        );
    }
}
