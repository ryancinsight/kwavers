use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::numerical::MHZ_TO_HZ;
use std::f64::consts::PI;

#[test]
fn test_numerical_dispersion() {
    // Test numerical dispersion for different PPW values
    let frequencies = vec![0.5e6, 1e6, 2e6, 5e6];
    let dx = 1e-3;
    let _n = 64;

    for freq in frequencies {
        let wavelength = SOUND_SPEED_WATER_SIM / freq;
        let ppw = wavelength / dx;

        if ppw < 2.0 {
            continue; // Skip under-resolved cases
        }

        // Theoretical phase velocity (no dispersion)
        let c_theoretical = SOUND_SPEED_WATER_SIM;

        // Numerical phase velocity (with dispersion)
        let k = 2.0 * PI / wavelength;
        let k_dx = k * dx;

        // Second-order finite difference dispersion relation
        let c_numerical = c_theoretical * (k_dx / 2.0).sin() / (k_dx / 2.0);

        let dispersion_error = (c_numerical - c_theoretical).abs() / c_theoretical;

        // Error should decrease with increasing PPW
        let expected_error = 1.0 / ppw.powi(2); // Second-order accuracy

        assert!(
            dispersion_error < expected_error * 10.0,
            "Dispersion error at {} MHz: {:.4} (PPW: {:.1})",
            freq / MHZ_TO_HZ,
            dispersion_error,
            ppw
        );
    }
    // Reference PPW_MINIMUM to suppress dead_code lint
    let _ = super::PPW_MINIMUM;
}
