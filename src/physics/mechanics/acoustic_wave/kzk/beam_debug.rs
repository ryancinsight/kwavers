//! Debug module for beam propagation issues

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use std::f64::consts::PI;

    #[test]
    fn test_gaussian_beam_analytical() {
        // Analytical Gaussian beam propagation
        let w0 = 5e-3; // 5mm beam waist
        let lambda = 1.5e-3; // 1.5mm wavelength
        let z_r = PI * w0 * w0 / lambda; // Rayleigh distance

        println!("Beam waist w0 = {:.3} mm", w0 * 1000.0);
        println!("Wavelength λ = {:.3} mm", lambda * 1000.0);
        println!("Rayleigh distance z_R = {:.3} mm", z_r * 1000.0);

        // Beam radius at various distances
        for factor in &[0.0, 0.5, 1.0, 2.0] {
            let z = factor * z_r;
            let w_z = w0 * (1.0 + (z / z_r).powi(2)).sqrt();
            println!("At z = {:.1} z_R: w(z) = {:.3} mm", factor, w_z * 1000.0);
        }

        // At Rayleigh distance
        let w_at_zr = w0 * 2.0_f64.sqrt();
        println!("\nAt z = z_R: w(z) = w0 × √2 = {:.3} mm", w_at_zr * 1000.0);

        // Verify the ratio
        assert!((w_at_zr / w0 - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_diffraction_strength() {
        // Check if diffraction is strong enough
        let w0 = 5e-3;
        let lambda = 1.5e-3;
        let k0 = 2.0 * PI / lambda;
        let z_r = PI * w0 * w0 / lambda;

        // Maximum transverse k for beam spreading
        let k_max = 1.0 / w0; // Approximately where Gaussian falls off

        // Phase accumulated over Rayleigh distance
        let phase_max = k_max * k_max * z_r / (2.0 * k0);

        println!("k0 = {:.3} rad/m", k0);
        println!("k_max ≈ {:.3} rad/m", k_max);
        println!("k_max/k0 = {:.3}", k_max / k0);
        println!("Maximum phase over z_R: {:.3} rad", phase_max);
        println!(
            "This corresponds to {:.1} wavelengths",
            phase_max / (2.0 * PI)
        );
    }
}
