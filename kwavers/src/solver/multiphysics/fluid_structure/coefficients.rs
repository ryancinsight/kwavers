use super::interface::FsiInterface;
use crate::math::fft::Complex64;

#[derive(Debug, Clone, Copy)]
pub struct ReflectionTransmissionCoefficients {
    /// Pressure reflection coefficient (amplitude ratio) []
    pub r_pp: Complex64,
    /// Velocity reflection coefficient []
    pub r_vv: Complex64,
    /// Longitudinal transmission coefficient (transmitted/incident pressure)
    pub t_pl: Complex64,
    /// Transverse transmission coefficient
    pub t_pt: Complex64,
    /// Energy reflection coefficient []
    pub r_energy: f64,
    /// Energy transmission coefficient (longitudinal) []
    pub t_energy_l: f64,
}

impl ReflectionTransmissionCoefficients {
    /// Calculate coefficients for normal incidence
    ///
    /// **Theorem**: Normal Incidence
    /// For θ = 0, the reflection coefficient reduces to:
    /// ```text
    /// R = (Z_s - Z_f) / (Z_s + Z_f)
    /// ```
    /// where Z_f = ρ_f c_f and Z_s = ρ_s c_l (solid longitudinal)
    ///
    /// **Energy Conservation**: |R|² + |T|² = 1 for lossless media
    pub fn normal_incidence(interface: &FsiInterface) -> Self {
        let z_f = interface.fluid_impedance();
        let z_s = interface.solid_longitudinal_impedance();

        // Amplitude coefficients
        let r = (z_s - z_f) / (z_s + z_f);
        let t = 2.0 * z_s / (z_s + z_f);

        // Energy coefficients
        let r_energy = r.powi(2);
        let t_energy = (z_f / z_s) * t.powi(2);

        Self {
            r_pp: Complex64::new(r, 0.0),
            r_vv: Complex64::new(-r, 0.0), // Velocity reflection opposite phase
            t_pl: Complex64::new(t, 0.0),
            t_pt: Complex64::new(0.0, 0.0), // No shear at normal incidence
            r_energy,
            t_energy_l: t_energy,
        }
    }

    /// Calculate for oblique incidence
    ///
    /// **Physics Note**: At oblique incidence, mode conversion occurs -
    /// incident pressure wave generates both longitudinal and transverse
    /// waves in the solid due to interface traction requirements.
    pub fn oblique_incidence(interface: &FsiInterface, theta_i: f64) -> Self {
        let c_f = interface.fluid_sound_speed;
        let c_l = interface.solid_c_l;
        let c_t = interface.solid_c_t;

        let sin_theta_i = theta_i.sin();
        let cos_theta_i = theta_i.cos();

        // Snell's law for transmitted angles
        let sin_theta_l = (c_l / c_f) * sin_theta_i;
        let sin_theta_t = (c_t / c_f) * sin_theta_i;

        // Check for critical angle
        if sin_theta_l.abs() > 1.0 || sin_theta_t.abs() > 1.0 {
            // Beyond critical angle - total internal reflection
            return Self {
                r_pp: Complex64::new(1.0, 0.0),
                r_vv: Complex64::new(-1.0, 0.0),
                t_pl: Complex64::new(0.0, 0.0),
                t_pt: Complex64::new(0.0, 0.0),
                r_energy: 1.0,
                t_energy_l: 0.0,
            };
        }

        let cos_theta_l = (1.0 - sin_theta_l.powi(2)).sqrt();
        let cos_theta_t = (1.0 - sin_theta_t.powi(2)).sqrt();

        let rho_f = interface.fluid_density;
        let rho_s = interface.solid_density;

        // Impedance terms
        let z_f = rho_f * c_f / cos_theta_i;
        let z_l = rho_s * c_l / cos_theta_l;
        let z_t = rho_s * c_t / cos_theta_t;

        // Coefficients from Brekhovskikh & Godin (1990)
        let _a = 2.0 * sin_theta_t * cos_theta_l;
        let d_den = z_l * cos_theta_t + z_f * cos_theta_t + rho_s * c_t * sin_theta_t * sin_theta_l;

        let r = (z_l * cos_theta_t - z_f * cos_theta_t + rho_s * c_t * sin_theta_t * sin_theta_l)
            / d_den;

        let t_l = 2.0 * z_f * cos_theta_t / d_den;
        let t_t = -2.0 * z_f * sin_theta_t / d_den;

        // Energy coefficients
        // Energy flux conservation: for displacement-potential coefficients the
        // transmitted energy ratio is (Z_transmitted / Z_incident), NOT the inverse.
        // Derivation: I_t/I_i = (Z_t/Z_f) |t|²  →  R² + Σ(Z_n/Z_f)|t_n|² = 1.
        let r_energy = r.powi(2);
        let t_energy_l = (z_l / z_f) * t_l.powi(2);
        let t_energy_t = (z_t / z_f) * t_t.powi(2);

        Self {
            r_pp: Complex64::new(r, 0.0),
            r_vv: Complex64::new(-r * cos_theta_i, 0.0),
            t_pl: Complex64::new(t_l, 0.0),
            t_pt: Complex64::new(t_t, 0.0),
            r_energy,
            t_energy_l: t_energy_l + t_energy_t,
        }
    }

    /// Verify energy conservation: R + T = 1
    ///
    /// **Numerical Verification**
    /// For computational implementations, tolerance accounts for:
    /// - Floating point errors (~1e-15)
    /// - Numerical dispersion in discretization
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn verify_energy_conservation(&self) -> Result<(), String> {
        let total = self.r_energy + self.t_energy_l;
        let tolerance = 1e-10;

        if (total - 1.0).abs() > tolerance {
            Err(format!(
                "Energy conservation violated: R + T = {}, deviation = {}",
                total,
                (total - 1.0).abs()
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Test reflection coefficient for normal incidence
    ///
    /// **Validation**: Water-Steel interface should have R ≈ 0.935
    /// reflecting almost all acoustic energy.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_normal_reflection_water_steel() {
        let interface = FsiInterface::new(
            1000.0, // water
            1500.0,
            7850.0, // steel
            5960.0,
            3240.0,
            [1.0, 0.0, 0.0],
            64,
            64,
            64,
        )
        .unwrap();

        let coeffs = ReflectionTransmissionCoefficients::normal_incidence(&interface);

        // Expected R ≈ 0.935 for water-steel
        let expected_r = 0.935;
        assert!(
            (coeffs.r_pp.re - expected_r).abs() < 0.01,
            "Reflection coefficient mismatch: got {}, expected ~{}",
            coeffs.r_pp.re,
            expected_r
        );
    }

    /// Test energy conservation
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_energy_conservation() {
        let interface = FsiInterface::new(
            1000.0,
            1500.0,
            2700.0, // aluminum
            6320.0,
            3080.0,
            [0.0, 1.0, 0.0],
            64,
            64,
            64,
        )
        .unwrap();

        let coeffs = ReflectionTransmissionCoefficients::normal_incidence(&interface);
        assert!(coeffs.verify_energy_conservation().is_ok());
    }

    /// Test oblique incidence at various angles
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_oblique_reflection_angles() {
        let interface = FsiInterface::new(
            1000.0,
            1500.0,
            7850.0,
            5960.0,
            3240.0,
            [1.0, 0.0, 0.0],
            64,
            64,
            64,
        )
        .unwrap();

        for angle_deg in [0.0, 15.0, 30.0, 45.0] {
            let theta = angle_deg * std::f64::consts::PI / 180.0;
            let coeffs = ReflectionTransmissionCoefficients::oblique_incidence(&interface, theta);

            // Energy should be conserved at all angles
            assert!(
                coeffs.verify_energy_conservation().is_ok(),
                "Energy conservation violated at {} degrees",
                angle_deg
            );
        }
    }
}
