//! Primary Bjerknes force calculation

use super::calculator::BjerknesCalculator;
use crate::core::error::{KwaversError, KwaversResult};

impl BjerknesCalculator {
    /// Calculate primary Bjerknes force on a bubble in acoustic field
    ///
    /// The primary Bjerknes force arises from the time-averaged radiation pressure:
    /// F_p = -4πR² ⟨p'(∂u'/∂z)⟩
    ///
    /// For a plane wave: F_p ≈ πR² I / c₀
    /// where I is acoustic intensity
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn primary_bjerknes_force(
        &self,
        bubble_radius: f64,
        acoustic_pressure_amplitude: f64,
        pressure_gradient: f64,
    ) -> KwaversResult<f64> {
        if bubble_radius <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radius must be positive".to_owned(),
            ));
        }

        // Primary Bjerknes force for acoustic wave
        // F_p = -4πR² ⟨p'(∂u'/∂z)⟩
        // Approximation: F_p ≈ -4πR² (p₀/(2ρc₀)) (∂p/∂z)
        // where the time-averaged acoustic intensity contribution is proportional to pressure amplitude and gradient

        let surface_area = 4.0 * std::f64::consts::PI * bubble_radius.powi(2);

        // Acoustic radiation pressure contribution
        let radiation_pressure =
            acoustic_pressure_amplitude.powi(2) / (2.0 * self.config.rho * self.config.c0);

        // Force from pressure gradient
        let force = -surface_area * radiation_pressure * (pressure_gradient / self.config.c0);

        Ok(force)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{calculator::BjerknesCalculator, types::BjerknesConfig};

    fn calc() -> BjerknesCalculator {
        BjerknesCalculator::new(BjerknesConfig::default())
    }

    /// Zero bubble radius returns Err.
    #[test]
    fn primary_bjerknes_force_errors_for_nonpositive_radius() {
        let c = calc();
        assert!(c.primary_bjerknes_force(0.0, 1e5, 1e3).is_err());
        assert!(c.primary_bjerknes_force(-1e-6, 1e5, 1e3).is_err());
    }

    /// Zero acoustic pressure amplitude → zero primary force.
    ///
    /// F_p ∝ p² → p=0 gives F_p = 0 regardless of gradient.
    #[test]
    fn primary_bjerknes_force_zero_for_zero_pressure_amplitude() {
        let f = calc().primary_bjerknes_force(5e-6, 0.0, 1e3).unwrap();
        assert_eq!(f, 0.0, "zero pressure amplitude must give zero force");
    }

    /// Force scales as R²: doubling radius quadruples force magnitude.
    ///
    /// F_p = -4πR² · radiation_pressure · (grad/c₀) ∝ R².
    #[test]
    fn primary_bjerknes_force_scales_as_radius_squared() {
        let c = calc();
        let f1 = c.primary_bjerknes_force(5e-6, 1e5, 1e3).unwrap();
        let f2 = c.primary_bjerknes_force(10e-6, 1e5, 1e3).unwrap();
        // f2/f1 = (10/5)² = 4
        assert!(
            (f2 / f1 - 4.0).abs() < 1e-12,
            "force ratio must be 4 when radius doubles (got {:.6})", f2 / f1
        );
    }

    /// Positive pressure gradient → negative force (radiation pushes toward minimum).
    #[test]
    fn primary_bjerknes_force_negative_for_positive_gradient() {
        let f = calc().primary_bjerknes_force(5e-6, 1e5, 1e3).unwrap();
        assert!(f < 0.0, "positive gradient must give negative force (got {f:.3e})");
    }
}
