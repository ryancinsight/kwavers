//! Primary Bjerknes force calculation

use super::calculator::BjerknesCalculator;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::error::{KwaversError, KwaversResult};

impl BjerknesCalculator {
    /// Calculate primary Bjerknes force on a bubble in acoustic field.
    ///
    /// ## Formula
    ///
    /// ```text
    /// F_B1 = −(4π/3)R³ · ∂p/∂z
    /// ```
    ///
    /// This is the standard primary Bjerknes force: the bubble occupies a volume
    /// V = (4π/3)R³ and the net acoustic force on that volume is −V·∇p
    /// (Leighton 1994, §3.4; Blake 1986).
    ///
    /// ## Dimensional analysis
    ///
    /// [V·∇p] = m³ · Pa/m = kg·m/s² = N ✓
    ///
    /// ## References
    ///
    /// - Blake JR (1986). J. Acoust. Soc. Am. 79(5), 1357–1360.
    /// - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §3.4.
    ///
    /// # Arguments
    /// * `bubble_radius` — instantaneous bubble radius R (m)
    /// * `_acoustic_pressure_amplitude` — unused; retained for API compatibility
    /// * `pressure_gradient` — ∂p/∂z at the bubble centre (Pa/m)
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if `bubble_radius` ≤ 0.
    pub fn primary_bjerknes_force(
        &self,
        bubble_radius: f64,
        _acoustic_pressure_amplitude: f64,
        pressure_gradient: f64,
    ) -> KwaversResult<f64> {
        if bubble_radius <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radius must be positive".to_owned(),
            ));
        }

        // F_B1 = −(4π/3)R³ · ∂p/∂z
        let volume = (FOUR_PI / 3.0) * bubble_radius.powi(3);
        Ok(-volume * pressure_gradient)
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

    /// Zero pressure gradient → zero primary force (F = −V·∇p, ∇p = 0 → F = 0).
    #[test]
    fn primary_bjerknes_force_zero_for_zero_pressure_gradient() {
        let f = calc().primary_bjerknes_force(5e-6, 1e5, 0.0).unwrap();
        assert_eq!(f, 0.0, "zero pressure gradient must give zero force");
    }

    /// Force scales as R³: doubling radius increases force magnitude by 8×.
    ///
    /// F_B1 = −(4π/3)R³ · ∇p ∝ R³.
    #[test]
    fn primary_bjerknes_force_scales_as_radius_cubed() {
        let c = calc();
        let f1 = c.primary_bjerknes_force(5e-6, 1e5, 1e3).unwrap();
        let f2 = c.primary_bjerknes_force(10e-6, 1e5, 1e3).unwrap();
        // f2/f1 = (10/5)³ = 8
        assert!(
            (f2 / f1 - 8.0).abs() < 1e-12,
            "force ratio must be 8 when radius doubles (got {:.6})",
            f2 / f1
        );
    }

    /// Positive pressure gradient → negative force (radiation pushes toward minimum).
    #[test]
    fn primary_bjerknes_force_negative_for_positive_gradient() {
        let f = calc().primary_bjerknes_force(5e-6, 1e5, 1e3).unwrap();
        assert!(
            f < 0.0,
            "positive gradient must give negative force (got {f:.3e})"
        );
    }
}