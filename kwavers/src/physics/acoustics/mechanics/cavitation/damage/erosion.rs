//! Erosion pattern analysis and cavitation intensity

use crate::core::constants::cavitation::IMPACT_ENERGY_COEFFICIENT;
use crate::physics::bubble_dynamics::bubble_field::BubbleStateFields;
use ndarray::{Array3, Zip};

/// Empirical flow-velocity enhancement coefficient for erosion potential.
///
/// At low to moderate flow velocities, the erosion potential scales
/// approximately as $E \propto (1 + \alpha \cdot v)$ where $\alpha = 0.1$
/// represents the linear coupling between flow velocity and impact
/// probability at the material surface.
///
/// # References
/// - Tullis, J. P. (1989). "Cavitation and erosion in hydraulic machinery".
///   In *Hydraulics of Pipelines*, Wiley, Chapter 7.
const FLOW_INFLUENCE_COEFFICIENT: f64 = 0.1;

/// Calculate cavitation intensity parameter
#[must_use]
pub fn cavitation_intensity(bubble_states: &BubbleStateFields, liquid_density: f64) -> Array3<f64> {
    let shape = bubble_states.radius.dim();
    let mut intensity = Array3::zeros(shape);

    Zip::from(&mut intensity)
        .and(&bubble_states.radius)
        .and(&bubble_states.velocity)
        .and(&bubble_states.compression_ratio)
        .par_for_each(|out, &r, &v, &compression| {
            *out = IMPACT_ENERGY_COEFFICIENT * liquid_density * v.powi(2) * r.powi(3)
                * compression.powi(2);
        });

    intensity
}

/// Predict erosion patterns
#[derive(Debug)]
pub struct ErosionPattern;

impl ErosionPattern {
    /// Calculate erosion potential field
    #[must_use]
    pub fn erosion_potential(
        damage_field: &Array3<f64>,
        flow_velocity: &Array3<f64>,
        surface_normal: &Array3<f64>,
    ) -> Array3<f64> {
        let shape = damage_field.dim();
        let mut potential = Array3::zeros(shape);

        Zip::from(&mut potential)
            .and(damage_field)
            .and(flow_velocity)
            .and(surface_normal)
            .par_for_each(|out, &damage, &v, &n| {
                *out = damage * FLOW_INFLUENCE_COEFFICIENT.mul_add(v, 1.0) * n.abs();
            });

        potential
    }

    /// Identify high-risk areas
    #[must_use]
    pub fn risk_map(damage_field: &Array3<f64>, threshold: f64) -> Array3<bool> {
        damage_field.mapv(|d| d > threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// `risk_map` marks cells strictly above the threshold as high-risk.
    ///
    /// Cells exactly at the threshold are NOT high-risk (strict inequality `>`).
    #[test]
    fn risk_map_marks_cells_above_threshold_only() {
        let mut field = Array3::<f64>::zeros((2, 2, 2));
        field[[0, 0, 0]] = 1.5; // above
        field[[1, 1, 1]] = 0.5; // at half (below)
        let threshold = 1.0;

        let risk = ErosionPattern::risk_map(&field, threshold);

        assert!(risk[[0, 0, 0]], "cell above threshold must be high-risk");
        assert!(!risk[[1, 1, 1]], "cell below threshold must not be high-risk");
        // Cells equal to threshold (none here) would also be false (strict >)
        let eq_field = Array3::<f64>::from_elem((2, 2, 2), 1.0);
        let risk_eq = ErosionPattern::risk_map(&eq_field, 1.0);
        assert!(!risk_eq[[0, 0, 0]], "cell equal to threshold must not be high-risk");
    }

    /// `erosion_potential` formula: damage × (1 + 0.1·v) × |n|.
    ///
    /// For damage=1, v=0, n=1: potential = 1·1·1 = 1.0.
    /// For damage=2, v=10, n=0.5: potential = 2·(1+1)·0.5 = 2.0.
    #[test]
    fn erosion_potential_matches_formula_analytically() {
        let damage = Array3::<f64>::from_elem((2, 2, 2), 1.0);
        let velocity = Array3::<f64>::zeros((2, 2, 2)); // v=0
        let normal = Array3::<f64>::from_elem((2, 2, 2), 1.0); // n=1

        let potential = ErosionPattern::erosion_potential(&damage, &velocity, &normal);

        // expected: 1.0 × (1 + 0.1×0) × |1.0| = 1.0
        for &v in potential.iter() {
            assert!(
                (v - 1.0).abs() < 1e-14,
                "erosion_potential must be 1.0 for damage=1,v=0,n=1 (got {v:.3e})"
            );
        }
    }

    /// Zero damage field produces zero erosion potential regardless of velocity.
    #[test]
    fn erosion_potential_zero_for_zero_damage() {
        let damage = Array3::<f64>::zeros((2, 2, 2));
        let velocity = Array3::<f64>::from_elem((2, 2, 2), 100.0);
        let normal = Array3::<f64>::from_elem((2, 2, 2), 1.0);

        let potential = ErosionPattern::erosion_potential(&damage, &velocity, &normal);

        for &v in potential.iter() {
            assert_eq!(v, 0.0, "zero damage must give zero erosion potential");
        }
    }
}
