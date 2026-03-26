//! Erosion pattern analysis and cavitation intensity

use crate::core::constants::cavitation::{COMPRESSION_FACTOR_EXPONENT, IMPACT_ENERGY_COEFFICIENT};
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
        .for_each(|out, &r, &v, &compression| {
            let collapse_energy = IMPACT_ENERGY_COEFFICIENT * liquid_density * v.powi(2) * r.powi(3);
            let compression_factor = compression.powf(COMPRESSION_FACTOR_EXPONENT);
            *out = collapse_energy * compression_factor;
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
            .for_each(|out, &damage, &v, &n| {
                let flow_factor = 1.0 + FLOW_INFLUENCE_COEFFICIENT * v;
                let angle_factor = n.abs();
                *out = damage * flow_factor * angle_factor;
            });

        potential
    }

    /// Identify high-risk areas
    #[must_use]
    pub fn risk_map(damage_field: &Array3<f64>, threshold: f64) -> Array3<bool> {
        damage_field.mapv(|d| d > threshold)
    }
}

