//! Transducer array trajectory and phase optimization

use super::planner::TreatmentPlanner;
use super::types::{
    TranscranialTargetVolume, TranscranialTransducerSpecification, TransducerSetup,
};
use crate::core::error::KwaversResult;
use crate::core::constants::numerical::{TWO_PI};

impl TreatmentPlanner {
    /// Optimize transducer setup for target focusing using phase conjugation.
    ///
    /// # Algorithm — Fibonacci Hemisphere Sampling + Phase Conjugation
    ///
    /// ## Element Placement (Fibonacci hemisphere)
    ///
    /// Uniform coverage of the upper hemisphere is achieved via the Fibonacci
    /// (golden angle) method (Roberts 2018; Álvarez & Güemes 2019):
    ///
    /// ```text
    /// zᵢ = (i + 0.5) / N        (cos θ uniformly spaced ∈ (0, 1) → upper hemisphere)
    /// φᵢ = 2π i / Φ             (golden angle spiral, Φ = (1+√5)/2 ≈ 1.618)
    /// xᵢ = R √(1−zᵢ²) cos φᵢ
    /// yᵢ = R √(1−zᵢ²) sin φᵢ
    /// ```
    ///
    /// This yields the minimum discrepancy point distribution on S²∩{z>0}
    /// (quasi-uniform, no clustering at poles or equator).
    ///
    /// ## Phase Delays (Phase Conjugation / Time Reversal)
    ///
    /// For focusing at target position `r_t`, the phase delay on element `i`
    /// at position `rᵢ` is (Daum & Hynynen 1999, Eq. 3):
    ///
    /// ```text
    /// φᵢ = −k · |rᵢ − r_t|     (k = 2π f / c)
    /// ```
    ///
    /// All elements then arrive in phase at the geometric focus (constructive
    /// interference), maximising focal intensity. Skull aberration correction
    /// would replace this with the measured one-way phase from each element.
    ///
    /// # References
    ///
    /// - Roberts M (2018). "Evenly distributing points on a sphere." *Extreme Learning* (blog).
    /// - Daum DR & Hynynen K (1999). "A 256-element ultrasonic phased array system for
    ///   the treatment of large volumes of deep seated tissue." *IEEE Trans Biomed Eng*
    ///   46(9):1070–1082.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn optimize_transducer_setup(
        &self,
        targets: &[TranscranialTargetVolume],
        spec: &TranscranialTransducerSpecification,
    ) -> KwaversResult<TransducerSetup> {
        let num_elements = spec.num_elements;
        let mut element_positions = Vec::with_capacity(num_elements);
        let element_amplitudes = vec![1.0; num_elements];

        // Fibonacci hemisphere: uniform angular spacing via golden angle
        const GOLDEN_RATIO: f64 = 1.618_033_988_749_895; // (1+√5)/2
        let radius = spec.radius;

        for i in 0..num_elements {
            // cos(θ) uniformly sampled ∈ (0, 1) — upper hemisphere only
            let z_norm = (i as f64 + 0.5) / num_elements as f64; // ∈ (0, 1)
            let r_xy = (1.0 - z_norm * z_norm).sqrt();
            let az = TWO_PI * i as f64 / GOLDEN_RATIO;

            element_positions.push([
                radius * r_xy * az.cos(),
                radius * r_xy * az.sin(),
                radius * z_norm,
            ]);
        }

        // Phase conjugation: φᵢ = −k · |rᵢ − r_target|
        // Focuses all elements at the first target (can be extended to multi-focus).
        let k = TWO_PI * spec.frequency / spec.sound_speed;
        let target_center = targets[0].center;

        let element_phases: Vec<f64> = element_positions
            .iter()
            .map(|&pos| {
                let dist = (pos[2] - target_center[2])
                    .mul_add(
                        pos[2] - target_center[2],
                        (pos[1] - target_center[1]).mul_add(
                            pos[1] - target_center[1],
                            (pos[0] - target_center[0]).powi(2),
                        ),
                    )
                    .sqrt();
                -k * dist
            })
            .collect();

        Ok(TransducerSetup {
            num_elements,
            element_positions,
            element_phases,
            element_amplitudes,
            frequency: spec.frequency,
            focal_distance: spec.focal_distance,
        })
    }
}
