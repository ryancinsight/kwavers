//! Transducer array trajectory and phase optimization

use super::planner::TreatmentPlanner;
use super::types::{
    TranscranialTargetVolume, TranscranialTransducerSpecification, TransducerSetup,
};
use crate::transcranial::aberration_correction::TranscranialAberrationCorrection;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;

/// Millimetre→metre factor. `TransducerSpecification` distances (`radius`,
/// `focal_distance`) and the stored `element_positions` are in millimetres
/// (the convention `simulate_acoustic_field` consumes); all physics is computed
/// in metres, the unit of `Grid` spacing and `TargetVolume::center`.
const MM_TO_M: f64 = 1.0e-3;

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
    /// ## Phase Delays (Phase Conjugation + CT Aberration Correction)
    ///
    /// Elements lie on a spherical cap of radius `R = focal_distance` centred on
    /// the target, so each is equidistant (`R`) from the focus. The geometric
    /// phase-conjugation delay on element `i` at position `rᵢ` is
    /// (Daum & Hynynen 1999, Eq. 3):
    ///
    /// ```text
    /// φᵢ_geo = −k · |rᵢ − r_t|     (k = 2π f / c,  = −k·R for the bowl)
    /// ```
    ///
    /// On its own this only corrects for *propagation distance*. Transcranial
    /// focusing additionally requires correcting for the **skull aberration** —
    /// the extra phase each ray accrues crossing bone. The CT phase-screen model
    /// ([`TranscranialAberrationCorrection`], Clement & Hynynen 2002) integrates
    /// `Δφᵢ = ∫(k_local − k_water) ds` along each element→target ray through the
    /// CT volume and returns the conjugate `−Δφᵢ`. The total applied delay is
    ///
    /// ```text
    /// φᵢ = φᵢ_geo + (−Δφᵢ)
    /// ```
    ///
    /// With a homogeneous CT volume every `Δφᵢ` is equal (equidistant rays in a
    /// uniform medium) so the correction is a constant offset and the bowl stays
    /// in phase; a heterogeneous skull makes `Δφᵢ` ray-dependent, which is
    /// exactly the aberration the correction compensates.
    ///
    /// # References
    ///
    /// - Roberts M (2018). "Evenly distributing points on a sphere." *Extreme Learning* (blog).
    /// - Daum DR & Hynynen K (1999). "A 256-element ultrasonic phased array system for
    ///   the treatment of large volumes of deep seated tissue." *IEEE Trans Biomed Eng*
    ///   46(9):1070–1082.
    /// - Clement GT & Hynynen K (2002). "A non-invasive method for focusing ultrasound
    ///   through the skull." *Phys. Med. Biol.* 47(8):1219–1235.
    /// # Errors
    /// - Propagates aberration-correction failures from the CT phase-screen model.
    ///
    pub(crate) fn optimize_transducer_setup(
        &self,
        targets: &[TranscranialTargetVolume],
        spec: &TranscranialTransducerSpecification,
    ) -> KwaversResult<TransducerSetup> {
        let num_elements = spec.num_elements;
        let element_amplitudes = vec![1.0; num_elements];

        // Target centre is in metres (grid frame, from the grid origin corner).
        let target_m = targets[0].center;
        // Bowl radius of curvature in metres (spec distances are millimetres).
        let radius_m = spec.focal_distance * MM_TO_M;

        // Fibonacci hemisphere: uniform angular spacing via golden angle.
        const GOLDEN_RATIO: f64 = 1.618_033_988_749_895; // (1+√5)/2

        // Element positions on the focal sphere, in metres (grid frame). The
        // upper-hemisphere directions place the cap on the +z entry side of the
        // target; every element is exactly `radius_m` from the focus.
        let mut element_positions_m = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let z_norm = (i as f64 + 0.5) / num_elements as f64; // cos θ ∈ (0,1)
            let r_xy = (1.0 - z_norm * z_norm).sqrt();
            let az = TWO_PI * i as f64 / GOLDEN_RATIO;
            element_positions_m.push([
                target_m[0] + radius_m * r_xy * az.cos(),
                target_m[1] + radius_m * r_xy * az.sin(),
                target_m[2] + radius_m * z_norm,
            ]);
        }

        // Geometric phase conjugation: φᵢ_geo = −k · |rᵢ − r_t| (metres).
        let k = TWO_PI * spec.frequency / spec.sound_speed;
        let geometric_phases: Vec<f64> = element_positions_m
            .iter()
            .map(|p| {
                let dx = p[0] - target_m[0];
                let dy = p[1] - target_m[1];
                let dz = p[2] - target_m[2];
                let dist = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
                -k * dist
            })
            .collect();

        // CT skull-aberration correction (phase-screen ray integral). Built with
        // the spec's frequency and coupling speed so its water reference k_water
        // matches the geometric k above.
        let mut corrector = TranscranialAberrationCorrection::new(&self.brain_grid)?;
        corrector.frequency = spec.frequency;
        corrector.reference_speed = spec.sound_speed;
        let correction =
            corrector.calculate_correction(&self.skull_ct, &element_positions_m, &target_m)?;

        // Total applied delay = geometric conjugation + skull-aberration conjugate.
        let element_phases: Vec<f64> = geometric_phases
            .iter()
            .zip(correction.phases.iter())
            .map(|(&geo, &corr)| geo + corr)
            .collect();

        // Store positions in millimetres (the convention consumed by
        // `simulate_acoustic_field`).
        let element_positions: Vec<[f64; 3]> = element_positions_m
            .iter()
            .map(|p| [p[0] / MM_TO_M, p[1] / MM_TO_M, p[2] / MM_TO_M])
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
