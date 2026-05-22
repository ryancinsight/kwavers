//! Numerical configuration for linear Born + PCG inversion.
//!
//! Anatomy- and transducer-neutral knobs only. The number and physical layout
//! of array elements live on the [`TransducerGeometry`](super::TransducerGeometry)
//! impl passed alongside this config; clinical adapters
//! (`clinical::imaging::reconstruction::*`) compose this struct with
//! anatomy-specific parameters into their own configs.
//!
//! # Measurement-cardinality invariant
//!
//! For an acquisition with `N` source elements, `Q` receiver offsets, `F`
//! frequencies, and `H` harmonic channels, the row set is the Cartesian product
//! `source × offset × frequency × harmonic`. Therefore the sensitivity matrix
//! has exactly `N·Q·F·H` rows. [`LinearBornInversionConfig::measurement_count`]
//! is the single implementation of that invariant; geometry supplies only `N`.

use crate::core::constants::fundamental::SOUND_SPEED_BRAIN;
use crate::core::error::{KwaversError, KwaversResult};

/// Generic linear Born + PCG inversion settings.
#[derive(Clone, Debug)]
pub struct LinearBornInversionConfig {
    /// Frequencies used by the encoded finite-frequency sensitivity [Hz].
    pub frequencies_hz: Vec<f64>,
    /// Receiver offsets from each emitting element. Semantics depend on the
    /// transducer geometry (cyclic for rings; azimuthal for bowls); see
    /// [`TransducerGeometry::receiver_indices`](super::TransducerGeometry::receiver_indices).
    pub receiver_offsets: Vec<usize>,
    /// Maximum outer PCG / Landweber iterations.
    pub iterations: usize,
    /// Initial dimensionless relaxation for the normalized gradient step.
    pub relaxation: f64,
    /// Tikhonov weight on slowness contrast.
    pub regularization: f64,
    /// Use low-to-high frequency continuation before the full-band pass.
    pub frequency_continuation: bool,
    /// Radius of the Sobolev gradient smoother in active voxels.
    pub sobolev_radius_voxels: usize,
    /// Convex weight applied to the Sobolev-smoothed update direction.
    pub sobolev_weight: f64,
    /// High-boost gain for the returned structure-enhanced display image.
    pub enhancement_gain: f64,
    /// Edge-preserving (Charbonnier) first-difference regularization weight.
    pub edge_preserving_weight: f64,
    /// Charbonnier transition scale `ε`.
    pub edge_preserving_epsilon: f64,
    /// Stable convex-projection step for edge-preserving proximal smoothing.
    pub edge_preserving_step: f64,
    /// Number of accepted proximal smoothing passes per PCG iteration.
    pub edge_preserving_iterations: usize,
    /// Apply medium-derived frequency-dependent path attenuation in sensitivity
    /// rows.
    pub attenuation_model: bool,
    /// Include weak-Westervelt second-harmonic encoded rows.
    pub nonlinear_harmonic_model: bool,
    /// Source pressure used for weak-nonlinear harmonic scaling [MPa].
    pub source_pressure_mpa: f64,
    /// Acoustic nonlinearity coefficient `β = 1 + B/(2A)`.
    pub nonlinear_beta: f64,
    /// Lower bound for reconstructed fractional speed contrast.
    pub contrast_min: f64,
    /// Upper bound for reconstructed fractional speed contrast.
    pub contrast_max: f64,
    /// Reference sound speed `c₀` used to convert frequency `f` to wavenumber
    /// `k = 2π f / c₀` and to anchor fractional speed contrasts [m/s]. Set by
    /// the clinical adapter to the anatomy-appropriate reference (e.g. brain,
    /// breast, abdomen).
    pub reference_sound_speed_m_s: f64,
    /// Reference medium density `ρ₀` used in the weak-shock distance
    /// `z_s = ρ₀ c₀³ / (β ω p₀)` for the second-harmonic scaling [kg/m³].
    pub reference_density_kg_m3: f64,
}

impl Default for LinearBornInversionConfig {
    fn default() -> Self {
        Self {
            frequencies_hz: vec![200_000.0, 350_000.0, 500_000.0, 650_000.0, 800_000.0],
            receiver_offsets: vec![512, 384, 640, 256, 768, 128, 448, 576],
            iterations: 24,
            relaxation: 0.85,
            regularization: 1.0e-4,
            frequency_continuation: true,
            sobolev_radius_voxels: 1,
            sobolev_weight: 0.35,
            enhancement_gain: 0.65,
            edge_preserving_weight: 1.0e-4,
            edge_preserving_epsilon: 0.004,
            edge_preserving_step: 0.12,
            edge_preserving_iterations: 1,
            attenuation_model: true,
            nonlinear_harmonic_model: true,
            source_pressure_mpa: 0.15,
            nonlinear_beta: 4.5,
            contrast_min: -0.08,
            contrast_max: 0.08,
            // Soft-tissue reference (brain/breast): SOUND_SPEED_BRAIN = 1546 m/s,
            // ρ₀ ≈ 1000 kg/m³. Adapters that target a different anatomy
            // override these explicitly at construction.
            reference_sound_speed_m_s: SOUND_SPEED_BRAIN,
            reference_density_kg_m3: 1000.0,
        }
    }
}

impl LinearBornInversionConfig {
    /// Validate numerical invariants. Called by clinical-adapter `validate` after
    /// the adapter's own anatomy/geometry-specific checks.
    ///
    /// # Errors
    /// Returns an error when any numerical knob is outside its admissible range.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.frequencies_hz.is_empty()
            || self
                .frequencies_hz
                .iter()
                .any(|f| !f.is_finite() || *f <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.frequencies_hz must contain positive values".to_owned(),
            ));
        }
        if self.receiver_offsets.is_empty()
            || self.receiver_offsets.iter().any(|offset| *offset == 0)
        {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.receiver_offsets must be nonempty and nonzero"
                    .to_owned(),
            ));
        }
        if self.iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.iterations must be positive".to_owned(),
            ));
        }
        if !self.relaxation.is_finite() || self.relaxation <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.relaxation must be finite and positive".to_owned(),
            ));
        }
        if !self.regularization.is_finite() || self.regularization < 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.regularization must be finite and non-negative"
                    .to_owned(),
            ));
        }
        if !self.sobolev_weight.is_finite() || !(0.0..=1.0).contains(&self.sobolev_weight) {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.sobolev_weight must be in [0, 1]".to_owned(),
            ));
        }
        if !self.enhancement_gain.is_finite() || self.enhancement_gain < 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.enhancement_gain must be finite and non-negative"
                    .to_owned(),
            ));
        }
        if !self.edge_preserving_weight.is_finite() || self.edge_preserving_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.edge_preserving_weight must be finite and non-negative"
                    .to_owned(),
            ));
        }
        if !self.edge_preserving_epsilon.is_finite() || self.edge_preserving_epsilon <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.edge_preserving_epsilon must be finite and positive"
                    .to_owned(),
            ));
        }
        if !self.edge_preserving_step.is_finite()
            || !(0.0..=1.0).contains(&self.edge_preserving_step)
        {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.edge_preserving_step must be in [0, 1]".to_owned(),
            ));
        }
        if !self.source_pressure_mpa.is_finite() || self.source_pressure_mpa <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.source_pressure_mpa must be finite and positive"
                    .to_owned(),
            ));
        }
        if !self.nonlinear_beta.is_finite() || self.nonlinear_beta <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.nonlinear_beta must be finite and positive".to_owned(),
            ));
        }
        if !self.reference_sound_speed_m_s.is_finite() || self.reference_sound_speed_m_s <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.reference_sound_speed_m_s must be finite and positive"
                    .to_owned(),
            ));
        }
        if !self.reference_density_kg_m3.is_finite() || self.reference_density_kg_m3 <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LinearBornInversionConfig.reference_density_kg_m3 must be finite and positive"
                    .to_owned(),
            ));
        }
        if self.contrast_min >= self.contrast_max {
            return Err(KwaversError::InvalidInput(
                "contrast_min must be lower than contrast_max".to_owned(),
            ));
        }
        Ok(())
    }

    /// Number of harmonic channels per source/offset/frequency acquisition.
    #[must_use]
    pub fn harmonic_count(&self) -> usize {
        if self.nonlinear_harmonic_model {
            2
        } else {
            1
        }
    }

    /// Number of encoded finite-frequency measurements for a given element count.
    #[must_use]
    pub fn measurement_count(&self, element_count: usize) -> usize {
        element_count
            * self.receiver_offsets.len()
            * self.frequencies_hz.len()
            * self.harmonic_count()
    }
}
