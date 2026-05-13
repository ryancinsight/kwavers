//! Configuration for 1024-element transcranial brain FWI.

use crate::core::error::{KwaversError, KwaversResult};

/// Reference element count for an INSIGHTEC-style hemispherical array.
pub const INSIGHTEC_ELEMENT_COUNT: usize = 1024;

/// Acoustic properties used by the CT-to-speed model.
pub const C_WATER_M_S: f64 = 1500.0;
pub const C_BRAIN_REF_M_S: f64 = 1540.0;
pub const C_BONE_M_S: f64 = 2900.0;

/// Numerical configuration for the finite-frequency encoded inversion.
#[derive(Clone, Debug)]
pub struct BrainHelmetFwiConfig {
    /// Number of array elements placed on the hemispherical helmet.
    pub element_count: usize,
    /// Hemispherical helmet radius around the CT volume center [m].
    pub radius_m: f64,
    /// Frequencies used by the encoded finite-frequency sensitivity [Hz].
    pub frequencies_hz: Vec<f64>,
    /// Receiver offsets from each emitting element, modulo `element_count`.
    pub receiver_offsets: Vec<usize>,
    /// Maximum Landweber/backtracking iterations.
    pub iterations: usize,
    /// Initial dimensionless relaxation for the normalized gradient step.
    pub relaxation: f64,
    /// Tikhonov weight on sound-speed contrast.
    pub regularization: f64,
    /// Use low-to-high frequency continuation before the full-band pass.
    pub frequency_continuation: bool,
    /// Radius of the Sobolev gradient smoother in active brain voxels.
    pub sobolev_radius_voxels: usize,
    /// Convex weight applied to the Sobolev-smoothed update direction.
    pub sobolev_weight: f64,
    /// High-boost gain for the returned structure-enhanced display image.
    pub enhancement_gain: f64,
    /// Edge-preserving first-difference regularization weight.
    pub edge_preserving_weight: f64,
    /// Charbonnier transition scale for edge-preserving regularization.
    pub edge_preserving_epsilon: f64,
    /// Stable convex-projection step for edge-preserving proximal smoothing.
    pub edge_preserving_step: f64,
    /// Number of accepted proximal smoothing passes per PCG iteration.
    pub edge_preserving_iterations: usize,
    /// Apply CT-derived frequency-dependent path attenuation in sensitivity rows.
    pub attenuation_model: bool,
    /// Include weak-Westervelt second-harmonic encoded rows.
    pub nonlinear_harmonic_model: bool,
    /// Source pressure used for weak-nonlinear harmonic scaling [MPa].
    pub source_pressure_mpa: f64,
    /// Acoustic nonlinearity coefficient beta = 1 + B/(2A).
    pub nonlinear_beta: f64,
    /// Lower bound for reconstructed fractional speed contrast.
    pub contrast_min: f64,
    /// Upper bound for reconstructed fractional speed contrast.
    pub contrast_max: f64,
}

impl Default for BrainHelmetFwiConfig {
    fn default() -> Self {
        Self {
            element_count: INSIGHTEC_ELEMENT_COUNT,
            radius_m: 0.11,
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
        }
    }
}

impl BrainHelmetFwiConfig {
    /// Validate configuration invariants before matrix construction.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.element_count < 8 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.element_count must be at least 8".to_owned(),
            ));
        }
        if !self.radius_m.is_finite() || self.radius_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.radius_m must be finite and positive".to_owned(),
            ));
        }
        if self.frequencies_hz.is_empty()
            || self
                .frequencies_hz
                .iter()
                .any(|f| !f.is_finite() || *f <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.frequencies_hz must contain positive values".to_owned(),
            ));
        }
        if self.receiver_offsets.is_empty()
            || self
                .receiver_offsets
                .iter()
                .any(|offset| *offset == 0 || *offset >= self.element_count)
        {
            return Err(KwaversError::InvalidInput(
                "receiver offsets must lie in 1..element_count".to_owned(),
            ));
        }
        if self.iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.iterations must be positive".to_owned(),
            ));
        }
        if !self.relaxation.is_finite() || self.relaxation <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.relaxation must be finite and positive".to_owned(),
            ));
        }
        if !self.regularization.is_finite() || self.regularization < 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.regularization must be finite and non-negative".to_owned(),
            ));
        }
        if !self.sobolev_weight.is_finite() || !(0.0..=1.0).contains(&self.sobolev_weight) {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.sobolev_weight must be in [0, 1]".to_owned(),
            ));
        }
        if !self.enhancement_gain.is_finite() || self.enhancement_gain < 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.enhancement_gain must be finite and non-negative".to_owned(),
            ));
        }
        if !self.edge_preserving_weight.is_finite() || self.edge_preserving_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.edge_preserving_weight must be finite and non-negative"
                    .to_owned(),
            ));
        }
        if !self.edge_preserving_epsilon.is_finite() || self.edge_preserving_epsilon <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.edge_preserving_epsilon must be finite and positive"
                    .to_owned(),
            ));
        }
        if !self.edge_preserving_step.is_finite()
            || !(0.0..=1.0).contains(&self.edge_preserving_step)
        {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.edge_preserving_step must be in [0, 1]".to_owned(),
            ));
        }
        if !self.source_pressure_mpa.is_finite() || self.source_pressure_mpa <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.source_pressure_mpa must be finite and positive".to_owned(),
            ));
        }
        if !self.nonlinear_beta.is_finite() || self.nonlinear_beta <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BrainHelmetFwiConfig.nonlinear_beta must be finite and positive".to_owned(),
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

    /// Number of encoded finite-frequency measurements.
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.element_count
            * self.receiver_offsets.len()
            * self.frequencies_hz.len()
            * self.harmonic_count()
    }
}
