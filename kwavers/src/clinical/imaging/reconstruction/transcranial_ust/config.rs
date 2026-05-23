//! Configuration for transcranial ultrasound tomography (clinical adapter).
//!
//! Wraps the solver-layer
//! [`LinearBornInversionConfig`](crate::solver::inverse::linear_born_inversion::LinearBornInversionConfig)
//! with anatomy/transducer-specific parameters (element count, focused-bowl
//! radius) and the brain-tissue acoustic constants used by the CT-to-speed
//! model. The generic numerical knobs (Tikhonov, Sobolev, edge-preserving
//! Charbonnier, harmonic encoding) live in the embedded `linear` field and
//! are passed to the generic linear-Born kernels as
//! `&config.linear`.

use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::inverse::linear_born_inversion::LinearBornInversionConfig;

/// Reference element count for the transcranial focused-bowl acquisition.
pub const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;

use crate::core::constants::acoustic_parameters::SOUND_SPEED_SKULL;
use crate::core::constants::fundamental::{
    DENSITY_BRAIN, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM,
};

/// Acoustic properties used by the CT-to-speed model.
pub const C_WATER_M_S: f64 = SOUND_SPEED_WATER_SIM;
pub const C_BRAIN_REF_M_S: f64 = SOUND_SPEED_TISSUE;
pub const C_BONE_M_S: f64 = SOUND_SPEED_SKULL;

/// Clinical configuration for the transcranial UST finite-frequency Born
/// inversion.
///
/// = generic [`LinearBornInversionConfig`] + transducer-geometry parameters
/// (`element_count`, `radius_m`). The clinical adapter constructs the
/// focused-bowl geometry from `element_count + radius_m`, then passes
/// `&self.linear` to every kernel call so the kernels remain anatomy-neutral.
#[derive(Clone, Debug)]
pub struct TranscranialUstBornInversionConfig {
    /// Generic linear-Born + PCG inversion knobs.
    pub linear: LinearBornInversionConfig,
    /// Number of array elements placed on the transcranial focused bowl.
    pub element_count: usize,
    /// Transcranial focused-bowl radius around the CT volume center [m].
    pub radius_m: f64,
}

impl Default for TranscranialUstBornInversionConfig {
    fn default() -> Self {
        // Brain-tissue reference: c₀ = C_BRAIN_REF_M_S (1546 m/s), ρ₀ = DENSITY_BRAIN (1040 kg/m³).
        // Pins the generic operator's wavenumber and weak-shock distance to the
        // brain anatomy this clinical adapter targets.
        let linear = LinearBornInversionConfig {
            reference_sound_speed_m_s: C_BRAIN_REF_M_S,
            reference_density_kg_m3: DENSITY_BRAIN,
            ..LinearBornInversionConfig::default()
        };
        Self {
            linear,
            element_count: TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT,
            radius_m: 0.11,
        }
    }
}

impl TranscranialUstBornInversionConfig {
    /// Validate the clinical-adapter invariants and delegate the numerical
    /// invariants to [`LinearBornInversionConfig::validate`].
    ///
    /// # Errors
    /// Returns an error when any anatomy or numerical invariant is violated.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.element_count < 8 {
            return Err(KwaversError::InvalidInput(
                "TranscranialUstBornInversionConfig.element_count must be at least 8".to_owned(),
            ));
        }
        if !self.radius_m.is_finite() || self.radius_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "TranscranialUstBornInversionConfig.radius_m must be finite and positive"
                    .to_owned(),
            ));
        }
        if self
            .linear
            .receiver_offsets
            .iter()
            .any(|offset| *offset >= self.element_count)
        {
            return Err(KwaversError::InvalidInput(
                "receiver offsets must lie in 1..element_count".to_owned(),
            ));
        }
        self.linear.validate()
    }

    /// Number of harmonic channels per source/offset/frequency acquisition.
    #[must_use]
    pub fn harmonic_count(&self) -> usize {
        self.linear.harmonic_count()
    }

    /// Number of encoded finite-frequency measurements for this clinical
    /// acquisition (`element_count × receiver_offsets × frequencies × harmonics`).
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.linear.measurement_count(self.element_count)
    }
}
