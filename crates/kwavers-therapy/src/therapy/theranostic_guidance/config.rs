//! Configuration for same-device therapy and imaging simulations.

use kwavers_core::error::{KwaversError, KwaversResult};

use super::misfit::WaveformMisfit;
use super::transmit_schedule::TransmitScheduleConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnatomyKind {
    Brain,
    Liver,
    Kidney,
}

impl AnatomyKind {
    pub fn from_name(name: &str) -> KwaversResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "brain" => Ok(Self::Brain),
            "liver" => Ok(Self::Liver),
            "kidney" => Ok(Self::Kidney),
            other => Err(KwaversError::InvalidInput(format!(
                "unsupported anatomy '{other}', expected brain, liver, or kidney"
            ))),
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Brain => "brain",
            Self::Liver => "liver",
            Self::Kidney => "kidney",
        }
    }

    #[must_use]
    pub fn default_element_count(self) -> usize {
        match self {
            Self::Brain => 1024,
            Self::Liver | Self::Kidney => 256,
        }
    }

    #[must_use]
    pub fn default_frequencies(self) -> Vec<f64> {
        match self {
            Self::Brain => vec![220_000.0, 650_000.0],
            Self::Liver | Self::Kidney => vec![250_000.0, 500_000.0, 750_000.0],
        }
    }
}

/// Reconstruction strategy for the passive cavitation channels (subharmonic and
/// ultraharmonic).
///
/// The therapy aperture also serves as a passive receive array: cavitation at
/// the focus emits broadband acoustic energy whose subharmonic (f₀/2) and
/// ultraharmonic (3f₀/2) content is recorded and mapped (Gyöngy & Coussios 2010;
/// real-time transcranial histotripsy ACE mapping, Sukovich et al. 2020).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassiveReconstructionMode {
    /// Solve the finite-frequency same-aperture operator against a synthetic
    /// lesion target (linearised normal-equation inverse). Default; preserves
    /// the established subharmonic/ultraharmonic figure and parity contracts.
    FiniteFrequencyOperator,
    /// Genuine passive acoustic mapping: simulate the cavitation acoustic
    /// emission through the heterogeneous medium, record per-receiver time
    /// traces on the imaging aperture, and beamform them with the
    /// delay-multiply-and-sum (DMAS) passive-acoustic-mapping imaging condition.
    PassiveAcousticMapping,
}

impl PassiveReconstructionMode {
    /// Parse a passive-reconstruction mode name (case-insensitive).
    ///
    /// Accepts `"operator"`/`"finite_frequency"` for
    /// [`Self::FiniteFrequencyOperator`] and `"pam"`/`"passive_acoustic_mapping"`
    /// for [`Self::PassiveAcousticMapping`].
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for any other name.
    pub fn from_name(name: &str) -> KwaversResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "operator" | "finite_frequency" | "finite_frequency_operator" => {
                Ok(Self::FiniteFrequencyOperator)
            }
            "pam" | "passive_acoustic_mapping" => Ok(Self::PassiveAcousticMapping),
            other => Err(KwaversError::InvalidInput(format!(
                "unsupported passive_reconstruction '{other}', expected \
                 'operator' or 'pam'"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TheranosticInverseConfig {
    pub anatomy: AnatomyKind,
    pub element_count: usize,
    pub grid_size: usize,
    pub iterations: usize,
    pub frequencies_hz: Vec<f64>,
    pub elastic_frequencies_hz: Vec<f64>,
    pub elastic_shear_speed_m_s: f64,
    pub elastic_fwi_iterations: usize,
    pub receiver_offsets: Vec<usize>,
    pub regularization: f64,
    pub smoothness_weight: f64,
    pub focal_radius_m: f64,
    pub lateral_extent_m: f64,
    pub central_cutout_m: f64,
    pub source_pressure_pa: f64,
    pub lesion_delta_c_m_s: f64,
    pub noise_fraction: f64,
    pub inverse_encoding_rows_per_code: usize,
    pub transmit_schedule: TransmitScheduleConfig,
    pub waveform_misfit: WaveformMisfit,
    pub waveform_misfit_scale_fraction: f64,
    /// Reconstruction strategy for the subharmonic / ultraharmonic passive
    /// cavitation channels. Defaults to [`PassiveReconstructionMode::FiniteFrequencyOperator`].
    pub passive_reconstruction: PassiveReconstructionMode,
}

impl TheranosticInverseConfig {
    #[must_use]
    pub fn new(anatomy: AnatomyKind) -> Self {
        Self {
            anatomy,
            element_count: anatomy.default_element_count(),
            grid_size: 64,
            iterations: 12,
            frequencies_hz: anatomy.default_frequencies(),
            elastic_frequencies_hz: vec![250.0, 500.0, 750.0],
            elastic_shear_speed_m_s: 2.5,
            elastic_fwi_iterations: 3,
            receiver_offsets: vec![32, 64, 96, 128],
            regularization: 1.0e-3,
            smoothness_weight: 6.0e-2,
            focal_radius_m: if matches!(anatomy, AnatomyKind::Brain) {
                0.11
            } else {
                0.142
            },
            lateral_extent_m: if matches!(anatomy, AnatomyKind::Brain) {
                0.22
            } else {
                0.23
            },
            central_cutout_m: 0.04,
            source_pressure_pa: if matches!(anatomy, AnatomyKind::Brain) {
                1.5e5
            } else {
                28.0e6
            },
            lesion_delta_c_m_s: -35.0,
            noise_fraction: 0.012,
            inverse_encoding_rows_per_code: 2,
            transmit_schedule: TransmitScheduleConfig::full(),
            waveform_misfit: WaveformMisfit::Charbonnier,
            waveform_misfit_scale_fraction: 0.012,
            passive_reconstruction: PassiveReconstructionMode::FiniteFrequencyOperator,
        }
    }

    pub fn validate(&self) -> KwaversResult<()> {
        if self.grid_size < 24 {
            return Err(KwaversError::InvalidInput(
                "theranostic grid_size must be at least 24".to_owned(),
            ));
        }
        if self.element_count < 16 {
            return Err(KwaversError::InvalidInput(
                "theranostic element_count must be at least 16".to_owned(),
            ));
        }
        if self.frequencies_hz.is_empty()
            || self
                .frequencies_hz
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "theranostic frequencies must be positive finite values".to_owned(),
            ));
        }
        if self.elastic_frequencies_hz.is_empty()
            || self
                .elastic_frequencies_hz
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "theranostic elastic frequencies must be positive finite values".to_owned(),
            ));
        }
        if self.receiver_offsets.is_empty()
            || self
                .receiver_offsets
                .iter()
                .any(|offset| *offset == 0 || *offset >= self.element_count)
        {
            return Err(KwaversError::InvalidInput(
                "theranostic receiver_offsets must lie in 1..element_count".to_owned(),
            ));
        }
        if self.inverse_encoding_rows_per_code == 0 {
            return Err(KwaversError::InvalidInput(
                "theranostic inverse_encoding_rows_per_code must be at least 1".to_owned(),
            ));
        }
        self.transmit_schedule.validate(self.element_count)?;
        for (name, value) in [
            ("regularization", self.regularization),
            ("smoothness_weight", self.smoothness_weight),
            ("focal_radius_m", self.focal_radius_m),
            ("source_pressure_pa", self.source_pressure_pa),
            ("noise_fraction", self.noise_fraction),
            (
                "waveform_misfit_scale_fraction",
                self.waveform_misfit_scale_fraction,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if !self.elastic_shear_speed_m_s.is_finite() || self.elastic_shear_speed_m_s <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "theranostic elastic_shear_speed_m_s must be positive and finite".to_owned(),
            ));
        }
        if self.elastic_fwi_iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "theranostic elastic_fwi_iterations must be at least 1".to_owned(),
            ));
        }
        if matches!(self.waveform_misfit, WaveformMisfit::Charbonnier)
            && self.waveform_misfit_scale_fraction <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "charbonnier waveform misfit requires positive scale fraction".to_owned(),
            ));
        }
        Ok(())
    }
}