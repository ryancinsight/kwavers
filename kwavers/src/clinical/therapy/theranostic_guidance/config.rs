//! Configuration for same-device therapy and imaging simulations.

use crate::core::error::{KwaversError, KwaversResult};

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

#[derive(Clone, Debug)]
pub struct TheranosticFwiConfig {
    pub anatomy: AnatomyKind,
    pub element_count: usize,
    pub grid_size: usize,
    pub iterations: usize,
    pub frequencies_hz: Vec<f64>,
    pub receiver_offsets: Vec<usize>,
    pub regularization: f64,
    pub smoothness_weight: f64,
    pub focal_radius_m: f64,
    pub lateral_extent_m: f64,
    pub central_cutout_m: f64,
    pub source_pressure_pa: f64,
    pub lesion_delta_c_m_s: f64,
    pub noise_fraction: f64,
}

impl TheranosticFwiConfig {
    #[must_use]
    pub fn new(anatomy: AnatomyKind) -> Self {
        Self {
            anatomy,
            element_count: anatomy.default_element_count(),
            grid_size: 64,
            iterations: 12,
            frequencies_hz: anatomy.default_frequencies(),
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
        for (name, value) in [
            ("regularization", self.regularization),
            ("smoothness_weight", self.smoothness_weight),
            ("focal_radius_m", self.focal_radius_m),
            ("source_pressure_pa", self.source_pressure_pa),
            ("noise_fraction", self.noise_fraction),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}
