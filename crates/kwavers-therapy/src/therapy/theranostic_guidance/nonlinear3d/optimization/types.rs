//! Data types for the FWI line-search update.

use crate::therapy::theranostic_guidance::nonlinear3d::{
    encoding::EncodedTrace,
    forward::TimeSchedule,
    types::{Nonlinear3dAperture, Nonlinear3dConfig},
};

pub(in crate::therapy::theranostic_guidance::nonlinear3d) struct LineSearchInput<'a> {
    pub current_speed: &'a mut [f64],
    pub current_beta: &'a mut [f64],
    pub workspace: &'a mut LineSearchWorkspace,
    pub background_speed: &'a [f64],
    pub background_beta: &'a [f64],
    pub body: &'a [bool],
    pub grad_speed: &'a [f64],
    pub grad_beta: &'a [f64],
    pub objective: f64,
    pub observed: &'a [EncodedTrace],
    pub observed_energy: f64,
    pub density: &'a [f64],
    pub attenuation_np_per_m_mhz: &'a [f64],
    pub attenuation_power_law_y: &'a [f64],
    pub source_body_mask: &'a [bool],
    pub n: usize,
    pub spacing_m: f64,
    pub aperture: &'a Nonlinear3dAperture,
    pub config: &'a Nonlinear3dConfig,
    pub schedule: TimeSchedule,
    pub source_scale: f64,
}

#[derive(Clone, Debug)]
pub(in crate::therapy::theranostic_guidance::nonlinear3d) struct LineSearchWorkspace {
    pub(super) candidate_speed: Vec<f64>,
    pub(super) candidate_beta: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::therapy::theranostic_guidance::nonlinear3d) enum AcceptedBlock {
    Coupled,
    SpeedOnly,
    BetaOnly,
}

impl AcceptedBlock {
    pub(in crate::therapy::theranostic_guidance::nonlinear3d) fn label(self) -> &'static str {
        match self {
            Self::Coupled => "coupled",
            Self::SpeedOnly => "speed_only",
            Self::BetaOnly => "beta_only",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::therapy::theranostic_guidance::nonlinear3d) struct LineSearchOutcome {
    pub(in crate::therapy::theranostic_guidance::nonlinear3d) accepted_block: AcceptedBlock,
    pub(in crate::therapy::theranostic_guidance::nonlinear3d) scale: f64,
    pub(in crate::therapy::theranostic_guidance::nonlinear3d) objective: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CandidateAcceptance {
    pub(super) block: AcceptedBlock,
    pub(super) objective: f64,
}

impl LineSearchWorkspace {
    #[must_use]
    pub(in crate::therapy::theranostic_guidance::nonlinear3d) fn new(cells: usize) -> Self {
        Self {
            candidate_speed: vec![0.0; cells],
            candidate_beta: vec![0.0; cells],
        }
    }

    pub(super) fn resize_for(&mut self, cells: usize) {
        self.candidate_speed.resize(cells, 0.0);
        self.candidate_beta.resize(cells, 0.0);
    }
}

pub(in crate::therapy::theranostic_guidance::nonlinear3d) struct ObjectiveInput<'a> {
    pub observed: &'a [EncodedTrace],
    pub observed_energy: f64,
    pub density: &'a [f64],
    pub attenuation_np_per_m_mhz: &'a [f64],
    pub attenuation_power_law_y: &'a [f64],
    pub background_speed: &'a [f64],
    pub background_beta: &'a [f64],
    pub body: &'a [bool],
    pub source_body_mask: &'a [bool],
    pub n: usize,
    pub spacing_m: f64,
    pub aperture: &'a Nonlinear3dAperture,
    pub config: &'a Nonlinear3dConfig,
    pub schedule: TimeSchedule,
    pub source_scale: f64,
}
