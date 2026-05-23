//! Public types for frequency-domain FWI.

use std::sync::Arc;

use ndarray::{Array2, Array3};
use num_complex::Complex64;

use super::operator::{HelmholtzForwardOperator, SingleScatterBornOperator};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

/// Solver model identifier for audit trails.
pub const FREQUENCY_DOMAIN_FWI_SOLVER_MODEL: &str = "frequency_domain_fwi";

/// Observed complex pressure for one frequency.
#[derive(Clone, Debug)]
pub struct FrequencyObservation {
    /// Frequency [Hz].
    pub frequency_hz: f64,
    /// Complex receiver data with shape `(transmit, receiver)`.
    pub observed_pressure: Array2<Complex64>,
}

impl FrequencyObservation {
    #[must_use]
    pub fn new(frequency_hz: f64, observed_pressure: Array2<Complex64>) -> Self {
        Self {
            frequency_hz,
            observed_pressure,
        }
    }
}

/// Nonlinear FWI settings.
///
/// The forward-operator selection lives on the dyn-dispatched `forward_operator`
/// field. Three impls ship in `super::operator`: `SingleScatterBornOperator`,
/// `DenseConvergentBornOperator`, `SpectralConvergentBornOperator`, and
/// `PstdSpectralConvergentBornOperator`. Adding a new operator
/// (BiCGSTAB-preconditioned Helmholtz, sparse-direct, FEM Helmholtz) is
/// `impl HelmholtzForwardOperator`; no config-enum edit required.
#[derive(Clone, Debug)]
pub struct Config {
    /// Reference homogeneous sound speed [m/s].
    pub reference_sound_speed_m_s: f64,
    /// Uniform reconstruction voxel spacing [m].
    pub spacing_m: f64,
    /// Nonlinear conjugate-gradient iterations.
    pub iterations: usize,
    /// Maximum slowness step before line-search halving [s/m].
    pub initial_step_s_per_m: f64,
    /// Minimum admissible sound speed [m/s].
    pub min_sound_speed_m_s: f64,
    /// Maximum admissible sound speed [m/s].
    pub max_sound_speed_m_s: f64,
    /// Estimate one complex source scale per frequency/transmit row.
    pub estimate_source_scaling: bool,
    /// Tikhonov weight around the reference slowness.
    pub tikhonov_weight: f64,
    /// Forward Helmholtz operator. Drives both prediction and the matching
    /// gradient path (single-scatter analytical sensitivity vs CBS
    /// volume-field adjoint).
    pub forward_operator: Arc<dyn HelmholtzForwardOperator>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
            spacing_m: 1.0e-3,
            iterations: 5,
            initial_step_s_per_m: 2.0e-6,
            min_sound_speed_m_s: 1400.0,
            max_sound_speed_m_s: 1600.0,
            estimate_source_scaling: true,
            tikhonov_weight: 0.0,
            forward_operator: Arc::new(SingleScatterBornOperator),
        }
    }
}

impl Config {
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    #[must_use]
    pub fn with_spacing_m(mut self, spacing_m: f64) -> Self {
        self.spacing_m = spacing_m;
        self
    }

    #[must_use]
    pub fn with_source_scaling(mut self, estimate_source_scaling: bool) -> Self {
        self.estimate_source_scaling = estimate_source_scaling;
        self
    }

    #[must_use]
    pub fn with_forward_operator(
        mut self,
        forward_operator: Arc<dyn HelmholtzForwardOperator>,
    ) -> Self {
        self.forward_operator = forward_operator;
        self
    }
}

/// Reconstructed sound-speed volume and convergence diagnostics.
#[derive(Clone, Debug)]
pub struct InversionResult {
    /// Reconstructed sound speed [m/s].
    pub sound_speed_m_s: Array3<f64>,
    /// Objective value after the initial model and each accepted update.
    pub objective_history: Vec<f64>,
    /// Frequencies used by the inversion.
    pub frequencies_used: usize,
    /// Transmit rows used per frequency.
    pub transmissions_used: usize,
    /// Receiver count per transmit row.
    pub receivers_used: usize,
    /// Model identifier for audit trails.
    pub model_family: &'static str,
}
