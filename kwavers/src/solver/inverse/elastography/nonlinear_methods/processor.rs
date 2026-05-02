//! `NonlinearInversion` — public dispatcher for all nonlinear methods.

use super::super::config::NonlinearInversionConfig;
use super::bayesian::bayesian_inversion;
use super::harmonic_ratio::harmonic_ratio_inversion;
use super::least_squares::nonlinear_least_squares_inversion;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::NonlinearParameterMap;
use crate::physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;

/// Nonlinear parameter inversion processor
#[derive(Debug)]
pub struct NonlinearInversion {
    config: NonlinearInversionConfig,
}

impl NonlinearInversion {
    pub fn new(config: NonlinearInversionConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn method(
        &self,
    ) -> crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod {
        self.config.method
    }

    #[must_use]
    pub fn config(&self) -> &NonlinearInversionConfig {
        &self.config
    }

    pub fn reconstruct(
        &self,
        harmonic_field: &HarmonicDisplacementField,
        grid: &Grid,
    ) -> KwaversResult<NonlinearParameterMap> {
        use crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod;

        match self.config.method {
            NonlinearInversionMethod::HarmonicRatio => {
                harmonic_ratio_inversion(harmonic_field, grid, &self.config)
            }
            NonlinearInversionMethod::NonlinearLeastSquares => {
                nonlinear_least_squares_inversion(harmonic_field, grid, &self.config)
            }
            NonlinearInversionMethod::BayesianInversion => {
                bayesian_inversion(harmonic_field, grid, &self.config)
            }
        }
    }
}
