//! One validated experiment and its model-independent Born coefficients.

use super::super::acquisition::TransmissionAcquisition;
use super::super::born_geometry::{products::WORKSPACE_BYTES, BornGeometry};
use super::super::forward::validate_forward_inputs;
use super::super::types::{Config, FrequencyObservation};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

#[cfg(test)]
mod tests;

/// Owns immutable geometry for one inversion; public operators stay stateless.
/// Every distinct model is evaluated afresh, including finite-difference
/// Hessian probes and the coupled source fit. An identical projected Born
/// probe permits the exact zero-difference identity. No tolerance changes.
pub(in super::super) struct Objective<'a> {
    pub(in super::super) config: &'a Config,
    observations: &'a [FrequencyObservation],
    acquisition: &'a dyn TransmissionAcquisition,
    shape: [usize; 3],
    born: Option<Vec<BornGeometry>>,
}

impl<'a> Objective<'a> {
    /// Prepared Born coefficients and IEEE polynomial arithmetic give the same
    /// gradient bits for the same model. Other operators retain their full
    /// evaluation because their transcendental precision is unspecified.
    pub(super) fn has_repeatable_gradient(&self) -> bool {
        self.born.is_some()
    }

    pub(in super::super) fn new(
        model: &Array3<f64>,
        observations: &'a [FrequencyObservation],
        acquisition: &'a dyn TransmissionAcquisition,
        config: &'a Config,
    ) -> KwaversResult<Self> {
        let mut result = Self {
            config,
            observations,
            acquisition,
            shape: model.shape(),
            born: None,
        };
        result.validate(model)?;
        if !config.forward_operator.uses_finite_window_adjoint()
            && !config.forward_operator.uses_volume_field_adjoint()
        {
            result.born = Some(
                observations
                    .iter()
                    .map(|observation| {
                        let mut geometry = BornGeometry::new(
                            result.shape,
                            observation.frequency_hz,
                            acquisition,
                            config,
                            observation.observed_pressure.shape()[0],
                        );
                        if !config.estimate_source_scaling {
                            geometry.prepare_products(WORKSPACE_BYTES / observations.len());
                        }
                        geometry
                    })
                    .collect(),
            );
        }
        Ok(result)
    }

    fn validate(&self, model: &Array3<f64>) -> KwaversResult<()> {
        if self.observations.is_empty() {
            return Err(KwaversError::InvalidInput(
                "FWI requires at least one frequency observation".to_owned(),
            ));
        }
        if model.shape() != self.shape {
            return Err(KwaversError::DimensionMismatch(format!(
                "prepared objective shape {:?}, model shape {:?}",
                self.shape,
                model.shape()
            )));
        }
        for observation in self.observations {
            let rows = observation.observed_pressure.shape()[0];
            if observation.observed_pressure.shape()[1] != self.acquisition.receiver_count() {
                return Err(KwaversError::DimensionMismatch(format!(
                    "receiver count mismatch: observed {}, geometry {}",
                    observation.observed_pressure.shape()[1],
                    self.acquisition.receiver_count()
                )));
            }
            validate_forward_inputs(
                model,
                self.acquisition,
                observation.frequency_hz,
                self.config,
                rows,
            )?;
        }
        Ok(())
    }

    pub(in super::super) fn evaluate(
        &self,
        model: &Array3<f64>,
    ) -> KwaversResult<(f64, Array3<f64>)> {
        self.validate(model)?;
        let mut objective = 0.0;
        let mut gradient = Array3::zeros(self.shape);
        for (index, observation) in self.observations.iter().enumerate() {
            if self.config.forward_operator.uses_finite_window_adjoint() {
                super::accumulate_finite_window_frequency_gradient(
                    model,
                    observation,
                    self.acquisition,
                    self.config,
                    &mut objective,
                    &mut gradient,
                )?;
            } else if self.config.forward_operator.uses_volume_field_adjoint() {
                super::accumulate_dense_cbs_frequency_gradient(
                    model,
                    observation,
                    self.acquisition,
                    self.config,
                    &mut objective,
                    &mut gradient,
                )?;
            } else {
                let geometry = &self
                    .born
                    .as_ref()
                    .expect("Born operator has prepared geometry")[index];
                super::accumulate_frequency_gradient(
                    model,
                    observation,
                    geometry,
                    self.config,
                    &mut objective,
                    &mut gradient,
                )?;
            }
        }
        super::apply_tikhonov(model, self.config, &mut objective, &mut gradient);
        Ok((objective, gradient))
    }
}
