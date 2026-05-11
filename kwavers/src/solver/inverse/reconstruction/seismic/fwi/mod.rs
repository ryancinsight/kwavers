//! Full Waveform Inversion (FWI) implementation
//!
//! Based on:
//! - Virieux & Operto (2009): "An overview of full-waveform inversion"
//! - Tarantola (1984): "Inversion of seismic reflection data"

pub mod gradient;
pub mod optimization;
pub mod regularization;
pub mod wavefield;

use self::optimization::ConjugateGradient;
use self::regularization::Regularizer;
use self::wavefield::WavefieldModeler;

use super::config::SeismicImagingConfig;
use crate::core::error::KwaversResult;
use crate::solver::inverse::acoustic_fwi::{l2_objective, l2_residual};
use ndarray::{Array2, Array3};

/// Full Waveform Inversion (FWI) reconstructor
#[derive(Debug)]
pub struct FullWaveformInversion {
    config: SeismicImagingConfig,
    /// Current velocity model
    velocity_model: Array3<f64>,
    /// Optimizer
    optimizer: ConjugateGradient,
    /// Regularizer
    regularizer: Regularizer,
    /// Wavefield modeler
    wavefield_modeler: WavefieldModeler,
}

impl FullWaveformInversion {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: SeismicImagingConfig) -> Self {
        let shape = (config.nx, config.ny, config.nz);
        Self {
            config,
            velocity_model: Array3::zeros(shape),
            optimizer: ConjugateGradient::new(),
            regularizer: Regularizer::new(),
            wavefield_modeler: WavefieldModeler::new(),
        }
    }

    /// Main FWI iteration
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn iterate(&mut self, observed_data: &Array2<f64>) -> KwaversResult<()> {
        // 1. Forward modeling
        let synthetic_data = self.wavefield_modeler.forward_model(&self.velocity_model)?;

        // 2. Compute residual and objective from the same data pair
        let residual = self.compute_adjoint_source(observed_data, &synthetic_data)?;
        let current_objective = self.compute_data_objective(observed_data, &synthetic_data)?;

        // 3. Compute the exact discrete gradient via the adjoint method.
        let mut gradient = self
            .wavefield_modeler
            .adjoint_model(&self.velocity_model, &residual)?;

        // 4. Apply regularization
        self.regularizer
            .apply_regularization(&mut gradient, &self.velocity_model);

        // 5. Compute search direction
        let direction = self.optimizer.compute_direction(&gradient);

        // 6. Line search for step size using Armijo-Wolfe conditions
        // Wolfe conditions ensure sufficient decrease and curvature conditions
        // Reference: Nocedal & Wright (2006) "Numerical Optimization", Chapter 3
        // Implement backtracking line search with Armijo condition
        let mut step_size = 1.0;
        let c1 = 1e-4; // Armijo constant
        let max_iterations = 20;
        let backtrack_factor = 0.5;

        // Compute directional derivative
        let directional_derivative = gradient
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum::<f64>();

        // Backtracking line search
        for _ in 0..max_iterations {
            let test_model = &self.velocity_model + step_size * &direction;
            let test_objective = self.compute_objective(&test_model, observed_data)?;

            // Armijo condition: f(x + αp) ≤ f(x) + c1·α·∇f(x)ᵀp
            if test_objective <= (c1 * step_size).mul_add(directional_derivative, current_objective) {
                break;
            }

            step_size *= backtrack_factor;
        }

        // 7. Update model
        self.velocity_model = &self.velocity_model + step_size * &direction;

        Ok(())
    }

    /// Multi-scale FWI strategy
    /// Based on Bunks et al. (1995): "Multiscale seismic waveform inversion"
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn multiscale_inversion(
        &mut self,
        observed_data: &Array2<f64>,
        frequency_bands: &[(f64, f64)],
    ) -> KwaversResult<()> {
        for (f_min, f_max) in frequency_bands {
            // Apply frequency band filter using Butterworth filter
            let filtered_data = self.apply_frequency_filter(observed_data, *f_min, *f_max)?;

            // Run FWI iteration for this frequency band
            self.iterate(&filtered_data)?;

            // Current model becomes starting point for next band
            // Model is already updated in self.velocity_model
        }
        Ok(())
    }
}

impl FullWaveformInversion {
    /// Compute the discrete L2 objective for a pair of data matrices.
    ///
    /// ## Theorem
    /// For `J = (dt / 2) ||d_syn - d_obs||²`, the objective is non-negative
    /// and vanishes if and only if the traces match pointwise.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_data_objective(
        &self,
        observed_data: &Array2<f64>,
        synthetic_data: &Array2<f64>,
    ) -> KwaversResult<f64> {
        l2_objective(self.config.dt, observed_data, synthetic_data)
    }

    /// Compute the discrete L2 adjoint source.
    ///
    /// This delegates to the shared acoustic adjoint-state core so the
    /// reconstruction and seismic FWI paths use the same residual convention.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_adjoint_source(
        &self,
        observed_data: &Array2<f64>,
        synthetic_data: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        l2_residual(observed_data, synthetic_data)
    }

    /// Compute objective function value (L2 misfit)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn compute_objective(
        &mut self,
        model: &Array3<f64>,
        observed_data: &Array2<f64>,
    ) -> KwaversResult<f64> {
        // Forward model with current velocity
        let modeled_data = self.wavefield_modeler.forward_model(model)?;

        // Compute L2 norm of data misfit
        self.compute_data_objective(observed_data, &modeled_data)
    }

    /// Apply Butterworth bandpass filter to data
    /// Apply 4th order Butterworth bandpass filter
    ///
    /// Implements digital Butterworth filter in frequency domain:
    /// H(f) = 1 / √(1 + (f/f_c)^(2n))
    /// where n is filter order (4 for this implementation)
    ///
    /// # References
    /// - Butterworth (1930): "On the Theory of Filter Amplifiers"
    /// - Oppenheim & Schafer (2009): "Discrete-Time Signal Processing"
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_frequency_filter(
        &self,
        data: &Array2<f64>,
        f_min: f64,
        f_max: f64,
    ) -> KwaversResult<Array2<f64>> {
        if !(f_min.is_finite() && f_max.is_finite()) || f_min < 0.0 || f_max <= f_min {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!("Invalid frequency band: [{f_min}, {f_max}]"),
                },
            ));
        }

        let shape = data.shape();
        let n_traces = shape[0];

        let sampling_frequency = self.config.base_config.sampling_frequency;
        if sampling_frequency <= 0.0 || !sampling_frequency.is_finite() {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "Sampling frequency must be positive and finite".to_owned(),
                },
            ));
        }

        let mut result = data.clone();

        for trace_idx in 0..n_traces {
            let trace = data.row(trace_idx).to_owned();
            let filtered_trace = crate::math::fft::apply_spectral_response_1d(
                &trace,
                sampling_frequency,
                move |_, freq, helper_nyquist| {
                    let hp_response = if f_min > 0.0 {
                        let ratio = freq / f_min;
                        let ratio4 = ratio.powi(4);
                        (ratio4 / (1.0 + ratio4)).sqrt()
                    } else {
                        1.0
                    };

                    let lp_response = if f_max < helper_nyquist {
                        let ratio = freq / f_max;
                        let ratio4 = ratio.powi(4);
                        (1.0 / (1.0 + ratio4)).sqrt()
                    } else {
                        1.0
                    };

                    hp_response * lp_response
                },
            );

            result.row_mut(trace_idx).assign(&filtered_trace);
        }

        log::debug!(
            "Applied 4th order Butterworth filter: {:.1} - {:.1} Hz",
            f_min,
            f_max
        );
        Ok(result)
    }

    /// Get configuration
    #[must_use]
    pub fn get_config(&self) -> &SeismicImagingConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SeismicImagingConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_l2_adjoint_source_sign_matches_misfit() {
        let fwi = FullWaveformInversion::new(SeismicImagingConfig::default());
        let observed = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 2.0]).expect("shape");
        let synthetic = Array2::from_shape_vec((1, 3), vec![3.0, 2.0, 1.0]).expect("shape");

        let residual = fwi
            .compute_adjoint_source(&observed, &synthetic)
            .expect("adjoint source must succeed");

        assert_eq!(
            residual,
            Array2::from_shape_vec((1, 3), vec![3.0, 1.0, -1.0]).expect("shape")
        );
    }

    #[test]
    fn test_objective_scales_with_dt() {
        let mut fwi = FullWaveformInversion::new(SeismicImagingConfig::default());
        fwi.config.dt = 0.5;

        let observed = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("shape");
        let synthetic = Array2::from_shape_vec((1, 2), vec![3.0, 5.0]).expect("shape");

        let objective = fwi
            .compute_data_objective(&observed, &synthetic)
            .expect("objective must succeed");

        // residual = [2, 4], 0.5 * dt * sum(residual^2) = 0.25 * 20 = 5
        assert!((objective - 5.0).abs() < f64::EPSILON);
    }
}
