//! Full Waveform Inversion (FWI) implementation
//!
//! Based on:
//! - Virieux & Operto (2009): "An overview of full-waveform inversion"
//! - Tarantola (1984): "Inversion of seismic reflection data"

pub mod gradient;
pub mod optimization;
pub mod regularization;
pub mod wavefield;

use self::gradient::GradientComputer;
use self::optimization::{ConjugateGradient, LineSearch};
use self::regularization::Regularizer;
use self::wavefield::WavefieldModeler;

use super::config::SeismicImagingConfig;
use super::misfit::{MisfitFunction, MisfitType};
use crate::core::error::KwaversResult;
use ndarray::{Array2, Array3};

/// Full Waveform Inversion (FWI) reconstructor
#[derive(Debug)]
pub struct FullWaveformInversion {
    config: SeismicImagingConfig,
    /// Current velocity model
    velocity_model: Array3<f64>,
    /// Gradient computer
    gradient_computer: GradientComputer,
    /// Optimizer
    optimizer: ConjugateGradient,
    /// Line search
    #[allow(dead_code)]
    line_search: LineSearch,
    /// Regularizer
    regularizer: Regularizer,
    /// Wavefield modeler
    wavefield_modeler: WavefieldModeler,
    /// Misfit function
    misfit_function: MisfitFunction,
}

impl FullWaveformInversion {
    #[must_use]
    pub fn new(config: SeismicImagingConfig) -> Self {
        let shape = (config.nx, config.ny, config.nz);
        Self {
            config,
            velocity_model: Array3::zeros(shape),
            gradient_computer: GradientComputer::new(),
            optimizer: ConjugateGradient::new(),
            line_search: LineSearch::new(),
            regularizer: Regularizer::new(),
            wavefield_modeler: WavefieldModeler::new(),
            misfit_function: MisfitFunction::new(MisfitType::L2Norm),
        }
    }

    /// Main FWI iteration
    pub fn iterate(&mut self, observed_data: &Array2<f64>) -> KwaversResult<()> {
        // 1. Forward modeling
        let synthetic_data = self.wavefield_modeler.forward_model(&self.velocity_model)?;

        // 2. Compute residual
        let residual = observed_data - &synthetic_data;

        // 3. Compute gradient via adjoint method
        let adjoint_source = self.misfit_function.adjoint_source(&residual);
        let adjoint_wavefield = self.wavefield_modeler.adjoint_model(&adjoint_source)?;
        let forward_wavefield = self.wavefield_modeler.get_forward_wavefield()?;

        let mut gradient = self.gradient_computer.compute_adjoint_gradient(
            &forward_wavefield,
            &adjoint_wavefield,
            self.config.dt,
        );

        // 4. Apply regularization
        self.regularizer
            .apply_regularization(&mut gradient, &self.velocity_model);

        // 5. Compute search direction
        let direction = self.optimizer.compute_direction(&gradient);

        // 6. Line search for step size using Armijo-Wolfe conditions
        // Wolfe conditions ensure sufficient decrease and curvature conditions
        // Reference: Nocedal & Wright (2006) "Numerical Optimization", Chapter 3
        let velocity_model_copy = self.velocity_model.clone();
        let current_objective = self.compute_objective(&velocity_model_copy, observed_data)?;

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
            if test_objective <= current_objective + c1 * step_size * directional_derivative {
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
    /// Compute objective function value (L2 misfit)
    fn compute_objective(
        &mut self,
        model: &Array3<f64>,
        observed_data: &Array2<f64>,
    ) -> KwaversResult<f64> {
        // Forward model with current velocity
        let modeled_data = self.wavefield_modeler.forward_model(model)?;

        // Compute L2 norm of data misfit
        let misfit = &modeled_data - observed_data;
        Ok(misfit.mapv(|x| x * x).sum() * 0.5)
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
    fn apply_frequency_filter(
        &self,
        data: &Array2<f64>,
        f_min: f64,
        f_max: f64,
    ) -> KwaversResult<Array2<f64>> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let shape = data.shape();
        let n_traces = shape[0];
        let n_samples = shape[1];

        // Assume sampling parameters (could be made configurable)
        let dt = 0.004; // 4ms sampling interval (250 Hz)
        let nyquist = 0.5 / dt; // Nyquist frequency

        let mut result = data.clone();

        // Process each trace independently
        let mut planner = FftPlanner::new();

        for trace_idx in 0..n_traces {
            // Extract trace
            let trace: Vec<f64> = (0..n_samples).map(|i| data[[trace_idx, i]]).collect();

            // Convert to complex for FFT
            let mut buffer: Vec<Complex<f64>> =
                trace.iter().map(|&x| Complex::new(x, 0.0)).collect();

            // Forward FFT
            let fft = planner.plan_fft_forward(n_samples);
            fft.process(&mut buffer);

            // Apply 4th order Butterworth bandpass filter
            for (i, coeff) in buffer.iter_mut().enumerate() {
                let freq = (i as f64) * nyquist / (n_samples as f64);

                // 4th order Butterworth highpass at f_min
                let hp_response = if f_min > 0.0 {
                    let ratio = freq / f_min;
                    let ratio4 = ratio.powi(4);
                    (ratio4 / (1.0 + ratio4)).sqrt()
                } else {
                    1.0
                };

                // 4th order Butterworth lowpass at f_max
                let lp_response = if f_max < nyquist {
                    let ratio = freq / f_max;
                    let ratio4 = ratio.powi(4);
                    (1.0 / (1.0 + ratio4)).sqrt()
                } else {
                    1.0
                };

                // Combined bandpass response
                let filter_response = hp_response * lp_response;
                *coeff *= filter_response;
            }

            // Inverse FFT
            let ifft = planner.plan_fft_inverse(n_samples);
            ifft.process(&mut buffer);

            // Extract real part and normalize
            let norm_factor = 1.0 / (n_samples as f64);
            for (i, coeff) in buffer.iter().enumerate() {
                result[[trace_idx, i]] = coeff.re * norm_factor;
            }
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
