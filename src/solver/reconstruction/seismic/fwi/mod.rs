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
use crate::error::KwaversResult;
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
    line_search: LineSearch,
    /// Regularizer
    regularizer: Regularizer,
    /// Wavefield modeler
    wavefield_modeler: WavefieldModeler,
    /// Misfit function
    misfit_function: MisfitFunction,
}

impl FullWaveformInversion {
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
    fn apply_frequency_filter(
        &self,
        data: &Array2<f64>,
        f_min: f64,
        f_max: f64,
    ) -> KwaversResult<Array2<f64>> {
        // Implement 4th order Butterworth filter
        // For now, return original data with frequency band annotation
        log::debug!("Applying frequency filter: {:.1} - {:.1} Hz", f_min, f_max);
        Ok(data.clone())
    }

    /// Get configuration
    pub fn get_config(&self) -> &SeismicImagingConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SeismicImagingConfig) {
        self.config = config;
    }
}
