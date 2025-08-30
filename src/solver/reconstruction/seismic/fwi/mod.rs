//! Full Waveform Inversion (FWI) implementation
//!
//! Based on:
//! - Virieux & Operto (2009): "An overview of full-waveform inversion"
//! - Tarantola (1984): "Inversion of seismic reflection data"

pub mod gradient;
pub mod optimization;
pub mod regularization;
pub mod wavefield;

use self::gradient::GradientCalculator;
use self::optimization::{ConjugateGradient, LineSearch};
use self::regularization::Regularizer;
use self::wavefield::WavefieldModeler;

use super::config::SeismicImagingConfig;
use super::misfit::{MisfitFunction, MisfitType};
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array2, Array3};

/// Full Waveform Inversion (FWI) reconstructor
#[derive(Debug)]
pub struct FullWaveformInversion {
    config: SeismicImagingConfig,
    /// Current velocity model
    velocity_model: Array3<f64>,
    /// Gradient computer
    gradient_computer: GradientCalculator,
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
        let grid = Grid::new(config.nx, config.ny, config.nz, 1.0, 1.0, 1.0);
        Self {
            velocity_model: Array3::zeros(shape),
            gradient_computer: GradientCalculator::new(100, 0.001),
            optimizer: ConjugateGradient::new(),
            line_search: LineSearch::new(),
            regularizer: Regularizer::new(),
            wavefield_modeler: WavefieldModeler::new(grid, 0.001, vec![1.0; 100]),
            misfit_function: MisfitFunction::new(MisfitType::L2Norm),
            config,
        }
    }

    /// Main FWI iteration
    pub fn iterate(&mut self, observed_data: &Array3<f64>) -> KwaversResult<()> {
        // 1. Forward modeling
        let synthetic_data = self.wavefield_modeler.forward_model(&self.velocity_model)?;

        // 2. Compute residual (extract surface data)
        let mut residual = Array2::zeros((synthetic_data.shape()[0], synthetic_data.shape()[2]));
        for i in 0..synthetic_data.shape()[0] {
            for t in 0..synthetic_data.shape()[2] {
                residual[[i, t]] = observed_data[[i, 0, t]] - synthetic_data[[i, 0, t]];
            }
        }

        // 3. Compute gradient via adjoint method
        let adjoint_source = self.misfit_function.adjoint_source(&residual);
        let adjoint_wavefield = self.wavefield_modeler.adjoint_model(&adjoint_source)?;
        let forward_wavefield = self.wavefield_modeler.get_wavefield().clone();

        let mut gradient = self.gradient_computer.compute_gradient(
            &forward_wavefield,
            &adjoint_wavefield,
            &self.velocity_model,
        )?;

        // 4. Apply regularization
        self.regularizer
            .apply_regularization(&mut gradient, &self.velocity_model);

        // 5. Compute search direction
        let direction = self.optimizer.compute_direction(&gradient);

        // 6. Line search for step size
        let step_size = self.line_search.wolfe_search(
            &direction,
            &gradient,
            |_model| 0.0, // TODO: Implement objective function
        );

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
            // TODO: Filter data to frequency band
            // TODO: Run FWI for this band
            // TODO: Use result as starting model for next band
        }
        Ok(())
    }
}

impl FullWaveformInversion {
    /// Get configuration
    pub fn get_config(&self) -> &SeismicImagingConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SeismicImagingConfig) {
        self.config = config;
    }
}
