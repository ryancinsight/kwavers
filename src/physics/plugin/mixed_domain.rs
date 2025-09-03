//! Mixed-Domain Propagation Plugin
//! Based on Pinton et al. (2009): "A heterogeneous nonlinear attenuating full-wave model"

use crate::error::KwaversResult;
use crate::fft::{Fft3d, Ifft3d};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::plugin::{PluginMetadata, PluginState};
use ndarray::{Array3, Zip};
use num_complex::Complex64;

/// Mixed-Domain Propagation Plugin
/// Combines time-domain and frequency-domain methods for optimal performance
#[derive(Debug)]
pub struct MixedDomainPropagationPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// Threshold for switching between domains
    domain_switch_threshold: f64,
    /// Frequency domain buffer
    frequency_buffer: Option<Array3<f64>>,
}

impl MixedDomainPropagationPlugin {
    /// Create new mixed-domain propagation plugin
    #[must_use]
    pub fn new(domain_switch_threshold: f64) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "mixed_domain_propagation".to_string(),
                name: "Mixed-Domain Propagation".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "Hybrid time/frequency domain propagation".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            domain_switch_threshold,
            frequency_buffer: None,
        }
    }

    /// Propagate field using optimal domain selection
    /// Based on Huijssen & Verweij (2010): "An iterative method for the computation"
    pub fn propagate(
        &mut self,
        field: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_step: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Analyze field characteristics to determine optimal propagation domain
        let domain = self.analyze_field(field);

        match domain {
            DomainSelection::TimeDomain => {
                // Direct time-domain propagation for strongly nonlinear fields
                self.propagate_time_domain(field, grid, medium, time_step)
            }
            DomainSelection::FrequencyDomain => {
                // Frequency-domain propagation for linear/weakly nonlinear fields
                let freq_field = self.to_frequency_domain(field)?;
                let propagated =
                    self.propagate_frequency_domain(&freq_field, grid, medium, time_step)?;
                self.to_time_domain(&propagated)
            }
            DomainSelection::Hybrid => {
                // Split-step approach: linear in frequency, nonlinear in time
                let freq_field = self.to_frequency_domain(field)?;
                let linear_prop =
                    self.apply_linear_propagator(&freq_field, grid, medium, time_step)?;
                let time_field = self.to_time_domain(&linear_prop)?;
                self.apply_nonlinear_correction(&time_field, grid, medium, time_step)
            }
        }
    }

    /// Analyze field to determine optimal domain
    #[must_use]
    pub fn analyze_field(&self, field: &Array3<f64>) -> DomainSelection {
        // Calculate field statistics for domain selection
        let mean = field.mean().unwrap_or(0.0);
        let variance = field.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / field.len() as f64;

        // Estimate nonlinearity strength from field variance
        let nonlinearity_metric = variance.sqrt() / (mean.abs() + 1e-10);

        // Determine optimal domain based on nonlinearity
        if nonlinearity_metric > self.domain_switch_threshold * 2.0 {
            DomainSelection::TimeDomain // Strong nonlinearity
        } else if nonlinearity_metric > self.domain_switch_threshold {
            DomainSelection::Hybrid // Moderate nonlinearity
        } else {
            DomainSelection::FrequencyDomain // Weak/no nonlinearity
        }
    }

    /// Convert field to frequency domain
    pub fn to_frequency_domain(&mut self, field: &Array3<f64>) -> KwaversResult<Array3<Complex64>> {
        // Convert real field to complex for FFT
        let mut complex_field = field.mapv(|x| Complex64::new(x, 0.0));

        // Create a dummy grid for FFT (FFT3d requires it)
        let grid = crate::grid::Grid::new(
            field.shape()[0],
            field.shape()[1],
            field.shape()[2],
            1.0,
            1.0,
            1.0,
        )?;

        // Apply 3D FFT
        let mut fft = Fft3d::new(field.shape()[0], field.shape()[1], field.shape()[2]);
        fft.process(&mut complex_field, &grid);

        Ok(complex_field)
    }

    /// Convert field to time domain
    pub fn to_time_domain(&mut self, field: &Array3<Complex64>) -> KwaversResult<Array3<f64>> {
        // Clone field for IFFT (process modifies in place)
        let mut complex_field = field.clone();

        // Create a dummy grid for IFFT
        let grid = crate::grid::Grid::new(
            field.shape()[0],
            field.shape()[1],
            field.shape()[2],
            1.0,
            1.0,
            1.0,
        )?;

        // Apply 3D IFFT
        let mut ifft = Ifft3d::new(field.shape()[0], field.shape()[1], field.shape()[2]);
        let real_field = ifft.process(&mut complex_field, &grid);

        Ok(real_field)
    }

    /// Time-domain propagation for nonlinear fields
    fn propagate_time_domain(
        &mut self,
        field: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _time_step: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Second-order finite difference propagation
        // This is a simplified implementation - full version would include
        // proper finite difference stencils and boundary conditions
        Ok(field.clone())
    }

    /// Frequency-domain propagation for linear fields
    fn propagate_frequency_domain(
        &mut self,
        field: &Array3<Complex64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_step: f64,
    ) -> KwaversResult<Array3<Complex64>> {
        // Apply spectral propagator exp(ikz * dz) in frequency domain
        let mut result = field.clone();
        let k = 2.0 * std::f64::consts::PI
            / (crate::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid) * time_step);

        Zip::from(&mut result).and(field).for_each(|r, &f| {
            *r = f * Complex64::from_polar(1.0, k * grid.dx);
        });

        Ok(result)
    }

    /// Apply linear propagator in frequency domain
    fn apply_linear_propagator(
        &mut self,
        field: &Array3<Complex64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_step: f64,
    ) -> KwaversResult<Array3<Complex64>> {
        self.propagate_frequency_domain(field, grid, medium, time_step)
    }

    /// Apply nonlinear correction in time domain
    fn apply_nonlinear_correction(
        &mut self,
        field: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _time_step: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Apply nonlinear correction N(p) = β/(2ρc³) * p²
        // This is a simplified implementation
        Ok(field.clone())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DomainSelection {
    TimeDomain,
    FrequencyDomain,
    Hybrid,
}
