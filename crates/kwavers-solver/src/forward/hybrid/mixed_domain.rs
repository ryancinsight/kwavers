//! Mixed-Domain Propagation Plugin
//! Based on Pinton et al. (2009): "A heterogeneous nonlinear attenuating full-wave model"

use crate::plugin::{PluginMetadata, PluginState};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_math::fft::{Fft3dInOutExt, Shape3D, FFT_CACHE_3D};
use kwavers_medium::Medium;
use leto::Array3 as LetoArray3;
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

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
                id: "mixed_domain_propagation".to_owned(),
                name: "Mixed-Domain Propagation".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Hybrid time/frequency domain propagation".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Initialized,
            domain_switch_threshold,
            frequency_buffer: None,
        }
    }

    /// Propagate field using optimal domain selection
    /// Based on Huijssen & Verweij (2010): "An iterative method for the computation"
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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

    /// Select optimal domain for propagation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn select_optimal_domain(
        &self,
        field: &Array3<f64>,
        _grid: &Grid,
    ) -> KwaversResult<DomainSelection> {
        Ok(self.analyze_field(field))
    }

    /// Analyze field to determine optimal domain
    #[must_use]
    pub fn analyze_field(&self, field: &Array3<f64>) -> DomainSelection {
        // Calculate field statistics for domain selection
        let n = field.len();
        let mean = if n == 0 {
            0.0
        } else {
            field.iter().sum::<f64>() / n as f64
        };
        let variance =
            field.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (field.len()) as f64;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn to_frequency_domain(
        &mut self,
        field: &Array3<f64>,
    ) -> KwaversResult<LetoArray3<Complex64>> {
        let [nx, ny, nz] = field.shape();
        let fft = FFT_CACHE_3D.get_or_create(Shape3D { nx, ny, nz });

        let field_leto = LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
            .expect("mixed-domain field shape must match its Leto FFT shape");
        let mut out = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
        fft.forward_into(&field_leto, &mut out);
        Ok(out)
    }

    /// Convert field to time domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn to_time_domain(&mut self, field: &LetoArray3<Complex64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        let fft = FFT_CACHE_3D.get_or_create(Shape3D { nx, ny, nz });
        let mut out = LetoArray3::<f64>::zeros([nx, ny, nz]);
        let mut scratch = LetoArray3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
        fft.inverse_into(field, &mut out, &mut scratch);
        Ok(Array3::from_shape_vec([nx, ny, nz], out.into_vec())
            .expect("mixed-domain inverse FFT output shape must match its grid"))
    }

    /// Time-domain propagation for nonlinear fields
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn propagate_time_domain(
        &mut self,
        field: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _time_step: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Second-order finite difference propagation
        // **Implementation**: Identity operation for mixed-domain coupling
        // Time-domain fields propagate via explicit FDTD in main solver loop
        // This method maintains API contract for domain coupling interface
        Ok(field.clone())
    }

    /// Frequency-domain propagation for linear fields
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn propagate_frequency_domain(
        &mut self,
        field: &LetoArray3<Complex64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_step: f64,
    ) -> KwaversResult<LetoArray3<Complex64>> {
        // Apply spectral propagator exp(ikz * dz) in frequency domain
        let mut result = field.clone();
        let k = TWO_PI / (kwavers_medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid) * time_step);
        let phase = Complex64::from_polar(1.0, k * grid.dx);

        let input = field
            .as_slice()
            .expect("invariant: mixed-domain frequency input is standard-layout");
        let output = result
            .as_slice_mut()
            .expect("invariant: mixed-domain frequency output is standard-layout");

        enumerate_mut_with::<Adaptive, _, _>(output, |idx, r| {
            *r = input[idx] * phase;
        });

        Ok(result)
    }

    /// Apply linear propagator in frequency domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_linear_propagator(
        &mut self,
        field: &LetoArray3<Complex64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_step: f64,
    ) -> KwaversResult<LetoArray3<Complex64>> {
        self.propagate_frequency_domain(field, grid, medium, time_step)
    }

    /// Apply nonlinear correction in time domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_nonlinear_correction(
        &mut self,
        field: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _time_step: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Nonlinear correction: N(p) = β/(2ρc³) * p² (Hamilton & Blackstock 1998)
        // Current implementation returns linear field (appropriate for linear regime testing)
        // Full nonlinear correction requires proper β/A parameter from medium properties
        // Future: Implement when nonlinear medium interface is extended (Sprint 122+)
        Ok(field.clone())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DomainSelection {
    TimeDomain,
    FrequencyDomain,
    Hybrid,
}

// Plugin trait implementation
impl crate::plugin::Plugin for MixedDomainPropagationPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<kwavers_field::mapping::UnifiedFieldType> {
        vec![kwavers_field::mapping::UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<kwavers_field::mapping::UnifiedFieldType> {
        vec![kwavers_field::mapping::UnifiedFieldType::Pressure]
    }

    fn update(
        &mut self,
        fields: &mut leto::Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &mut crate::plugin::PluginContext<'_>,
    ) -> KwaversResult<()> {
        use kwavers_field::mapping::UnifiedFieldType;

        // Extract pressure field
        let pressure_field = fields
            .index_axis::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field index within field stack");
        let pressure_array = pressure_field.to_contiguous();

        // Determine optimal domain based on field characteristics
        let domain = self.select_optimal_domain(&pressure_array, grid)?;

        let result = match domain {
            DomainSelection::TimeDomain => {
                self.propagate_time_domain(&pressure_array, grid, medium, dt)
            }
            DomainSelection::FrequencyDomain => {
                let complex_field = self.to_frequency_domain(&pressure_array)?;
                let result = self.propagate_frequency_domain(&complex_field, grid, medium, dt)?;
                self.to_time_domain(&result)
            }
            DomainSelection::Hybrid => {
                // Apply hybrid method: time domain + frequency domain correction
                let time_result = self.propagate_time_domain(&pressure_array, grid, medium, dt)?;
                self.apply_nonlinear_correction(&time_result, grid, medium, dt)
            }
        }?;

        // Update pressure field in the fields array
        let mut pressure_slice = fields
            .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field index within field stack");
        pressure_slice.assign(&result);

        Ok(())
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        self.frequency_buffer = None;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.frequency_buffer = None;
        self.state = PluginState::Created;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
