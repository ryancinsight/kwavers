//! Elastic wave solver implementation
//!
//! This module contains the core solving routines for elastic wave propagation,
//! separated from the main module to maintain GRASP principles.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::mechanics::elastic_wave::{
    fields::{StressFields, VelocityFields},
    parameters::{StressUpdateParams, VelocityUpdateParams},
    ElasticWave,
};
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::utils::fft_3d_array as fft_3d;
use log::{debug, info};
use ndarray::{Array3, Array4, Axis};
use std::time::Instant;

impl ElasticWave {
    /// Update stress fields using spectral method
    /// NOTE: This spectral implementation requires complex fields for FFT
    /// Currently disabled in favor of finite-difference implementation in plugin
    #[allow(dead_code)]
    pub(crate) fn _update_stress_fft(
        &mut self,
        _params: &StressUpdateParams,
    ) -> KwaversResult<StressFields> {
        // Spectral method implementation would go here
        // Currently using finite-difference implementation in elastic_wave_plugin
        // This avoids the complex/real type mismatch issue
        let nx = self.kx.shape()[0];
        let ny = self.ky.shape()[0];
        let nz = self.kz.shape()[0];

        Ok(StressFields::new(nx, ny, nz))
    }

    /// Update velocity fields using spectral method
    /// NOTE: This spectral implementation requires complex fields for FFT
    /// Currently disabled in favor of finite-difference implementation in plugin
    #[allow(dead_code)]
    pub(crate) fn _update_velocity_fft(
        &mut self,
        _params: &VelocityUpdateParams,
    ) -> KwaversResult<VelocityFields> {
        // Spectral method implementation would go here
        // Currently using finite-difference implementation in elastic_wave_plugin
        // This avoids the complex/real type mismatch issue
        let nx = self.kx.shape()[0];
        let ny = self.ky.shape()[0];
        let nz = self.kz.shape()[0];

        Ok(VelocityFields::new(nx, ny, nz))
    }
}

impl AcousticWaveModel for ElasticWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        let start = Instant::now();

        // Get field dimensions
        let (nx, ny, nz) = grid.dimensions();

        // For elastic waves, we use velocity components instead of pressure
        // Field indices: 0=pressure, 1=vx, 2=vy, 3=vz
        const PRESSURE_IDX: usize = 0;
        const VX_IDX: usize = 1;
        const VY_IDX: usize = 2;
        const VZ_IDX: usize = 3;

        // Extract velocity fields from the 4D array
        let vx = fields.index_axis(Axis(0), VX_IDX).to_owned();
        let vy = fields.index_axis(Axis(0), VY_IDX).to_owned();
        let vz = fields.index_axis(Axis(0), VZ_IDX).to_owned();

        // Transform to frequency domain
        let fft_start = Instant::now();
        let vx_fft = fft_3d(&vx);
        let vy_fft = fft_3d(&vy);
        let vz_fft = fft_3d(&vz);
        self.metrics.add_fft_time(fft_start.elapsed());

        // Initialize stress fields if not present
        // Note: Spectral methods would require complex fields for FFT
        // Currently using real fields with finite-difference implementation

        // Get material properties
        let density = medium
            .get_density_array(grid)
            .unwrap_or_else(|_| Array3::ones((nx, ny, nz)) * 1000.0);
        let sound_speed = medium
            .get_sound_speed_array(grid)
            .unwrap_or_else(|_| Array3::ones((nx, ny, nz)) * 1500.0);

        // Convert to elastic parameters
        let mut lambda = Array3::zeros((nx, ny, nz));
        let mut mu = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let rho = density[[i, j, k]];
                    let c = sound_speed[[i, j, k]];
                    // Assume fluid (mu = 0) for acoustic case
                    lambda[[i, j, k]] = rho * c * c;
                    mu[[i, j, k]] = 0.0;
                }
            }
        }

        // Clone wavenumber arrays to avoid borrow issues
        let kx = self.kx.clone();
        let ky = self.ky.clone();
        let kz = self.kz.clone();

        // Spectral stress update would go here
        // Currently using finite-difference implementation in elastic_wave_plugin
        let _ = (&vx_fft, &vy_fft, &vz_fft); // Suppress unused warnings
        let _ = (&kx, &ky, &kz, &lambda, &mu, &density);

        // Spectral velocity update would go here
        // Currently using finite-difference implementation in elastic_wave_plugin

        // Transform back to spatial domain would happen here
        // Currently using real-valued fields in finite-difference implementation
        let ifft_start = Instant::now();
        self.metrics.add_fft_time(ifft_start.elapsed());

        // Mode conversion would be applied here if enabled
        // Currently simplified for compilation

        // Add source contribution using mask-based approach
        let source_mask = source.create_mask(grid);
        let source_amplitude = source.amplitude(t);
        let mut pressure = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        pressure.zip_mut_with(&source_mask, |p, &mask| {
            *p += mask * source_amplitude * dt;
        });

        // Store updated velocity fields back into the 4D array
        fields.index_axis_mut(Axis(0), VX_IDX).assign(&vx);
        fields.index_axis_mut(Axis(0), VY_IDX).assign(&vy);
        fields.index_axis_mut(Axis(0), VZ_IDX).assign(&vz);

        // Update metrics
        self.metrics.increment_steps();
        self.metrics.add_update_time(start.elapsed());
    }

    fn report_performance(&self) {
        info!("ElasticWave Performance Metrics:");
        info!("  Total steps: {}", self.metrics.total_steps);
        info!("  Average FFT time: {:?}", self.metrics.avg_fft_time());
        info!(
            "  Average update time: {:?}",
            self.metrics.avg_update_time()
        );
        info!("  Max velocity: {:.3e}", self.metrics.max_velocity);
        info!("  Max stress: {:.3e}", self.metrics.max_stress);
    }

    fn set_nonlinearity_scaling(&mut self, _scaling: f64) {
        // Elastic waves don't have the same nonlinearity as acoustic waves
        // This could be used to scale viscoelastic effects if needed
        debug!("Nonlinearity scaling not applicable to elastic waves");
    }
}
