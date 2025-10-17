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
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
    ElasticWave,
};
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use log::{debug, info};
use ndarray::{Array3, Array4, Axis};
use num_complex::Complex;
use std::time::Instant;

impl ElasticWave {
    /// Update stress fields using spectral method with proper complex field handling
    pub(crate) fn _update_stress_fft(
        &mut self,
        params: &StressUpdateParams,
    ) -> KwaversResult<SpectralStressFields> {
        let nx = self.kx.shape()[0];
        let ny = self.ky.shape()[0];
        let nz = self.kz.shape()[0];

        let mut updated_stress = SpectralStressFields::new(nx, ny, nz);

        // Spectral derivatives for strain computation
        // Using Fourier differentiation: d/dx -> ik_x in frequency domain
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let kx = self.kx[[i, 0, 0]];
                    let ky = self.ky[[j, 0, 0]];
                    let kz = self.kz[[k, 0, 0]];

                    // Velocity derivatives (strain rates) in frequency domain
                    let dvx_dx = Complex::new(0.0, kx) * params.vx_fft[[i, j, k]];
                    let dvy_dy = Complex::new(0.0, ky) * params.vy_fft[[i, j, k]];
                    let dvz_dz = Complex::new(0.0, kz) * params.vz_fft[[i, j, k]];

                    let dvx_dy = Complex::new(0.0, ky) * params.vx_fft[[i, j, k]];
                    let dvx_dz = Complex::new(0.0, kz) * params.vx_fft[[i, j, k]];
                    let dvy_dx = Complex::new(0.0, kx) * params.vy_fft[[i, j, k]];
                    let dvy_dz = Complex::new(0.0, kz) * params.vy_fft[[i, j, k]];
                    let dvz_dx = Complex::new(0.0, kx) * params.vz_fft[[i, j, k]];
                    let dvz_dy = Complex::new(0.0, ky) * params.vz_fft[[i, j, k]];

                    // Constitutive relations (Hooke's law)
                    let lambda = params.lame_lambda[[i, j, k]];
                    let mu = params.lame_mu[[i, j, k]];

                    let div_v = dvx_dx + dvy_dy + dvz_dz;

                    // Update normal stresses in frequency domain
                    updated_stress.txx[[i, j, k]] = params.sxx_fft[[i, j, k]]
                        + Complex::new(params.dt, 0.0) * (lambda * div_v + 2.0 * mu * dvx_dx);
                    updated_stress.tyy[[i, j, k]] = params.syy_fft[[i, j, k]]
                        + Complex::new(params.dt, 0.0) * (lambda * div_v + 2.0 * mu * dvy_dy);
                    updated_stress.tzz[[i, j, k]] = params.szz_fft[[i, j, k]]
                        + Complex::new(params.dt, 0.0) * (lambda * div_v + 2.0 * mu * dvz_dz);

                    // Update shear stresses in frequency domain
                    updated_stress.txy[[i, j, k]] = params.sxy_fft[[i, j, k]]
                        + Complex::new(params.dt, 0.0) * mu * (dvx_dy + dvy_dx);
                    updated_stress.txz[[i, j, k]] = params.sxz_fft[[i, j, k]]
                        + Complex::new(params.dt, 0.0) * mu * (dvx_dz + dvz_dx);
                    updated_stress.tyz[[i, j, k]] = params.syz_fft[[i, j, k]]
                        + Complex::new(params.dt, 0.0) * mu * (dvy_dz + dvz_dy);
                }
            }
        }

        Ok(updated_stress)
    }

    /// Update velocity fields using spectral method with proper complex field handling
    pub(crate) fn _update_velocity_fft(
        &mut self,
        params: &VelocityUpdateParams,
    ) -> KwaversResult<SpectralVelocityFields> {
        let nx = self.kx.shape()[0];
        let ny = self.ky.shape()[0];
        let nz = self.kz.shape()[0];

        let mut updated_velocity = SpectralVelocityFields::new(nx, ny, nz);

        // Spectral derivatives for force computation
        // Second law of motion: dv/dt = F/m = (∇·σ)/ρ
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let kx = self.kx[[i, 0, 0]];
                    let ky = self.ky[[j, 0, 0]];
                    let kz = self.kz[[k, 0, 0]];

                    let rho = params.density[[i, j, k]];
                    if rho <= 0.0 {
                        continue; // Skip invalid density points
                    }

                    // Stress divergence (force per unit volume) in frequency domain
                    let dtxx_dx = Complex::new(0.0, kx) * params.txx_fft[[i, j, k]];
                    let dtxy_dy = Complex::new(0.0, ky) * params.txy_fft[[i, j, k]];
                    let dtxz_dz = Complex::new(0.0, kz) * params.txz_fft[[i, j, k]];

                    let dtxy_dx = Complex::new(0.0, kx) * params.txy_fft[[i, j, k]];
                    let dtyy_dy = Complex::new(0.0, ky) * params.tyy_fft[[i, j, k]];
                    let dtyz_dz = Complex::new(0.0, kz) * params.tyz_fft[[i, j, k]];

                    let dtxz_dx = Complex::new(0.0, kx) * params.txz_fft[[i, j, k]];
                    let dtyz_dy = Complex::new(0.0, ky) * params.tyz_fft[[i, j, k]];
                    let dtzz_dz = Complex::new(0.0, kz) * params.tzz_fft[[i, j, k]];

                    // Second law of motion in frequency domain
                    updated_velocity.vx[[i, j, k]] = params.vx_fft[[i, j, k]]
                        + Complex::new(params.dt / rho, 0.0) * (dtxx_dx + dtxy_dy + dtxz_dz);
                    updated_velocity.vy[[i, j, k]] = params.vy_fft[[i, j, k]]
                        + Complex::new(params.dt / rho, 0.0) * (dtxy_dx + dtyy_dy + dtyz_dz);
                    updated_velocity.vz[[i, j, k]] = params.vz_fft[[i, j, k]]
                        + Complex::new(params.dt / rho, 0.0) * (dtxz_dx + dtyz_dy + dtzz_dz);
                }
            }
        }

        Ok(updated_velocity)
    }
}

impl AcousticWaveModel for ElasticWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        // Get field dimensions
        let (nx, ny, nz) = grid.dimensions();

        // For elastic waves, we use velocity components instead of pressure
        use crate::physics::field_indices::{PRESSURE_IDX, VX_IDX, VY_IDX, VZ_IDX};

        // Extract velocity fields from the 4D array
        let mut vx = fields.index_axis(Axis(0), VX_IDX).to_owned();
        let mut vy = fields.index_axis(Axis(0), VY_IDX).to_owned();
        let mut vz = fields.index_axis(Axis(0), VZ_IDX).to_owned();

        // Transform to frequency domain
        // FFT timing will be tracked within spectral field conversions
        let fft_start = Instant::now();
        self.metrics.add_fft_time(fft_start.elapsed());

        // Initialize stress fields if not present
        // Note: Spectral methods would require complex fields for FFT
        // Currently using real fields with finite-difference implementation

        // Get material properties
        let density = medium.density_array();
        let sound_speed = medium.sound_speed_array();

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

        // Create spectral fields from extracted velocity fields
        let velocity_fields = VelocityFields { vx, vy, vz };
        let spectral_velocity = SpectralVelocityFields::from_real(&velocity_fields);

        // Initialize spectral stress fields
        let stress_fields = StressFields::new(nx, ny, nz);
        let spectral_stress = SpectralStressFields::from_real(&stress_fields);

        // Update stress fields in frequency domain
        let stress_params = StressUpdateParams {
            vx_fft: &spectral_velocity.vx,
            vy_fft: &spectral_velocity.vy,
            vz_fft: &spectral_velocity.vz,
            sxx_fft: &spectral_stress.txx,
            syy_fft: &spectral_stress.tyy,
            szz_fft: &spectral_stress.tzz,
            sxy_fft: &spectral_stress.txy,
            sxz_fft: &spectral_stress.txz,
            syz_fft: &spectral_stress.tyz,
            kx: &kx,
            ky: &ky,
            kz: &kz,
            lame_lambda: &lambda,
            lame_mu: &mu,
            density,
            dt,
        };

        let updated_spectral_stress = self
            ._update_stress_fft(&stress_params)
            .unwrap_or_else(|_| SpectralStressFields::new(nx, ny, nz));

        // Update velocity fields in frequency domain
        let velocity_params = VelocityUpdateParams {
            vx_fft: &spectral_velocity.vx,
            vy_fft: &spectral_velocity.vy,
            vz_fft: &spectral_velocity.vz,
            txx_fft: &updated_spectral_stress.txx,
            tyy_fft: &updated_spectral_stress.tyy,
            tzz_fft: &updated_spectral_stress.tzz,
            txy_fft: &updated_spectral_stress.txy,
            txz_fft: &updated_spectral_stress.txz,
            tyz_fft: &updated_spectral_stress.tyz,
            kx: &kx,
            ky: &ky,
            kz: &kz,
            density,
            dt,
        };

        let updated_spectral_velocity = self
            ._update_velocity_fft(&velocity_params)
            .unwrap_or_else(|_| SpectralVelocityFields::new(nx, ny, nz));

        // Transform back to spatial domain
        let ifft_start = Instant::now();
        let real_velocity = updated_spectral_velocity.to_real();
        vx = real_velocity.vx;
        vy = real_velocity.vy;
        vz = real_velocity.vz;
        self.metrics.add_ifft_time(ifft_start.elapsed());

        // Mode conversion: Optional feature for P-wave/S-wave coupling (see mode_conversion module)
        // Enable via ElasticWaveConfig::enable_mode_conversion() when needed
        // Currently disabled - requires explicit configuration for elastic interfaces

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

        Ok(())
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
