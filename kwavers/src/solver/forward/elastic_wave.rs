//! Elastic wave solver implementation
//!
//! This module contains the core solving routines for elastic wave propagation,
//! separated from the main module to maintain GRASP principles.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields,
    parameters::{StressUpdateParams, VelocityUpdateParams},
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
    ElasticWave,
};
use crate::physics::acoustics::traits::AcousticWaveModel;
use log::{debug, info};
use ndarray::{Array3, Array4, Axis, Zip};
use num_complex::Complex;
use std::time::Instant;

impl ElasticWave {
    /// Update stress fields in-place using spectral (Fourier-differentiation) method.
    ///
    /// # Mathematical specification
    /// Hooke's law for an isotropic fluid (μ = 0) in the spectral domain, with zero
    /// initial stress (stress state is not persisted between steps):
    /// ```text
    ///   σ̃ₓₓ = dt · (λ · div_v + 2μ · ikₓ ṽₓ)
    ///   σ̃ᵧᵧ = dt · (λ · div_v + 2μ · ikᵧ ṽᵧ)
    ///   σ̃ᵤᵤ = dt · (λ · div_v + 2μ · ikᵤ ṽᵤ)
    ///   σ̃ₓᵧ = dt · μ · (ikᵧ ṽₓ + ikₓ ṽᵧ)
    ///   σ̃ₓᵤ = dt · μ · (ikᵤ ṽₓ + ikₓ ṽᵤ)
    ///   σ̃ᵧᵤ = dt · μ · (ikᵤ ṽᵧ + ikᵧ ṽᵤ)
    /// ```
    /// Parallelised with `Zip::indexed` + `par_for_each`; no heap allocation per call.
    pub(crate) fn update_stress_in_place(
        params: &StressUpdateParams,
        out: &mut SpectralStressFields,
    ) {
        // kx/ky/kz have shape (n,1,1) — contiguous 1-D wavenumber arrays.
        let kx_s = params.kx.as_slice().expect("kx must be contiguous");
        let ky_s = params.ky.as_slice().expect("ky must be contiguous");
        let kz_s = params.kz.as_slice().expect("kz must be contiguous");
        let c_dt = Complex::new(params.dt, 0.0);

        // ndarray 0.16 Zip supports at most 6 total producers (index counts as 1).
        // Split into two passes: normal stresses (txx/tyy/tzz) and shear stresses
        // (txy/txz/tyz). Both passes read the same inputs; LLVM fuses common
        // subexpressions. For acoustic fluid (μ=0), the shear pass constant-folds
        // to zero and is eliminated by the optimiser.

        // Pass A — normal stresses (index + 3 outputs = 4 ≤ 6)
        Zip::indexed(out.txx.view_mut())
            .and(out.tyy.view_mut())
            .and(out.tzz.view_mut())
            .par_for_each(|(i, j, k), o_txx, o_tyy, o_tzz| {
                let c_kx = Complex::new(0.0, kx_s[i]);
                let c_ky = Complex::new(0.0, ky_s[j]);
                let c_kz = Complex::new(0.0, kz_s[k]);

                let vx = params.vx_fft[[i, j, k]];
                let vy = params.vy_fft[[i, j, k]];
                let vz = params.vz_fft[[i, j, k]];

                let lambda = params.lame_lambda[[i, j, k]];
                let mu = params.lame_mu[[i, j, k]];
                let div_v = c_kx * vx + c_ky * vy + c_kz * vz;

                *o_txx = c_dt * (lambda * div_v + 2.0 * mu * (c_kx * vx));
                *o_tyy = c_dt * (lambda * div_v + 2.0 * mu * (c_ky * vy));
                *o_tzz = c_dt * (lambda * div_v + 2.0 * mu * (c_kz * vz));
            });

        // Pass B — shear stresses (index + 3 outputs = 4 ≤ 6)
        Zip::indexed(out.txy.view_mut())
            .and(out.txz.view_mut())
            .and(out.tyz.view_mut())
            .par_for_each(|(i, j, k), o_txy, o_txz, o_tyz| {
                let c_kx = Complex::new(0.0, kx_s[i]);
                let c_ky = Complex::new(0.0, ky_s[j]);
                let c_kz = Complex::new(0.0, kz_s[k]);

                let vx = params.vx_fft[[i, j, k]];
                let vy = params.vy_fft[[i, j, k]];
                let vz = params.vz_fft[[i, j, k]];
                let mu = params.lame_mu[[i, j, k]];

                *o_txy = c_dt * mu * (c_ky * vx + c_kx * vy);
                *o_txz = c_dt * mu * (c_kz * vx + c_kx * vz);
                *o_tyz = c_dt * mu * (c_kz * vy + c_ky * vz);
            });
    }

    /// Update velocity fields in-place using spectral method.
    ///
    /// # Mathematical specification
    /// Newton's second law in the spectral domain:
    /// ```text
    ///   ṽₓ(t+dt) = ṽₓ(t) + dt/ρ · (ikₓ σ̃ₓₓ + ikᵧ σ̃ₓᵧ + ikᵤ σ̃ₓᵤ)
    ///   ṽᵧ(t+dt) = ṽᵧ(t) + dt/ρ · (ikₓ σ̃ₓᵧ + ikᵧ σ̃ᵧᵧ + ikᵤ σ̃ᵧᵤ)
    ///   ṽᵤ(t+dt) = ṽᵤ(t) + dt/ρ · (ikₓ σ̃ₓᵤ + ikᵧ σ̃ᵧᵤ + ikᵤ σ̃ᵤᵤ)
    /// ```
    /// Parallelised with `Zip::indexed` + `par_for_each`; no heap allocation per call.
    pub(crate) fn update_velocity_in_place(
        params: &VelocityUpdateParams,
        out: &mut SpectralVelocityFields,
    ) {
        let kx_s = params.kx.as_slice().expect("kx must be contiguous");
        let ky_s = params.ky.as_slice().expect("ky must be contiguous");
        let kz_s = params.kz.as_slice().expect("kz must be contiguous");

        Zip::indexed(out.vx.view_mut())
            .and(out.vy.view_mut())
            .and(out.vz.view_mut())
            .par_for_each(|(i, j, k), o_vx, o_vy, o_vz| {
                let rho = params.density[[i, j, k]];
                if rho <= 0.0 {
                    // Preserve current velocity at invalid density points.
                    *o_vx = params.vx_fft[[i, j, k]];
                    *o_vy = params.vy_fft[[i, j, k]];
                    *o_vz = params.vz_fft[[i, j, k]];
                    return;
                }

                let c_kx = Complex::new(0.0, kx_s[i]);
                let c_ky = Complex::new(0.0, ky_s[j]);
                let c_kz = Complex::new(0.0, kz_s[k]);
                let c_dt_rho = Complex::new(params.dt / rho, 0.0);

                // Stress divergence (force per unit volume)
                let dtxx_dx = c_kx * params.txx_fft[[i, j, k]];
                let dtxy_dy = c_ky * params.txy_fft[[i, j, k]];
                let dtxz_dz = c_kz * params.txz_fft[[i, j, k]];

                let dtxy_dx = c_kx * params.txy_fft[[i, j, k]];
                let dtyy_dy = c_ky * params.tyy_fft[[i, j, k]];
                let dtyz_dz = c_kz * params.tyz_fft[[i, j, k]];

                let dtxz_dx = c_kx * params.txz_fft[[i, j, k]];
                let dtyz_dy = c_ky * params.tyz_fft[[i, j, k]];
                let dtzz_dz = c_kz * params.tzz_fft[[i, j, k]];

                *o_vx = params.vx_fft[[i, j, k]] + c_dt_rho * (dtxx_dx + dtxy_dy + dtxz_dz);
                *o_vy = params.vy_fft[[i, j, k]] + c_dt_rho * (dtxy_dx + dtyy_dy + dtyz_dz);
                *o_vz = params.vz_fft[[i, j, k]] + c_dt_rho * (dtxz_dx + dtyz_dy + dtzz_dz);
            });
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

        let (nx, ny, nz) = grid.dimensions();

        use crate::domain::field::indices::{PRESSURE_IDX, VX_IDX, VY_IDX, VZ_IDX};

        // ── 1. Extract velocity fields ────────────────────────────────────────
        let vx = fields.index_axis(Axis(0), VX_IDX).to_owned();
        let vy = fields.index_axis(Axis(0), VY_IDX).to_owned();
        let vz = fields.index_axis(Axis(0), VZ_IDX).to_owned();

        // ── 2. Pre-compute Lamé parameters into scratch (zero alloc) ─────────
        //
        // Acoustic fluid: λ = ρc², μ = 0.
        let density = medium.density_array();
        let sound_speed = medium.sound_speed_array();

        Zip::from(self.lambda_scratch.view_mut())
            .and(density.view())
            .and(sound_speed.view())
            .par_for_each(|lam, &rho, &c| *lam = rho * c * c);
        self.mu_scratch.fill(0.0);

        // ── 3. Forward-FFT velocity fields ────────────────────────────────────
        let fft_start = Instant::now();
        let spectral_velocity = SpectralVelocityFields::from_real(&VelocityFields { vx, vy, vz });
        self.metrics.add_fft_time(fft_start.elapsed());

        // ── 4. Stress update (in-place, no allocation) ───────────────────────
        //
        // kx/ky/kz are cloned here (shape (n,1,1), O(n) cost) to allow the
        // disjoint mutable borrow of self.stress_scratch below.
        let kx = self.kx.clone();
        let ky = self.ky.clone();
        let kz = self.kz.clone();

        let stress_params = StressUpdateParams {
            vx_fft: &spectral_velocity.vx,
            vy_fft: &spectral_velocity.vy,
            vz_fft: &spectral_velocity.vz,
            kx: &kx,
            ky: &ky,
            kz: &kz,
            lame_lambda: &self.lambda_scratch,
            lame_mu: &self.mu_scratch,
            density,
            dt,
        };
        Self::update_stress_in_place(&stress_params, &mut self.stress_scratch);

        // ── 5. Velocity update (in-place, no allocation) ─────────────────────
        //
        // density view is re-acquired after stress_params is dropped.
        let density2 = medium.density_array();
        let velocity_params = VelocityUpdateParams {
            vx_fft: &spectral_velocity.vx,
            vy_fft: &spectral_velocity.vy,
            vz_fft: &spectral_velocity.vz,
            txx_fft: &self.stress_scratch.txx,
            tyy_fft: &self.stress_scratch.tyy,
            tzz_fft: &self.stress_scratch.tzz,
            txy_fft: &self.stress_scratch.txy,
            txz_fft: &self.stress_scratch.txz,
            tyz_fft: &self.stress_scratch.tyz,
            kx: &kx,
            ky: &ky,
            kz: &kz,
            density: density2,
            dt,
        };
        Self::update_velocity_in_place(&velocity_params, &mut self.velocity_scratch);

        // ── 6. Inverse-FFT velocity scratch back to real space ───────────────
        let ifft_start = Instant::now();
        let real_velocity = self.velocity_scratch.to_real();
        self.metrics.add_ifft_time(ifft_start.elapsed());

        // ── 7. Source injection ───────────────────────────────────────────────
        let source_mask = source.create_mask(grid);
        let source_amplitude = source.amplitude(t);
        let mut pressure = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        pressure.zip_mut_with(&source_mask, |p, &mask| {
            *p += mask * source_amplitude * dt;
        });

        // ── 8. Write updated velocity fields back ─────────────────────────────
        fields
            .index_axis_mut(Axis(0), VX_IDX)
            .assign(&real_velocity.vx);
        fields
            .index_axis_mut(Axis(0), VY_IDX)
            .assign(&real_velocity.vy);
        fields
            .index_axis_mut(Axis(0), VZ_IDX)
            .assign(&real_velocity.vz);

        // Suppress unused-variable warnings for nx/ny/nz (used implicitly via grid)
        let _ = (nx, ny, nz);

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
        // Elastic waves do not have the same nonlinearity as acoustic waves.
        // Reserved for future viscoelastic scaling if required.
        debug!("Nonlinearity scaling not applicable to elastic waves");
    }
}
