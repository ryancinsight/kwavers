//! Passive subharmonic Green operator and Tikhonov inverse for cavitation density.
//!
//! The operator maps a cavitation source density (voxel vector) to subharmonic
//! pressure observations at passive receivers via a spherical-wave Green's
//! function attenuated by the path-integrated tissue power-law absorption.
//! The inverse solves the nonnegative Tikhonov problem by projected gradient
//! descent with a fixed step bounded by the Frobenius norm of the operator.

use ndarray::Array3;
use rayon::prelude::*;

use crate::math::numerics::operators::interpolation::trilinear_index_space;

use super::super::types::{
    grid_point_m, Nonlinear3dAperture, Nonlinear3dConfig, Nonlinear3dVolume,
};
use super::helpers::grid_index;

pub(super) struct PassiveOperator {
    pub(super) values: Vec<f64>,
    pub(super) cols: usize,
}

impl PassiveOperator {
    pub(super) fn new(
        volume: &Nonlinear3dVolume,
        aperture: &Nonlinear3dAperture,
        active: &[usize],
        config: &Nonlinear3dConfig,
    ) -> Self {
        let n = volume.body_mask.dim().0;
        let rows = aperture.receivers.len();
        let cols = active.len();
        // The passive subharmonic receiver path observes cavitation activity at
        // the subharmonic of the therapy drive: f_s = f0 / 2. Both the angular
        // wavenumber and the tissue power-law absorption are evaluated at this
        // subharmonic frequency.
        let subharmonic_hz = 0.5 * config.frequency_hz;
        let subharmonic_mhz = subharmonic_hz * 1.0e-6;
        let c_ref_m_s = 1540.0_f64;
        let k_subharmonic = 2.0 * std::f64::consts::PI * subharmonic_hz / c_ref_m_s;
        let spacing_m = volume.spacing_m;
        let min_distance_m = 0.5 * spacing_m;
        // Path-integrated tissue power-law attenuation. The cavitation source
        // sits inside the body and the receiver sits on the body surface;
        // for transcranial helmets the line crosses skull, whose attenuation
        // is ~26× soft tissue (Hamilton & Blackstock 1998 Table 4.1, Connor
        // & Hynynen 2002). Treeby & Cox 2010 / Szabo 1995: real tissue has a
        // per-class power-law exponent `y`, with `α(f) = α(1MHz) · f_MHz^y`.
        // Soft tissue has `y ≈ 1.05` (near-linear); cortical skull bone has
        // `y ≈ 2.0` (classical Stokes-Kirchhoff viscous limit). At the
        // subharmonic `f_s < 1 MHz`, the `y = 2` skull behavior produces 3×
        // less attenuation than a naive `y = 1` extrapolation would predict.
        //
        // Approach: trace the source→receiver line through the heterogeneous
        // `(α₁_MHz, y)` fields, sample with trilinear interpolation, and
        // evaluate `α_s(position) = α₁_MHz(position) · f_s_MHz^y(position)`
        // per sample. Integrate by trapezoidal rule and use as the exponent
        // in the spherical-wave Green's kernel
        //   G_s(r, s) = exp(-∫ α_s(t) dt) · cos(k_s · r) / (4π · r).
        let attenuation_field = &volume.attenuation_np_per_m_mhz;
        let power_law_y_field = &volume.attenuation_power_law_y;
        let mut values = vec![0.0_f64; rows.saturating_mul(cols)];
        // Row-parallel dense Green's-function fill: each row depends only on
        // the row's receiver position and reads `active` immutably.
        values
            .par_chunks_mut(cols)
            .zip(aperture.receivers.par_iter())
            .for_each(|(row_slice, receiver)| {
                let rp = grid_point_m(*receiver, n, spacing_m);
                let receiver_idx_f = [receiver.x as f64, receiver.y as f64, receiver.z as f64];
                for (col, cell) in active.iter().enumerate() {
                    let idx = grid_index(*cell, n);
                    let vp = grid_point_m(idx, n, spacing_m);
                    let r = (rp.x_m - vp.x_m)
                        .hypot(rp.y_m - vp.y_m)
                        .hypot(rp.z_m - vp.z_m)
                        .max(min_distance_m);
                    let source_idx_f = [idx.x as f64, idx.y as f64, idx.z as f64];
                    let path_alpha = integrate_power_law_attenuation_along_ray(
                        attenuation_field,
                        power_law_y_field,
                        source_idx_f,
                        receiver_idx_f,
                        spacing_m,
                        subharmonic_mhz,
                    );
                    row_slice[col] = (-path_alpha).exp() * (k_subharmonic * r).cos()
                        / (4.0 * std::f64::consts::PI * r);
                }
            });
        Self { values, cols }
    }

    pub(super) fn apply(&self, model: &[f64]) -> Vec<f64> {
        let cols = self.cols;
        self.values
            .par_chunks(cols)
            .map(|row| row.iter().zip(model.iter()).map(|(a, x)| a * x).sum())
            .collect()
    }

    pub(super) fn normal_gradient(&self, residual: &[f64], model: &[f64], lambda: f64) -> Vec<f64> {
        let cols = self.cols;
        // Column-parallel A^T r + lambda * model; each grad cell sums
        // contributions from every row of `values` (the dense Green's matrix).
        (0..cols)
            .into_par_iter()
            .map(|col| {
                let mut sum = lambda * model[col];
                for (row, residual_value) in residual.iter().enumerate() {
                    sum += self.values[row * cols + col] * residual_value;
                }
                sum
            })
            .collect()
    }

    pub(super) fn frobenius_norm_squared(&self) -> f64 {
        self.values.iter().map(|value| value * value).sum()
    }
}

pub(super) struct InverseResult {
    pub(super) model: Vec<f64>,
    pub(super) objective_history: Vec<f64>,
}

pub(super) fn solve_projected_tikhonov(
    operator: &PassiveOperator,
    data: &[f64],
    config: &Nonlinear3dConfig,
) -> InverseResult {
    let lambda = config.cavitation_regularization;
    let step = 1.0 / (operator.frobenius_norm_squared() + lambda).max(1.0e-18);
    let mut model = vec![0.0; operator.cols];
    let mut objective_history = Vec::with_capacity(config.cavitation_iterations);
    for _ in 0..config.cavitation_iterations {
        let prediction = operator.apply(&model);
        let residual = prediction
            .iter()
            .zip(data.iter())
            .map(|(p, d)| p - d)
            .collect::<Vec<_>>();
        objective_history.push(
            0.5 * residual.iter().map(|value| value * value).sum::<f64>()
                + 0.5 * lambda * model.iter().map(|value| value * value).sum::<f64>(),
        );
        let grad = operator.normal_gradient(&residual, &model, lambda);
        for (value, g) in model.iter_mut().zip(grad.iter()) {
            *value = (*value - step * g).max(0.0);
        }
    }
    InverseResult {
        model,
        objective_history,
    }
}

/// Integrate `α(s, f_s) = α₁_MHz(s) · f_s_MHz^y(s)` along the source-to-
/// receiver ray, returning the path-integrated Np (dimensionless exponent
/// for the `exp(-·)` Green's-function kernel).
///
/// # Algorithm
///
/// 1. Compute the source and receiver positions in voxel coordinates.
/// 2. Determine the number of samples `M` from the line length in voxels,
///    using one sample per grid spacing (with a minimum of 1).
/// 3. At each sample, trilinearly interpolate both `α₁_MHz` and the power-
///    law exponent `y`.
/// 4. Evaluate the frequency-scaled `α_s = α₁_MHz · f_s_MHz^y` per sample.
/// 5. Apply the trapezoidal rule with physical step length
///    `step_m = path_length_m / (M - 1)`.
///
/// # Mathematical contract
///
/// For a continuous power-law attenuation field along a ray of length `L`:
/// `∫₀^L α(s) ds = ∫₀^L α₁_MHz(s) · f_s_MHz^y(s) ds`. The trapezoidal
/// sampling error is `O((L/M)^2)`. Using `M ≈ |source - receiver|_voxels`
/// gives one sample per voxel — sufficient for the leading-order path
/// integral because both fields are piecewise nearly constant inside each
/// tissue class.
fn integrate_power_law_attenuation_along_ray(
    attenuation_field: &Array3<f64>,
    power_law_y_field: &Array3<f64>,
    source_idx: [f64; 3],
    receiver_idx: [f64; 3],
    spacing_m: f64,
    subharmonic_mhz: f64,
) -> f64 {
    let dx = receiver_idx[0] - source_idx[0];
    let dy = receiver_idx[1] - source_idx[1];
    let dz = receiver_idx[2] - source_idx[2];
    let voxel_length = (dx * dx + dy * dy + dz * dz).sqrt();
    if voxel_length <= f64::EPSILON {
        let (ix, iy, iz) = (
            source_idx[0].round() as usize,
            source_idx[1].round() as usize,
            source_idx[2].round() as usize,
        );
        let dim = attenuation_field.dim();
        let ix = ix.min(dim.0 - 1);
        let iy = iy.min(dim.1 - 1);
        let iz = iz.min(dim.2 - 1);
        let alpha_1mhz = attenuation_field[[ix, iy, iz]];
        let y = power_law_y_field[[ix, iy, iz]];
        return alpha_1mhz * subharmonic_mhz.powf(y) * spacing_m;
    }
    let samples = voxel_length.ceil() as usize + 1;
    let path_length_m = voxel_length * spacing_m;
    let step_m = path_length_m / (samples - 1).max(1) as f64;
    let mut integral = 0.0;
    for sample in 0..samples {
        let t = sample as f64 / (samples - 1).max(1) as f64;
        let px = source_idx[0] + t * dx;
        let py = source_idx[1] + t * dy;
        let pz = source_idx[2] + t * dz;
        let alpha_1mhz = trilinear_index_space(attenuation_field, px, py, pz);
        let y = trilinear_index_space(power_law_y_field, px, py, pz);
        let alpha_at_fs = alpha_1mhz * subharmonic_mhz.powf(y);
        // Trapezoidal weight: endpoints get 0.5, interior gets 1.0.
        let weight = if sample == 0 || sample + 1 == samples {
            0.5
        } else {
            1.0
        };
        integral += alpha_at_fs * weight;
    }
    integral * step_m
}
