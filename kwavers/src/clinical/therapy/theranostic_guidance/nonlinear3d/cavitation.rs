//! Rayleigh-Plesset cavitation forward map and passive inverse.
//!
//! Algorithm: each active voxel receives the local Westervelt peak pressure as
//! the acoustic forcing amplitude in the Rayleigh-Plesset ODE. The source
//! density is the maximum period-doubled radius response, which is then mapped
//! to passive receivers by a subharmonic Green operator. The inverse solves a
//! nonnegative Tikhonov problem by projected gradient descent with step bounded
//! by the Frobenius norm of the discrete operator.

use ndarray::Array3;
use rayon::prelude::*;

use crate::physics::acoustics::bubble_dynamics::{
    BubbleParameters, BubbleState, RayleighPlessetSolver,
};

use super::metrics::metrics_from_score;
use super::types::{
    flat_index, grid_point_m, GridIndex, Nonlinear3dAperture, Nonlinear3dConfig, Nonlinear3dVolume,
    VolumeReconstructionMetrics,
};

#[derive(Clone, Debug)]
pub(crate) struct CavitationResult {
    pub source_density: Array3<f64>,
    pub reconstructed_density: Array3<f64>,
    pub objective_history: Vec<f64>,
    pub metrics: VolumeReconstructionMetrics,
}

pub(crate) fn run_cavitation_inverse(
    volume: &Nonlinear3dVolume,
    aperture: &Nonlinear3dAperture,
    peak_pressure: &Array3<f64>,
    config: &Nonlinear3dConfig,
) -> CavitationResult {
    let n = volume.body_mask.dim().0;
    let body = volume.body_mask.iter().copied().collect::<Vec<_>>();
    let source = cavitation_source(volume, peak_pressure, config);
    let source_vec = source.iter().copied().collect::<Vec<_>>();
    let active = active_indices(&body);
    let operator = PassiveOperator::new(volume, aperture, &active, config);
    let data = operator.apply(&source_vec);
    let inverse = solve_projected_tikhonov(&operator, &data, config);
    let mut reconstructed = vec![0.0; n * n * n];
    for (col, cell) in active.iter().enumerate() {
        reconstructed[*cell] = inverse.model[col];
    }
    let source_mask = positive_mask(&source_vec, &body);
    let score = normalize(&reconstructed, &body);
    CavitationResult {
        source_density: source,
        reconstructed_density: unflatten(&reconstructed, n),
        objective_history: inverse.objective_history,
        metrics: metrics_from_score(&score, &source_mask, &body),
    }
}

fn cavitation_source(
    volume: &Nonlinear3dVolume,
    peak_pressure: &Array3<f64>,
    config: &Nonlinear3dConfig,
) -> Array3<f64> {
    let max_pressure = peak_pressure.iter().copied().fold(0.0, f64::max).max(1.0);
    Array3::from_shape_fn(peak_pressure.dim(), |idx| {
        if !volume.body_mask[idx] {
            return 0.0;
        }
        let pressure = peak_pressure[idx];
        let response = rayleigh_plesset_subharmonic_response(pressure, config);
        response * (pressure / max_pressure).clamp(0.0, 1.0)
    })
}

fn rayleigh_plesset_subharmonic_response(pressure_pa: f64, config: &Nonlinear3dConfig) -> f64 {
    if pressure_pa <= 0.0 {
        return 0.0;
    }
    let mut params = BubbleParameters {
        r0: config.bubble_radius_m,
        driving_frequency: config.frequency_hz,
        driving_amplitude: pressure_pa,
        use_thermal_effects: false,
        use_mass_transfer: false,
        use_compressibility: false,
        ..BubbleParameters::default()
    };
    params.initial_gas_pressure = params.p0;
    let solver = RayleighPlessetSolver::new(params.clone());
    let mut state = BubbleState::at_equilibrium(&params);
    let steps_per_period = config.bubble_time_steps_per_period;
    let total_steps = (config.cycles.ceil() as usize + 2) * steps_per_period;
    let dt = 1.0 / (config.frequency_hz * steps_per_period as f64);
    let mut radii = Vec::with_capacity(total_steps + 1);
    radii.push(state.radius);
    let mut max_subharmonic: f64 = 0.0;
    let mut max_compression: f64 = 1.0;
    for step in 0..total_steps {
        state = rk4_step(
            &solver,
            &state,
            pressure_pa,
            step as f64 * dt,
            dt,
            params.r0,
        );
        state.update_compression(params.r0);
        max_compression = max_compression.max(state.compression_ratio);
        radii.push(state.radius);
        if radii.len() > steps_per_period {
            let previous_period = radii[radii.len() - 1 - steps_per_period];
            max_subharmonic =
                max_subharmonic.max((state.radius - previous_period).abs() / params.r0);
        }
        if state.radius <= 0.05 * params.r0 {
            max_subharmonic = max_subharmonic.max(max_compression);
            break;
        }
    }
    max_subharmonic.max((max_compression - 1.0).max(0.0))
}

fn rk4_step(
    solver: &RayleighPlessetSolver,
    state: &BubbleState,
    pressure_pa: f64,
    t: f64,
    dt: f64,
    r0: f64,
) -> BubbleState {
    let k1 = derivative(solver, state, pressure_pa, t);
    let s2 = shifted_state(state, k1, 0.5 * dt, r0);
    let k2 = derivative(solver, &s2, pressure_pa, t + 0.5 * dt);
    let s3 = shifted_state(state, k2, 0.5 * dt, r0);
    let k3 = derivative(solver, &s3, pressure_pa, t + 0.5 * dt);
    let s4 = shifted_state(state, k3, dt, r0);
    let k4 = derivative(solver, &s4, pressure_pa, t + dt);
    let mut out = state.clone();
    out.radius = (state.radius + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0).max(0.05 * r0);
    out.wall_velocity = state.wall_velocity + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
    out.wall_acceleration = solver.calculate_acceleration(&out, pressure_pa, t + dt);
    out
}

fn derivative(
    solver: &RayleighPlessetSolver,
    state: &BubbleState,
    pressure_pa: f64,
    t: f64,
) -> (f64, f64) {
    (
        state.wall_velocity,
        solver.calculate_acceleration(state, pressure_pa, t),
    )
}

fn shifted_state(state: &BubbleState, derivative: (f64, f64), dt: f64, r0: f64) -> BubbleState {
    let mut shifted = state.clone();
    shifted.radius = (state.radius + dt * derivative.0).max(0.05 * r0);
    shifted.wall_velocity = state.wall_velocity + dt * derivative.1;
    shifted
}

struct PassiveOperator {
    values: Vec<f64>,
    cols: usize,
}

impl PassiveOperator {
    fn new(
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

    fn apply(&self, model: &[f64]) -> Vec<f64> {
        let cols = self.cols;
        self.values
            .par_chunks(cols)
            .map(|row| row.iter().zip(model.iter()).map(|(a, x)| a * x).sum())
            .collect()
    }

    fn normal_gradient(&self, residual: &[f64], model: &[f64], lambda: f64) -> Vec<f64> {
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

    fn frobenius_norm_squared(&self) -> f64 {
        self.values.iter().map(|value| value * value).sum()
    }
}

struct InverseResult {
    model: Vec<f64>,
    objective_history: Vec<f64>,
}

fn solve_projected_tikhonov(
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

fn active_indices(body: &[bool]) -> Vec<usize> {
    body.iter()
        .enumerate()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect()
}

fn positive_mask(values: &[f64], body: &[bool]) -> Vec<bool> {
    let peak = values
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .fold(0.0, f64::max);
    values
        .iter()
        .zip(body.iter())
        .map(|(value, active)| *active && *value >= 0.35 * peak)
        .collect()
}

fn normalize(values: &[f64], body: &[bool]) -> Vec<f64> {
    let peak = values
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .fold(0.0, f64::max)
        .max(1.0e-12);
    values
        .iter()
        .zip(body.iter())
        .map(|(value, active)| active.then_some(*value / peak).unwrap_or(0.0))
        .collect()
}

fn grid_index(flat: usize, n: usize) -> GridIndex {
    GridIndex {
        x: flat / (n * n),
        y: (flat / n) % n,
        z: flat % n,
    }
}

fn unflatten(values: &[f64], n: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, n, n), |(x, y, z)| {
        values[flat_index(GridIndex { x, y, z }, n)]
    })
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
        let alpha_1mhz = trilinear_attenuation(attenuation_field, px, py, pz);
        let y = trilinear_attenuation(power_law_y_field, px, py, pz);
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

fn trilinear_attenuation(field: &Array3<f64>, x: f64, y: f64, z: f64) -> f64 {
    let dim = field.dim();
    let x0 = x.floor().clamp(0.0, (dim.0 - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (dim.1 - 1) as f64) as usize;
    let z0 = z.floor().clamp(0.0, (dim.2 - 1) as f64) as usize;
    let x1 = (x0 + 1).min(dim.0 - 1);
    let y1 = (y0 + 1).min(dim.1 - 1);
    let z1 = (z0 + 1).min(dim.2 - 1);
    let tx = (x - x0 as f64).clamp(0.0, 1.0);
    let ty = (y - y0 as f64).clamp(0.0, 1.0);
    let tz = (z - z0 as f64).clamp(0.0, 1.0);
    let c000 = field[[x0, y0, z0]];
    let c100 = field[[x1, y0, z0]];
    let c010 = field[[x0, y1, z0]];
    let c110 = field[[x1, y1, z0]];
    let c001 = field[[x0, y0, z1]];
    let c101 = field[[x1, y0, z1]];
    let c011 = field[[x0, y1, z1]];
    let c111 = field[[x1, y1, z1]];
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    c0 * (1.0 - tz) + c1 * tz
}
