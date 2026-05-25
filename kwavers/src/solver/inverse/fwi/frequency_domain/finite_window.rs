//! Finite-window PSTD Born scattering for frequency-domain FWI diagnostics.
//!
//! This module owns the finite-time theorem that is not represented by the
//! stationary Helmholtz CBS algebra.  For reference slowness `s0` and
//! contrast `chi = (s^2 - s0^2) / s0^2`, the first-order scattered PSTD field
//! satisfies the same homogeneous leapfrog recurrence as the direct field with
//! source term `-chi * (p0[n+1] - 2 p0[n] + p0[n-1])`.

use super::cbs::{
    pstd_modal_theta_squared, pstd_source_kappa_symbol, GridSpec, PstdTemporalTransferConfig,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    sound_speed_to_slowness, MultiRowRingArray,
};
use crate::solver::inverse::linear_born_inversion::ElementPosition;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::TAU;

/// Acquisition parameters for finite-window PSTD Born prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PstdFiniteWindowBornConfig {
    /// Homogeneous reference sound speed [m/s].
    pub reference_sound_speed_m_s: f64,
    /// Isotropic voxel spacing [m].
    pub spacing_m: f64,
    /// PSTD leapfrog time step [s].
    pub time_step_s: f64,
    /// Scalar pressure-source amplitude [Pa].
    pub source_amplitude_pa: f64,
    /// Number of drive cycles simulated for each frequency.
    pub cycles_per_frequency: usize,
    /// Number of trailing cycles used for the complex frequency bin.
    pub frequency_bin_cycles: usize,
}

/// Simulate finite-window first-order PSTD Born receiver rows.
///
/// # Errors
/// Returns an error when the grid, acquisition parameters, ring geometry, or
/// sound-speed values violate the finite-window PSTD contract.
pub fn simulate_pstd_finite_window_born_observation(
    sound_speed_m_s: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: PstdFiniteWindowBornConfig,
    transmissions: usize,
) -> KwaversResult<Array2<Complex64>> {
    let slowness = sound_speed_to_slowness(sound_speed_m_s)?;
    validate_inputs(&slowness, array, frequency_hz, config, transmissions)?;
    let grid = GridSpec::new(slowness.dim(), config.spacing_m)?;
    let bin_config = config.temporal_transfer().bin_config(
        frequency_hz,
        config.time_step_s,
        config.spacing_m,
        config.reference_sound_speed_m_s,
    )?;
    let contrast = normalized_slowness_squared_contrast(&slowness, config)?;
    let symbols = ModalSymbols::new(grid, config);
    let receiver_indices = receiver_indices_on_grid(grid, array)?;
    let mut output = Array2::zeros((transmissions, array.element_count()));

    for transmit in 0..transmissions {
        let source_hat = source_spectrum_on_grid(
            grid,
            &array.cylindrical_source(transmit),
            &symbols.source_kappa,
        )?;
        let row = simulate_transmit(
            grid,
            &source_hat,
            &contrast,
            &symbols.theta_squared,
            &receiver_indices,
            bin_config,
        );
        for (receiver_index, pressure) in row.into_iter().enumerate() {
            output[[transmit, receiver_index]] = pressure;
        }
    }

    Ok(output)
}

impl PstdFiniteWindowBornConfig {
    fn temporal_transfer(self) -> PstdTemporalTransferConfig {
        PstdTemporalTransferConfig {
            source_amplitude_pa: self.source_amplitude_pa,
            cycles_per_frequency: self.cycles_per_frequency,
            frequency_bin_cycles: self.frequency_bin_cycles,
        }
    }
}

struct ModalSymbols {
    theta_squared: Array3<f64>,
    source_kappa: Array3<f64>,
}

impl ModalSymbols {
    fn new(grid: GridSpec, config: PstdFiniteWindowBornConfig) -> Self {
        let (nx, ny, nz) = grid.dimensions;
        let theta_squared = Array3::from_shape_fn(grid.dimensions, |(ix, iy, iz)| {
            let k = modal_wavenumber(grid, ix, iy, iz);
            pstd_modal_theta_squared(k, config.time_step_s, config.reference_sound_speed_m_s)
        });
        let source_kappa = Array3::from_shape_fn((nx, ny, nz), |(ix, iy, iz)| {
            let k = modal_wavenumber(grid, ix, iy, iz);
            pstd_source_kappa_symbol(k, config.time_step_s, config.reference_sound_speed_m_s)
        });
        Self {
            theta_squared,
            source_kappa,
        }
    }
}

fn simulate_transmit(
    grid: GridSpec,
    source_hat: &Array3<Complex64>,
    contrast: &Array3<f64>,
    theta_squared: &Array3<f64>,
    receiver_indices: &[(usize, usize, usize)],
    bin_config: super::cbs::PstdTemporalBinConfig,
) -> Vec<Complex64> {
    let mut direct_prev_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut direct_curr_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut direct_next_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut scatter_prev_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut scatter_curr_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut scatter_next_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut direct_prev = Array3::<Complex64>::zeros(grid.dimensions);
    let mut direct_curr = Array3::<Complex64>::zeros(grid.dimensions);
    let mut direct_next = Array3::<Complex64>::zeros(grid.dimensions);
    let mut scatter_next = Array3::<Complex64>::zeros(grid.dimensions);
    let mut scatter_source = Array3::<Complex64>::zeros(grid.dimensions);
    let mut scatter_source_hat = Array3::<Complex64>::zeros(grid.dimensions);
    let mut receivers = vec![Complex64::new(0.0, 0.0); receiver_indices.len()];
    let angular_step = TAU * bin_config.frequency_hz * bin_config.time_step_s;
    let mut previous_signal = 0.0;

    for step in 0..bin_config.total_steps {
        let signal = (angular_step * step as f64).sin();
        let source_scale = bin_config.source_gain * (signal - previous_signal);
        for (((next, &current), &previous), (&theta, &source)) in direct_next_hat
            .iter_mut()
            .zip(direct_curr_hat.iter())
            .zip(direct_prev_hat.iter())
            .zip(theta_squared.iter().zip(source_hat.iter()))
        {
            *next = current * (2.0 - theta) - previous + source * source_scale;
        }

        direct_next.assign(&direct_next_hat);
        ifft_3d_complex_inplace(&mut direct_next);
        for (((dst, &next), &current), (&previous, &chi)) in scatter_source
            .iter_mut()
            .zip(direct_next.iter())
            .zip(direct_curr.iter())
            .zip(direct_prev.iter().zip(contrast.iter()))
        {
            *dst = -(next - current * 2.0 + previous) * chi;
        }

        fft_3d_complex_into(&scatter_source, &mut scatter_source_hat);
        for (((next, &current), &previous), (&theta, &source)) in scatter_next_hat
            .iter_mut()
            .zip(scatter_curr_hat.iter())
            .zip(scatter_prev_hat.iter())
            .zip(theta_squared.iter().zip(scatter_source_hat.iter()))
        {
            *next = current * (2.0 - theta) - previous + source;
        }

        scatter_next.assign(&scatter_next_hat);
        ifft_3d_complex_inplace(&mut scatter_next);
        if step >= bin_config.bin_start_step {
            let phase = -angular_step * step as f64;
            let demodulation = Complex64::new(phase.cos(), phase.sin());
            for (dst, &(ix, iy, iz)) in receivers.iter_mut().zip(receiver_indices.iter()) {
                *dst += (direct_next[[ix, iy, iz]] + scatter_next[[ix, iy, iz]]) * demodulation;
            }
        }

        direct_prev_hat.assign(&direct_curr_hat);
        direct_curr_hat.assign(&direct_next_hat);
        direct_prev.assign(&direct_curr);
        direct_curr.assign(&direct_next);
        scatter_prev_hat.assign(&scatter_curr_hat);
        scatter_curr_hat.assign(&scatter_next_hat);
        previous_signal = signal;
    }

    let scale = 2.0 / (bin_config.total_steps - bin_config.bin_start_step) as f64;
    for pressure in &mut receivers {
        *pressure *= scale;
    }
    receivers
}

fn normalized_slowness_squared_contrast(
    slowness_s_per_m: &Array3<f64>,
    config: PstdFiniteWindowBornConfig,
) -> KwaversResult<Array3<f64>> {
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let reference_squared = reference_slowness * reference_slowness;
    let mut contrast = Array3::<f64>::zeros(slowness_s_per_m.dim());
    for (dst, &slowness) in contrast.iter_mut().zip(slowness_s_per_m.iter()) {
        if !slowness.is_finite() || slowness <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "finite-window PSTD slowness must be positive and finite, got {slowness}"
            )));
        }
        *dst = (slowness * slowness - reference_squared) / reference_squared;
    }
    Ok(contrast)
}

fn source_spectrum_on_grid(
    grid: GridSpec,
    sources: &[ElementPosition],
    source_kappa: &Array3<f64>,
) -> KwaversResult<Array3<Complex64>> {
    let mut mask = Array3::<Complex64>::zeros(grid.dimensions);
    for &source in sources {
        let index = exact_grid_index(grid, source, "source")?;
        mask[index] += Complex64::new(1.0, 0.0);
    }
    let mut spectrum = Array3::<Complex64>::zeros(grid.dimensions);
    fft_3d_complex_into(&mask, &mut spectrum);
    for (value, &kappa) in spectrum.iter_mut().zip(source_kappa.iter()) {
        *value *= kappa;
    }
    Ok(spectrum)
}

fn receiver_indices_on_grid(
    grid: GridSpec,
    array: &MultiRowRingArray,
) -> KwaversResult<Vec<(usize, usize, usize)>> {
    array
        .elements()
        .iter()
        .map(|&receiver| exact_grid_index(grid, receiver, "receiver"))
        .collect()
}

fn validate_inputs(
    slowness_s_per_m: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    config: PstdFiniteWindowBornConfig,
    transmissions: usize,
) -> KwaversResult<()> {
    GridSpec::new(slowness_s_per_m.dim(), config.spacing_m)?;
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "finite-window PSTD frequency must be positive and finite, got {frequency_hz}"
        )));
    }
    if !config.reference_sound_speed_m_s.is_finite() || config.reference_sound_speed_m_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "finite-window PSTD reference sound speed must be positive and finite, got {}",
            config.reference_sound_speed_m_s
        )));
    }
    if !config.time_step_s.is_finite() || config.time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "finite-window PSTD time step must be positive and finite, got {}",
            config.time_step_s
        )));
    }
    if transmissions == 0 || transmissions > array.circumferential_elements() {
        return Err(KwaversError::InvalidInput(format!(
            "finite-window PSTD transmissions must be in 1..={}, got {transmissions}",
            array.circumferential_elements()
        )));
    }
    config.temporal_transfer().validate()
}

fn exact_grid_index(
    grid: GridSpec,
    point: ElementPosition,
    role: &str,
) -> KwaversResult<(usize, usize, usize)> {
    Ok((
        exact_axis_index(grid.dimensions.0, grid.spacing_m, point.x_m, role)?,
        exact_axis_index(grid.dimensions.1, grid.spacing_m, point.y_m, role)?,
        exact_axis_index(grid.dimensions.2, grid.spacing_m, point.z_m, role)?,
    ))
}

fn exact_axis_index(n: usize, spacing_m: f64, value_m: f64, role: &str) -> KwaversResult<usize> {
    let raw = value_m / spacing_m + 0.5 * (n - 1) as f64;
    let rounded = raw.round();
    if (raw - rounded).abs() > 1.0e-9 || rounded < 0.0 || rounded > (n - 1) as f64 {
        return Err(KwaversError::InvalidInput(format!(
            "finite-window PSTD {role} coordinate {value_m} is not on the centered grid axis"
        )));
    }
    Ok(rounded as usize)
}

fn modal_wavenumber(grid: GridSpec, ix: usize, iy: usize, iz: usize) -> f64 {
    let kx = angular_mode(ix, grid.dimensions.0, grid.spacing_m);
    let ky = angular_mode(iy, grid.dimensions.1, grid.spacing_m);
    let kz = angular_mode(iz, grid.dimensions.2, grid.spacing_m);
    kx.mul_add(kx, ky.mul_add(ky, kz * kz)).sqrt()
}

fn angular_mode(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    TAU * signed_index / (count as f64 * spacing_m)
}
