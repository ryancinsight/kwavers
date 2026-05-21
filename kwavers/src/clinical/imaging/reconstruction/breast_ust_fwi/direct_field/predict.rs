use super::grid::{
    grid_index_to_point, outgoing_green, point_to_grid_index, source_mask, validate_sampling,
    wavenumber_magnitude, GridShape,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::TAU;

pub(super) fn point_source_observation_cube(
    array: &MultiRowRingArray,
    frequencies_hz: &[f64],
    sound_speed_m_s: f64,
    spacing_m: f64,
) -> KwaversResult<Array3<Complex64>> {
    validate_sampling(sound_speed_m_s, spacing_m, 1.0)?;
    validate_array(array)?;
    validate_frequencies(frequencies_hz)?;
    let transmissions = array.circumferential_elements();
    let receivers = array.element_count();
    let min_distance = 0.5 * spacing_m;
    let mut output = Array3::<Complex64>::zeros((frequencies_hz.len(), transmissions, receivers));

    for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
        let wavenumber = TAU * frequency_hz / sound_speed_m_s;
        for transmit_index in 0..transmissions {
            for receiver_index in 0..receivers {
                let receiver = array.elements()[receiver_index];
                let mut value = Complex64::new(0.0, 0.0);
                for row in 0..array.rows() {
                    let source = array.elements()[row * transmissions + transmit_index];
                    value += outgoing_green(source, receiver, wavenumber, min_distance);
                }
                output[[frequency_index, transmit_index, receiver_index]] = value;
            }
        }
    }
    Ok(output)
}

pub(super) fn source_kappa_filtered_observation_cube(
    array: &MultiRowRingArray,
    frequencies_hz: &[f64],
    sound_speed_m_s: f64,
    spacing_m: f64,
    grid_dimensions: (usize, usize, usize),
    time_step_s: f64,
) -> KwaversResult<Array3<Complex64>> {
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)?;
    validate_array(array)?;
    validate_frequencies(frequencies_hz)?;
    let shape = GridShape::new(grid_dimensions)?;
    let transmissions = array.circumferential_elements();
    let receivers = array.element_count();
    let min_distance = 0.5 * spacing_m;
    let mut output = Array3::<Complex64>::zeros((frequencies_hz.len(), transmissions, receivers));

    for transmit_index in 0..transmissions {
        let source_indices = source_indices_for_transmit(array, transmit_index, shape, spacing_m)?;
        let weights = source_kappa_filtered_source_weights(
            shape,
            spacing_m,
            sound_speed_m_s,
            time_step_s,
            &source_indices,
        )?;
        for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
            let wavenumber = TAU * frequency_hz / sound_speed_m_s;
            for receiver_index in 0..receivers {
                let receiver = array.elements()[receiver_index];
                let mut value = Complex64::new(0.0, 0.0);
                for ix in 0..shape.nx {
                    for iy in 0..shape.ny {
                        for iz in 0..shape.nz {
                            let weight = weights[[ix, iy, iz]].re;
                            if weight != 0.0 {
                                let source = grid_index_to_point(shape, spacing_m, (ix, iy, iz));
                                value += weight
                                    * outgoing_green(source, receiver, wavenumber, min_distance);
                            }
                        }
                    }
                }
                output[[frequency_index, transmit_index, receiver_index]] = value;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pstd_periodic_observation_cube(
    array: &MultiRowRingArray,
    frequencies_hz: &[f64],
    sound_speed_m_s: f64,
    spacing_m: f64,
    grid_dimensions: (usize, usize, usize),
    time_step_s: f64,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
    source_amplitude_pa: f64,
) -> KwaversResult<Array3<Complex64>> {
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)?;
    validate_array(array)?;
    validate_frequencies(frequencies_hz)?;
    if !source_amplitude_pa.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "source_amplitude_pa must be finite, got {source_amplitude_pa}"
        )));
    }
    if time_steps_per_frequency.len() != frequencies_hz.len()
        || frequency_bin_start_steps_per_frequency.len() != frequencies_hz.len()
    {
        return Err(KwaversError::DimensionMismatch(
            "frequency metadata length must match frequencies_hz".into(),
        ));
    }
    let shape = GridShape::new(grid_dimensions)?;
    let receiver_indices = array
        .elements()
        .iter()
        .copied()
        .map(|point| point_to_grid_index(shape, spacing_m, point))
        .collect::<KwaversResult<Vec<_>>>()?;
    let transmissions = array.circumferential_elements();
    let receivers = array.element_count();
    let mut output = Array3::<Complex64>::zeros((frequencies_hz.len(), transmissions, receivers));

    for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
        let steps = time_steps_per_frequency[frequency_index];
        let bin_start = frequency_bin_start_steps_per_frequency[frequency_index];
        if bin_start >= steps {
            return Err(KwaversError::InvalidInput(format!(
                "frequency bin start {bin_start} must be smaller than total steps {steps}"
            )));
        }
        for transmit_index in 0..transmissions {
            let source_indices =
                source_indices_for_transmit(array, transmit_index, shape, spacing_m)?;
            let bin = pstd_periodic_frequency_bin(
                shape,
                spacing_m,
                sound_speed_m_s,
                time_step_s,
                frequency_hz,
                steps,
                bin_start,
                &source_indices,
                &receiver_indices,
                source_amplitude_pa,
            )?;
            for (receiver_index, value) in bin.into_iter().enumerate() {
                output[[frequency_index, transmit_index, receiver_index]] = value;
            }
        }
    }
    Ok(output)
}

pub(super) fn source_kappa_filtered_source_weights(
    shape: GridShape,
    spacing_m: f64,
    sound_speed_m_s: f64,
    time_step_s: f64,
    source_indices: &[(usize, usize, usize)],
) -> KwaversResult<Array3<Complex64>> {
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)?;
    let mask = source_mask(shape, source_indices)?;
    let mut spectrum = Array3::<Complex64>::zeros(shape.dimensions());
    fft_3d_complex_into(&mask, &mut spectrum);
    for ix in 0..shape.nx {
        for iy in 0..shape.ny {
            for iz in 0..shape.nz {
                let symbol =
                    source_kappa_symbol(shape, spacing_m, sound_speed_m_s, time_step_s, ix, iy, iz);
                spectrum[[ix, iy, iz]] *= Complex64::new(symbol, 0.0);
            }
        }
    }
    ifft_3d_complex_inplace(&mut spectrum);
    Ok(spectrum)
}

#[allow(clippy::too_many_arguments)]
fn pstd_periodic_frequency_bin(
    shape: GridShape,
    spacing_m: f64,
    sound_speed_m_s: f64,
    time_step_s: f64,
    frequency_hz: f64,
    steps: usize,
    bin_start: usize,
    source_indices: &[(usize, usize, usize)],
    receiver_indices: &[(usize, usize, usize)],
    source_amplitude_pa: f64,
) -> KwaversResult<Vec<Complex64>> {
    let mask = source_mask(shape, source_indices)?;
    let mut source_hat = Array3::<Complex64>::zeros(shape.dimensions());
    fft_3d_complex_into(&mask, &mut source_hat);
    let theta_squared = pstd_leapfrog_theta_squared(shape, spacing_m, sound_speed_m_s, time_step_s);
    for ix in 0..shape.nx {
        for iy in 0..shape.ny {
            for iz in 0..shape.nz {
                let symbol =
                    source_kappa_symbol(shape, spacing_m, sound_speed_m_s, time_step_s, ix, iy, iz);
                source_hat[[ix, iy, iz]] *= Complex64::new(symbol, 0.0);
            }
        }
    }

    let mut p_previous = Array3::<Complex64>::zeros(shape.dimensions());
    let mut p_current = Array3::<Complex64>::zeros(shape.dimensions());
    let mut p_next = Array3::<Complex64>::zeros(shape.dimensions());
    let mut real_space = Array3::<Complex64>::zeros(shape.dimensions());
    let mut bins = vec![Complex64::new(0.0, 0.0); receiver_indices.len()];
    let angular_step = TAU * frequency_hz * time_step_s;
    let gain = 2.0 * sound_speed_m_s * time_step_s * source_amplitude_pa / spacing_m;
    let mut previous_signal = 0.0;

    for step in 0..steps {
        let signal = (angular_step * step as f64).sin();
        let source_scale = gain * (signal - previous_signal);
        for ix in 0..shape.nx {
            for iy in 0..shape.ny {
                for iz in 0..shape.nz {
                    p_next[[ix, iy, iz]] = (2.0 - theta_squared[[ix, iy, iz]])
                        * p_current[[ix, iy, iz]]
                        - p_previous[[ix, iy, iz]]
                        + source_scale * source_hat[[ix, iy, iz]];
                }
            }
        }

        real_space.assign(&p_next);
        ifft_3d_complex_inplace(&mut real_space);
        if step >= bin_start {
            let phase = -TAU * frequency_hz * step as f64 * time_step_s;
            let weight = Complex64::new(phase.cos(), phase.sin());
            for (receiver, &(ix, iy, iz)) in receiver_indices.iter().enumerate() {
                bins[receiver] += real_space[[ix, iy, iz]].re * weight;
            }
        }

        std::mem::swap(&mut p_previous, &mut p_current);
        std::mem::swap(&mut p_current, &mut p_next);
        previous_signal = signal;
    }

    let scale = 2.0 / (steps - bin_start) as f64;
    for value in &mut bins {
        *value *= scale;
    }
    Ok(bins)
}

fn pstd_leapfrog_theta_squared(
    shape: GridShape,
    spacing_m: f64,
    sound_speed_m_s: f64,
    time_step_s: f64,
) -> Array3<f64> {
    Array3::from_shape_fn(shape.dimensions(), |(ix, iy, iz)| {
        let k = wavenumber_magnitude(shape, spacing_m, ix, iy, iz);
        if k == 0.0 {
            0.0
        } else {
            let half_phase = 0.5 * sound_speed_m_s * time_step_s * k;
            let kappa = half_phase.sin() / half_phase;
            (sound_speed_m_s * time_step_s * k * kappa).powi(2)
        }
    })
}

fn source_kappa_symbol(
    shape: GridShape,
    spacing_m: f64,
    sound_speed_m_s: f64,
    time_step_s: f64,
    ix: usize,
    iy: usize,
    iz: usize,
) -> f64 {
    let k = wavenumber_magnitude(shape, spacing_m, ix, iy, iz);
    (0.5 * sound_speed_m_s * time_step_s * k).cos()
}

fn source_indices_for_transmit(
    array: &MultiRowRingArray,
    transmit_index: usize,
    shape: GridShape,
    spacing_m: f64,
) -> KwaversResult<Vec<(usize, usize, usize)>> {
    array
        .cylindrical_source(transmit_index)
        .into_iter()
        .map(|point| point_to_grid_index(shape, spacing_m, point))
        .collect()
}

fn validate_array(array: &MultiRowRingArray) -> KwaversResult<()> {
    if array.circumferential_elements() < 2 || array.rows() == 0 {
        return Err(KwaversError::InvalidInput(
            "array topology requires at least two angular elements and one row".into(),
        ));
    }
    if array.element_count() != array.circumferential_elements() * array.rows() {
        return Err(KwaversError::DimensionMismatch(
            "array element count must equal circumferential_elements * rows".into(),
        ));
    }
    Ok(())
}

fn validate_frequencies(frequencies_hz: &[f64]) -> KwaversResult<()> {
    if frequencies_hz.is_empty() {
        return Err(KwaversError::InvalidInput(
            "frequencies_hz must not be empty".into(),
        ));
    }
    for &frequency_hz in frequencies_hz {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "frequency_hz must be positive and finite, got {frequency_hz}"
            )));
        }
    }
    Ok(())
}
