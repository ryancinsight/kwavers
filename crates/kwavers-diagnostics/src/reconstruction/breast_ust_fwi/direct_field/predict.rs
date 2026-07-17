use super::grid::{
    grid_index_to_point, outgoing_green, point_to_grid_index, source_mask, validate_sampling,
    wavenumber_magnitude, GridShape,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use kwavers_math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers_solver::inverse::fwi::frequency_domain::cbs::{
    pstd_modal_frequency_bin_response, pstd_modal_theta_squared, pstd_source_kappa_symbol,
    PstdTemporalBinConfig,
};
use leto::Array3;
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
                let k = wavenumber_magnitude(shape, spacing_m, ix, iy, iz);
                let symbol = pstd_source_kappa_symbol(k, time_step_s, sound_speed_m_s);
                spectrum[[ix, iy, iz]] *= Complex64::new(symbol, 0.0);
            }
        }
    }
    ifft_3d_complex_inplace(&mut spectrum);
    let [nx, ny, nz] = spectrum.shape();
    Ok(Array3::from_shape_vec(
        (nx, ny, nz),
        spectrum
            .as_slice()
            .expect("FFT spectrum must be densely stored")
            .to_vec(),
    )
    .expect("FFT spectrum shape must match its flattened length"))
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
    for ix in 0..shape.nx {
        for iy in 0..shape.ny {
            for iz in 0..shape.nz {
                let k = wavenumber_magnitude(shape, spacing_m, ix, iy, iz);
                let symbol = pstd_source_kappa_symbol(k, time_step_s, sound_speed_m_s);
                source_hat[[ix, iy, iz]] *= Complex64::new(symbol, 0.0);
            }
        }
    }

    let gain = 2.0 * sound_speed_m_s * time_step_s * source_amplitude_pa / spacing_m;
    let temporal_config = PstdTemporalBinConfig {
        frequency_hz,
        time_step_s,
        total_steps: steps,
        bin_start_step: bin_start,
        source_gain: gain,
    };
    for ix in 0..shape.nx {
        for iy in 0..shape.ny {
            for iz in 0..shape.nz {
                let k = wavenumber_magnitude(shape, spacing_m, ix, iy, iz);
                let theta_squared = pstd_modal_theta_squared(k, time_step_s, sound_speed_m_s);
                source_hat[[ix, iy, iz]] *=
                    pstd_modal_frequency_bin_response(theta_squared, temporal_config)?;
            }
        }
    }

    ifft_3d_complex_inplace(&mut source_hat);
    let bins = receiver_indices
        .iter()
        .map(|&(ix, iy, iz)| source_hat[[ix, iy, iz]])
        .collect();
    Ok(bins)
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
