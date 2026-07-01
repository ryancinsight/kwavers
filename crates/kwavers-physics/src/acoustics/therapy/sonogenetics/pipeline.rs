//! Vector sonogenetics pipelines shared by Rust callers and PyO3 bindings.

use kwavers_core::constants::fundamental::BOLTZMANN as K_B;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array1, Array3};

use super::{
    boltzmann_p_open, compute_membrane_tension, BoltzmannGatingParams, CellMembraneParams,
    LifNeuron, LifParams,
};

const MEMBRANE_TENSION_N_PER_M_TO_MN_PER_M: f64 = 1.0e3;
const MEMBRANE_TENSION_MN_PER_M_TO_N_PER_M: f64 = 1.0e-3;
const DEFAULT_MEMBRANE_THICKNESS_M: f64 = 5.0e-9;

/// Output of [`gaussian_beam_pressure_field`].
#[derive(Debug, Clone)]
pub struct GaussianBeamPressureField {
    /// X coordinate field [m].
    pub x_m: Array3<f64>,
    /// Y coordinate field [m].
    pub y_m: Array3<f64>,
    /// Z coordinate field [m].
    pub z_m: Array3<f64>,
    /// Pressure amplitude field [Pa].
    pub pressure_pa: Array3<f64>,
}

/// Output of [`simulate_lif_trace`].
#[derive(Debug, Clone, PartialEq)]
pub struct LifTrace {
    /// Membrane voltage trace [V].
    pub voltage_v: Vec<f64>,
    /// Chronological spike times [s].
    pub spike_times_s: Vec<f64>,
}

/// Output of [`lif_response_probability`].
#[derive(Debug, Clone, PartialEq)]
pub struct LifResponseProbability {
    /// Binary spike train sampled at the neural time step.
    pub spike_train: Vec<f64>,
    /// Gaussian-smoothed spike-density response probability in `[0, 1]`.
    pub response_probability: Vec<f64>,
}

fn invalid_value(parameter: &'static str, value: f64, reason: &'static str) -> KwaversError {
    KwaversError::Validation(ValidationError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        reason: reason.to_owned(),
    })
}

fn validate_positive(parameter: &'static str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_value(
            parameter,
            value,
            "must be finite and strictly positive",
        ))
    }
}

/// Convert acoustic pressure samples [Pa] to membrane tension [mN/m].
///
/// Formula: `I = p^2 / (2 rho c)`, then `Delta T = I R / (2 c)`.
pub fn pressure_to_membrane_tension_mn_m(
    pressure_pa: &[f64],
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    cell_radius_m: f64,
) -> KwaversResult<Vec<f64>> {
    validate_positive("density_kg_m3", density_kg_m3)?;
    validate_positive("sound_speed_m_s", sound_speed_m_s)?;
    validate_positive("cell_radius_m", cell_radius_m)?;
    if pressure_pa.iter().any(|v| !v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "pressure_pa contains non-finite values".to_owned(),
        ));
    }

    let n = pressure_pa.len();
    let intensity_1d = Array1::from_iter(
        pressure_pa
            .iter()
            .map(|&p| p * p / (2.0 * density_kg_m3 * sound_speed_m_s)),
    );
    let intensity_3d = intensity_1d
        .into_shape_with_order((n, 1, 1))
        .expect("1-D-to-(N,1,1) reshape is infallible");
    let sound_speed_3d = Array3::from_elem((n, 1, 1), sound_speed_m_s);
    let params = CellMembraneParams {
        radius_m: cell_radius_m,
        thickness_m: DEFAULT_MEMBRANE_THICKNESS_M,
    };
    Ok(
        compute_membrane_tension(&intensity_3d, &sound_speed_3d, &params)
            .into_shape_with_order(n)
            .expect("(N,1,1)-to-1-D reshape is infallible")
            .iter()
            .map(|&t| t * MEMBRANE_TENSION_N_PER_M_TO_MN_PER_M)
            .collect(),
    )
}

/// Compute Boltzmann open probability from membrane tension [mN/m].
pub fn boltzmann_open_probability_from_tension_mn_m(
    tension_mn_m: &[f64],
    half_tension_mn_m: f64,
    slope_mn_m: f64,
    temperature_k: f64,
) -> KwaversResult<Vec<f64>> {
    validate_positive("temperature_k", temperature_k)?;
    validate_positive("slope_mn_m", slope_mn_m)?;
    if tension_mn_m.iter().any(|v| !v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "tension_mn_m contains non-finite values".to_owned(),
        ));
    }

    let n = tension_mn_m.len();
    let gating_area_m2 = K_B * temperature_k / (slope_mn_m * MEMBRANE_TENSION_MN_PER_M_TO_N_PER_M);
    let params = BoltzmannGatingParams {
        gating_area_m2,
        half_tension_n_per_m: half_tension_mn_m * MEMBRANE_TENSION_MN_PER_M_TO_N_PER_M,
        single_channel_conductance_s: 0.0,
        reversal_potential_v: 0.0,
    };
    let tension_3d = Array1::from_iter(
        tension_mn_m
            .iter()
            .map(|&t| t * MEMBRANE_TENSION_MN_PER_M_TO_N_PER_M),
    )
    .into_shape_with_order((n, 1, 1))
    .expect("1-D-to-(N,1,1) reshape is infallible");
    Ok(boltzmann_p_open(&tension_3d, &params, temperature_k)?
        .into_shape_with_order(n)
        .expect("(N,1,1)-to-1-D reshape is infallible")
        .to_vec())
}

/// Compute normalized coupled mechanochemical channel drive from acoustic pressure.
#[allow(clippy::too_many_arguments)]
pub fn coupled_channel_drive(
    pressure_pa: &[f64],
    half_tensions_mn_m: &[f64],
    slopes_mn_m: &[f64],
    conductance_weights: &[f64],
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    cell_radius_m: f64,
    temperature_k: f64,
) -> KwaversResult<Vec<f64>> {
    let channel_count = half_tensions_mn_m.len();
    if slopes_mn_m.len() != channel_count || conductance_weights.len() != channel_count {
        return Err(KwaversError::InvalidInput(
            "half_tensions_mn_m, slopes_mn_m, and conductance_weights must have equal length"
                .to_owned(),
        ));
    }
    for (index, &slope) in slopes_mn_m.iter().enumerate() {
        if !(slope.is_finite() && slope > 0.0) {
            return Err(KwaversError::InvalidInput(format!(
                "slopes_mn_m[{index}] = {slope} must be finite and strictly positive"
            )));
        }
    }

    let tension_mn_m = pressure_to_membrane_tension_mn_m(
        pressure_pa,
        density_kg_m3,
        sound_speed_m_s,
        cell_radius_m,
    )?;
    let mut drive = vec![0.0; pressure_pa.len()];
    let norm = conductance_weights.iter().map(|w| w.abs()).sum::<f64>();
    for channel in 0..channel_count {
        let p_open = boltzmann_open_probability_from_tension_mn_m(
            &tension_mn_m,
            half_tensions_mn_m[channel],
            slopes_mn_m[channel],
            temperature_k,
        )?;
        for (out, probability) in drive.iter_mut().zip(p_open) {
            *out += conductance_weights[channel] * probability;
        }
    }
    if norm > 0.0 {
        for value in &mut drive {
            *value = (*value / norm).clamp(-1.0, 1.0);
        }
    }
    Ok(drive)
}

/// Generate an analytical 3-D paraxial Gaussian beam pressure field.
#[allow(clippy::too_many_arguments)]
pub fn gaussian_beam_pressure_field(
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    peak_pressure_pa: f64,
    lateral_fwhm_m: f64,
    axial_fwhm_m: f64,
) -> KwaversResult<GaussianBeamPressureField> {
    validate_positive("dx_m", dx_m)?;
    validate_positive("dy_m", dy_m)?;
    validate_positive("dz_m", dz_m)?;
    validate_positive("lateral_fwhm_m", lateral_fwhm_m)?;
    validate_positive("axial_fwhm_m", axial_fwhm_m)?;

    let fwhm_to_sigma = (2.0 * (2.0_f64 * std::f64::consts::LN_2).sqrt()).recip();
    let sigma_lat = lateral_fwhm_m * fwhm_to_sigma;
    let sigma_ax = axial_fwhm_m * fwhm_to_sigma;
    let inv2sl2 = 0.5 / (sigma_lat * sigma_lat);
    let inv2sa2 = 0.5 / (sigma_ax * sigma_ax);
    let cx = (nx as f64 - 1.0) / 2.0;
    let cy = (ny as f64 - 1.0) / 2.0;
    let cz = (nz as f64 - 1.0) / 2.0;

    let mut x_m = Array3::<f64>::zeros((nx, ny, nz));
    let mut y_m = Array3::<f64>::zeros((nx, ny, nz));
    let mut z_m = Array3::<f64>::zeros((nx, ny, nz));
    let mut pressure_pa = Array3::<f64>::zeros((nx, ny, nz));
    for i in 0..nx {
        let xi = (i as f64 - cx) * dx_m;
        for j in 0..ny {
            let yj = (j as f64 - cy) * dy_m;
            let lat_term = (xi * xi + yj * yj) * inv2sl2;
            for k in 0..nz {
                let zk = (k as f64 - cz) * dz_m;
                x_m[[i, j, k]] = xi;
                y_m[[i, j, k]] = yj;
                z_m[[i, j, k]] = zk;
                pressure_pa[[i, j, k]] = peak_pressure_pa * (-(lat_term + zk * zk * inv2sa2)).exp();
            }
        }
    }

    Ok(GaussianBeamPressureField {
        x_m,
        y_m,
        z_m,
        pressure_pa,
    })
}

/// Simulate a LIF neuron driven by an ion-current trace.
pub fn simulate_lif_trace(
    i_ion_a: &[f64],
    dt_s: f64,
    params: LifParams,
) -> KwaversResult<LifTrace> {
    validate_positive("dt_s", dt_s)?;
    if !params.is_valid() {
        return Err(KwaversError::InvalidInput(
            "LifParams are not physically valid".to_owned(),
        ));
    }
    if i_ion_a.iter().any(|v| !v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "i_ion_a contains non-finite values".to_owned(),
        ));
    }

    let mut neuron = LifNeuron::new(params);
    let mut voltage_v = Vec::with_capacity(i_ion_a.len());
    for (index, &current) in i_ion_a.iter().enumerate() {
        neuron.step(current, dt_s, index as f64 * dt_s)?;
        voltage_v.push(neuron.membrane_voltage());
    }
    Ok(LifTrace {
        voltage_v,
        spike_times_s: neuron.spike_times().to_vec(),
    })
}

/// Convert LIF spike times into a Gaussian-smoothed response probability.
///
/// The spike train is sampled on the same uniform grid as the LIF trace, then
/// convolved with a normalized Gaussian kernel and divided by `f_max_hz`.
///
/// # Errors
/// Returns an error when the sample count is zero, `dt_s`, `smoothing_sigma_s`,
/// or `f_max_hz` is nonpositive/nonfinite, or any spike time is nonfinite.
pub fn lif_response_probability(
    spike_times_s: &[f64],
    n_samples: usize,
    dt_s: f64,
    smoothing_sigma_s: f64,
    f_max_hz: f64,
) -> KwaversResult<LifResponseProbability> {
    if n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "n_samples must be greater than zero".to_owned(),
        ));
    }
    validate_positive("dt_s", dt_s)?;
    validate_positive("smoothing_sigma_s", smoothing_sigma_s)?;
    validate_positive("f_max_hz", f_max_hz)?;
    if spike_times_s.iter().any(|value| !value.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "spike_times_s contains non-finite values".to_owned(),
        ));
    }

    let mut spike_train = vec![0.0; n_samples];
    for &spike_time_s in spike_times_s {
        let index = (spike_time_s / dt_s).round();
        if index >= 0.0 {
            let index = index as usize;
            if index < n_samples {
                spike_train[index] = 1.0;
            }
        }
    }

    let sigma_samples = smoothing_sigma_s / dt_s;
    let radius = (4.0 * sigma_samples).ceil() as isize;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let mut kernel_sum = 0.0;
    for offset in -radius..=radius {
        let x = offset as f64 / sigma_samples;
        let value = (-0.5 * x * x).exp();
        kernel.push(value);
        kernel_sum += value;
    }
    for value in &mut kernel {
        *value /= kernel_sum;
    }

    let mut response_probability = vec![0.0; n_samples];
    for (sample, output) in response_probability.iter_mut().enumerate() {
        let mut spike_density_hz = 0.0;
        for (kernel_index, &weight) in kernel.iter().enumerate() {
            let offset = kernel_index as isize - radius;
            let source = sample as isize + offset;
            if (0..n_samples as isize).contains(&source) {
                spike_density_hz += spike_train[source as usize] * weight / dt_s;
            }
        }
        *output = (spike_density_hz / f_max_hz).clamp(0.0, 1.0);
    }

    Ok(LifResponseProbability {
        spike_train,
        response_probability,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn pressure_to_tension_matches_laplace_reference() {
        let out = pressure_to_membrane_tension_mn_m(&[1.0e5], 1000.0, 1500.0, 10.0e-6).unwrap();
        let intensity = 1.0e10 / (2.0 * 1000.0 * 1500.0);
        let expected_mn_m = intensity * 10.0e-6 / (2.0 * 1500.0) * 1.0e3;
        assert_relative_eq!(out[0], expected_mn_m, max_relative = 1.0e-12);
    }

    #[test]
    fn coupled_drive_is_bounded() {
        let drive = coupled_channel_drive(
            &[0.0, 1.0e5, 2.0e5],
            &[1.0, 2.0],
            &[0.5, 0.75],
            &[1.0, -0.5],
            1000.0,
            1500.0,
            10.0e-6,
            310.0,
        )
        .unwrap();
        assert_eq!(drive.len(), 3);
        assert!(drive.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn gaussian_beam_places_peak_at_center_for_odd_grid() {
        let field =
            gaussian_beam_pressure_field(3, 3, 3, 1.0e-3, 1.0e-3, 1.0e-3, 2.0, 2.0e-3, 2.0e-3)
                .unwrap();
        assert_relative_eq!(field.pressure_pa[[1, 1, 1]], 2.0, max_relative = 1.0e-12);
    }

    #[test]
    fn lif_trace_delegates_spike_state() {
        let trace =
            simulate_lif_trace(&vec![200.0e-12; 2000], 0.05e-3, LifParams::default()).unwrap();
        assert_eq!(trace.voltage_v.len(), 2000);
        assert!(!trace.spike_times_s.is_empty());
    }

    #[test]
    fn lif_response_probability_is_bounded_and_input_sensitive() {
        let empty = lif_response_probability(&[], 101, 0.001, 0.01, 100.0).unwrap();
        let active = lif_response_probability(&[0.05], 101, 0.001, 0.01, 100.0).unwrap();

        assert_eq!(active.spike_train.len(), 101);
        assert_eq!(active.response_probability.len(), 101);
        assert_eq!(active.spike_train[50], 1.0);
        assert!(active
            .response_probability
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
        assert!(
            active.response_probability[50] > empty.response_probability[50],
            "spike should raise the smoothed response"
        );
    }

    #[test]
    fn lif_response_probability_rejects_invalid_domains() {
        assert!(lif_response_probability(&[0.0], 0, 0.001, 0.01, 100.0).is_err());
        assert!(lif_response_probability(&[0.0], 10, 0.0, 0.01, 100.0).is_err());
        assert!(lif_response_probability(&[f64::NAN], 10, 0.001, 0.01, 100.0).is_err());
    }
}
