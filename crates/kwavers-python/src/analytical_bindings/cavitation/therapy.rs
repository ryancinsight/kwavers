//! Histotripsy therapy-delivery PyO3 wrappers.

use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Deterministic pulse timeline for a rastered/interleaved sub-spot grid
/// sonication — the single source of truth for the pulsing-pattern diagram and
/// the monitor time-base.
///
/// Args:
///     n_subspots: number of sub-spots in the grid.
///     n_repetitions: number of passes (repetitions) over the grid.
///     pulse_duration_s: single histotripsy pulse duration [s] (microseconds).
///     prf_hz: rate of *fired* pulses (any spot) [Hz].
///     interleaved: True → round-robin (one pulse per spot per repetition);
///         False → sequential (all repetitions at a spot before the next).
///
/// Returns:
///     (onset_s, subspot_idx, repetition_idx, pulse_duration_s,
///      repetition_time_s, sonication_duration_s, n_repetitions) — the first
///     three are arrays of length n_subspots·n_repetitions in fire order.
#[pyfunction]
#[pyo3(signature = (n_subspots, n_repetitions, pulse_duration_s, prf_hz, interleaved=true))]
pub fn sonication_schedule(
    py: Python<'_>,
    n_subspots: usize,
    n_repetitions: usize,
    pulse_duration_s: f64,
    prf_hz: f64,
    interleaved: bool,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    f64,
    f64,
    f64,
    usize,
)> {
    use kwavers_physics::analytical::cavitation::{build_sonication_schedule, SonicationOrder};
    let order = if interleaved {
        SonicationOrder::Interleaved
    } else {
        SonicationOrder::Sequential
    };
    let s = build_sonication_schedule(n_subspots, n_repetitions, pulse_duration_s, prf_hz, order);
    let subspot_i64: Vec<i64> = s.subspot.iter().map(|&v| v as i64).collect();
    let repetition_i64: Vec<i64> = s.repetition.iter().map(|&v| v as i64).collect();
    Ok((
        s.onset_s.to_pyarray(py).unbind(),
        subspot_i64.to_pyarray(py).unbind(),
        repetition_i64.to_pyarray(py).unbind(),
        s.pulse_duration_s,
        s.repetition_time_s,
        s.sonication_duration_s,
        s.n_repetitions,
    ))
}

/// One-way delivered-pressure fraction at the focus: electronic-steering
/// efficiency × interface pressure transmission × tissue power-law attenuation ×
/// residual-gas (Commander–Prosperetti) attenuation over the path.
///
/// Args:
///     steering_eff, interface_z_prox, interface_z_focal, alpha_tissue_np_m,
///     path_len_m, void_beta, freq_hz, r0_m, c_liquid, rho_liquid, mu_liquid,
///     p0_pa, polytropic.
#[pyfunction]
#[pyo3(signature = (steering_eff, interface_z_prox, interface_z_focal, alpha_tissue_np_m,
                    path_len_m, void_beta, freq_hz, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn forward_delivery_fraction(
    steering_eff: f64,
    interface_z_prox: f64,
    interface_z_focal: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    void_beta: f64,
    freq_hz: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::forward_delivery_fraction(
            steering_eff,
            interface_z_prox,
            interface_z_focal,
            alpha_tissue_np_m,
            path_len_m,
            void_beta,
            freq_hz,
            r0_m,
            c_liquid,
            rho_liquid,
            mu_liquid,
            p0_pa,
            polytropic,
        ),
    )
}

/// Two-way (round-trip) amplitude fraction of a passive cavitation emission
/// measured back at the transducer — the genuine reflection/scattering/
/// attenuation loss that derates the *measured* signal vs the cavitation
/// actually produced at the focus.
///
/// Args:
///     interface_z_prox, interface_z_focal, alpha_tissue_np_m, path_len_m,
///     void_beta, freq_hz, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic.
#[pyfunction]
#[pyo3(signature = (interface_z_prox, interface_z_focal, alpha_tissue_np_m, path_len_m,
                    void_beta, freq_hz, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn received_signal_fraction(
    interface_z_prox: f64,
    interface_z_focal: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    void_beta: f64,
    freq_hz: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::received_signal_fraction(
            interface_z_prox,
            interface_z_focal,
            alpha_tissue_np_m,
            path_len_m,
            void_beta,
            freq_hz,
            r0_m,
            c_liquid,
            rho_liquid,
            mu_liquid,
            p0_pa,
            polytropic,
        ),
    )
}

/// Normal-incidence pressure transmission coefficient `2·z2/(z1+z2)` between two
/// acoustic impedances [Pa·s/m].
#[pyfunction]
#[pyo3(signature = (z1, z2))]
pub fn pressure_transmission_coefficient(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::pressure_transmission_coefficient(z1, z2))
}

/// Local peak-pressure enhancement `1 + |（z2−z1)/(z2+z1)|` at an acoustic interface
/// between impedances `z1` and `z2` [Pa·s/m] — the incident+reflected superposition
/// that makes cavitation nucleate preferentially at tissue boundaries (mild for
/// soft-tissue contrasts, approaching 2 against a gas-filled lacuna).
#[pyfunction]
#[pyo3(signature = (z1, z2))]
pub fn interface_pressure_enhancement(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::interface_pressure_enhancement(z1, z2))
}

/// Cavitation-susceptibility multiplier of already-fractionated tissue ("lesion
/// memory"): `S = 1 + k_immediate·f + k_lacuna·f·(1 − exp(−t_since/τ_lacuna))`, with
/// `f` the local fractionation, `time_since_lesion_s` the elapsed time, and
/// `tau_lacuna_s` the gas-evolution (lacuna formation) time constant. The delayed
/// lacuna term is negligible during the first procedure (`t ≪ τ`) and saturates on
/// re-treatment (`t ≫ τ`).
///
/// Args:
///     fractionation, time_since_lesion_s, tau_lacuna_s, k_immediate, k_lacuna.
#[pyfunction]
#[pyo3(signature = (fractionation, time_since_lesion_s, tau_lacuna_s, k_immediate=0.5, k_lacuna=4.0))]
pub fn lacuna_cavitation_susceptibility(
    fractionation: f64,
    time_since_lesion_s: f64,
    tau_lacuna_s: f64,
    k_immediate: f64,
    k_lacuna: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::lacuna_cavitation_susceptibility(
            fractionation,
            time_since_lesion_s,
            tau_lacuna_s,
            k_immediate,
            k_lacuna,
        ),
    )
}

/// Histotripsy mechanical cell-kill fraction from cumulative cavitation dose via the
/// Weibull survival dose–response `kill = 1 − exp(−(dose/d0)^k)` (the cumulative
/// cell-survival form underlying radiobiology's biologically-effective dose, but the
/// mechanism here is mechanical fractionation). `d0` = characteristic dose (≈63 % kill),
/// `weibull_k` > 1 the threshold/shoulder exponent.
#[pyfunction]
#[pyo3(signature = (dose, d0, weibull_k=2.5))]
pub fn histotripsy_kill_fraction(dose: f64, d0: f64, weibull_k: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::histotripsy_kill_fraction(dose, d0, weibull_k))
}

/// Lethal cumulative cavitation dose LD_x for cell-kill `fraction` (LD50 ⇒ 0.5):
/// `D = d0·(−ln(1−fraction))^(1/k)`. Inverse of `histotripsy_kill_fraction`.
#[pyfunction]
#[pyo3(signature = (fraction, d0, weibull_k=2.5))]
pub fn histotripsy_lethal_dose(fraction: f64, d0: f64, weibull_k: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::histotripsy_lethal_dose(fraction, d0, weibull_k))
}

/// Lateral semi-axis that keeps an anisotropic focal ellipsoid within a clearance
/// constraint.
#[pyfunction]
#[pyo3(signature = (natural_lateral_radius_m, clearance_m, axial_to_lateral_ratio))]
pub fn clipped_lateral_radius_for_clearance(
    natural_lateral_radius_m: f64,
    clearance_m: f64,
    axial_to_lateral_ratio: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::clipped_lateral_radius_for_clearance(
            natural_lateral_radius_m,
            clearance_m,
            axial_to_lateral_ratio,
        ),
    )
}

/// Check whether a beam-axis elongated focal ellipsoid remains inside an allowed mask.
#[pyfunction]
#[pyo3(signature = (allowed_mask, nx, ny, nz, center_x, center_y, center_z, lateral_radius_m, axial_radius_m, dx_m))]
#[allow(clippy::too_many_arguments)]
pub fn ellipsoid_respects_allowed_mask(
    allowed_mask: PyReadonlyArray1<bool>,
    nx: usize,
    ny: usize,
    nz: usize,
    center_x: usize,
    center_y: usize,
    center_z: usize,
    lateral_radius_m: f64,
    axial_radius_m: f64,
    dx_m: f64,
) -> PyResult<bool> {
    let mask = allowed_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(
        kwavers_physics::analytical::cavitation::ellipsoid_respects_allowed_mask(
            mask,
            nx,
            ny,
            nz,
            center_x,
            center_y,
            center_z,
            lateral_radius_m,
            axial_radius_m,
            dx_m,
        ),
    )
}

/// Apply receive-path and tissue-state scaling to a passive cavitation PSD.
#[pyfunction]
#[pyo3(signature = (psd, receive_fraction, susceptibility))]
pub fn scale_measured_emission_spectrum(
    py: Python<'_>,
    psd: PyReadonlyArray1<f64>,
    receive_fraction: f64,
    susceptibility: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p = psd
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = kwavers_physics::analytical::cavitation::scale_measured_emission_spectrum(
        p,
        receive_fraction,
        susceptibility,
    );
    Ok(out.to_pyarray(py).unbind())
}

/// Convert delivered cumulative histotripsy dose samples to kill fractions.
#[pyfunction]
#[pyo3(signature = (dose, d0, weibull_k))]
pub fn delivered_histotripsy_progress(
    py: Python<'_>,
    dose: PyReadonlyArray1<f64>,
    d0: f64,
    weibull_k: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let d = dose
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out =
        kwavers_physics::analytical::cavitation::delivered_histotripsy_progress(d, d0, weibull_k);
    Ok(out.to_pyarray(py).unbind())
}

/// Compare a modeled cavitation-cloud erosion curve with a k-wave or experimental reference.
///
/// Returns `(model_scale, rmse, normalized_rmse, max_abs_error,
/// max_relative_error, pearson_r, sample_count)`, or `None` when the paired
/// samples are invalid. The scale is the non-negative least-squares calibration
/// of the model erosion-efficiency coefficient; the remaining metrics validate
/// curve shape after that single empirical calibration.
#[pyfunction]
#[pyo3(signature = (reference_erosion, model_erosion))]
pub fn cloud_erosion_validation_metrics(
    reference_erosion: PyReadonlyArray1<f64>,
    model_erosion: PyReadonlyArray1<f64>,
) -> PyResult<Option<(f64, f64, f64, f64, f64, f64, usize)>> {
    let reference = reference_erosion
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let model = model_erosion
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(
        kwavers_physics::analytical::cavitation::cloud_erosion_validation_metrics(reference, model)
            .map(|m| {
                (
                    m.model_scale,
                    m.rmse,
                    m.normalized_rmse,
                    m.max_abs_error,
                    m.max_relative_error,
                    m.pearson_r,
                    m.sample_count,
                )
            }),
    )
}

