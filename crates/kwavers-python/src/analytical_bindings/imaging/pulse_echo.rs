//! PyO3 wrappers for pulse-echo RF and B-mode helpers.

use crate::breast_fwi_bindings::complex_compat::{
    leto1_to_nd1, leto2_to_nd2, nd_to_leto1, nd_to_leto2,
};
use kwavers_analysis::signal_processing::b_mode::envelope as core_bmode_envelope;
use kwavers_physics::analytical::pulse_echo::{
    bmode_db_fixed_reference as core_bmode_db_fixed_reference,
    delta_bmode_db as core_delta_bmode_db, simulate_receive_rf as core_simulate_receive_rf,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// First-Born synthetic-aperture pulse-echo channel RF from point scatterers.
///
/// Each scatterer re-radiates a Gaussian-modulated tone burst reaching element `s`
/// at the one-way time of flight `|r_i − r_s|/c` (1/r spreading, reflectivity
/// weighting); contributions sum coherently. Beamforming the result with the one-way
/// `beamform_image_delay_and_sum` reconstructs the reflectivity map — a genuine
/// receive-data → B-mode pipeline.
///
/// Args:
///     scat_pos: (n_scat, 3) scatterer positions `m`.
///     scat_amp: (n_scat,) reflectivity weights.
///     elem_pos: (n_elem, 3) array element positions `m`.
///     c, fs, f0: sound speed [m/s], sampling `Hz`, imaging centre frequency `Hz`.
///     frac_bw: fractional −6 dB pulse bandwidth.
///     n_samples: RF record length `samples`.
///
/// Returns:
///     (n_elem, n_samples) channel RF.
#[pyfunction]
#[pyo3(signature = (scat_pos, scat_amp, elem_pos, c, fs, f0, n_samples, frac_bw=0.6))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_receive_rf<'py>(
    py: Python<'py>,
    scat_pos: PyReadonlyArray2<'py, f64>,
    scat_amp: PyReadonlyArray1<'py, f64>,
    elem_pos: PyReadonlyArray2<'py, f64>,
    c: f64,
    fs: f64,
    f0: f64,
    n_samples: usize,
    frac_bw: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let sp = scat_pos.as_array();
    let sa = scat_amp.as_array();
    let ep = elem_pos.as_array();
    if sp.ncols() != 3 || ep.ncols() != 3 {
        return Err(PyValueError::new_err(
            "scat_pos and elem_pos must have shape (n, 3)",
        ));
    }
    if sp.nrows() != sa.len() {
        return Err(PyValueError::new_err(
            "scat_pos rows must match scat_amp length",
        ));
    }
    let sp_leto = nd_to_leto2(sp.to_owned());
    let sa_leto = nd_to_leto1(sa.to_owned());
    let ep_leto = nd_to_leto2(ep.to_owned());
    let rf = core_simulate_receive_rf(
        sp_leto.view(),
        sa_leto.view(),
        ep_leto.view(),
        c,
        fs,
        f0,
        frac_bw,
        n_samples,
    );
    Ok(leto2_to_nd2(rf).to_pyarray(py).unbind())
}

/// B-mode envelope detection: the analytic-signal magnitude `|z(t)|`, where
/// `z = s + i·H{s}` and `H` is the Hilbert transform (book §9.1.3, Theorem 9.1).
/// For a narrowband RF line this recovers the modulating amplitude `A(t)`.
///
/// Args:
///     rf: Beamformed RF line.
///
/// Returns:
///     Envelope `|z(t)|`, same length as `rf`.
#[pyfunction]
#[pyo3(signature = (rf,))]
pub fn bmode_envelope(py: Python<'_>, rf: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let rf_arr = nd_to_leto1(rf.as_array().to_owned());
    let env = py.detach(|| core_bmode_envelope(&rf_arr));
    let env = leto1_to_nd1(env);
    Ok(env.to_pyarray(py).unbind())
}

/// Log-compress an envelope image with a fixed sequence reference.
#[pyfunction]
#[pyo3(signature = (envelope, reference, floor_db))]
pub fn bmode_db_fixed_reference(
    py: Python<'_>,
    envelope: PyReadonlyArray1<f64>,
    reference: f64,
    floor_db: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let env = envelope
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = py.detach(|| core_bmode_db_fixed_reference(env, reference, floor_db));
    Ok(out.to_pyarray(py).unbind())
}

/// Baseline-relative delta B-mode in dB.
#[pyfunction]
#[pyo3(signature = (envelope, baseline, epsilon=1.0e-12))]
pub fn delta_bmode_db(
    py: Python<'_>,
    envelope: PyReadonlyArray1<f64>,
    baseline: PyReadonlyArray1<f64>,
    epsilon: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let env = envelope
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let base = baseline
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = py.detach(|| core_delta_bmode_db(env, base, epsilon));
    Ok(out.to_pyarray(py).unbind())
}
