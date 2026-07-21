//! Passive cavitation receive and coherence PyO3 wrappers.

use crate::array_utils::{copy_pyarray1_to_vec, copy_pyarray2_to_vec, vec_to_pyarray1, vec_to_pyarray2};
use kwavers_physics::analytical::cavitation;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Propagate one cavitation source PSD to passive receiver-channel PSDs.
#[pyfunction]
#[pyo3(signature = (source_psd, source_xyz, receiver_xyz, alpha_np_m))]
pub fn receiver_channel_psd_from_source(
    py: Python<'_>,
    source_psd: PyReadonlyArray1<f64>,
    source_xyz: PyReadonlyArray1<f64>,
    receiver_xyz: PyReadonlyArray2<f64>,
    alpha_np_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let psd = copy_pyarray1_to_vec(&source_psd)?;
    let src = copy_pyarray1_to_vec(&source_xyz)?;
    let (recv_flat, recv_shape) = copy_pyarray2_to_vec(&receiver_xyz)?;
    if src.len() != 3 || recv_shape[1] != 3 {
        return Err(PyRuntimeError::new_err(
            "source_xyz must have length 3 and receiver_xyz shape (n, 3)",
        ));
    }
    let n_receivers = recv_shape[0];
    let flat = py.detach(|| {
        cavitation::receiver_channel_psd_from_source(
            &psd,
            [src[0], src[1], src[2]],
            &recv_flat,
            alpha_np_m,
        )
    });
    vec_to_pyarray2(py, [n_receivers, psd.len()], flat)
}

/// Sum receiver-channel PSDs into the measured array spectrum.
#[pyfunction]
#[pyo3(signature = (channel_psd))]
pub fn integrate_channel_psd(
    py: Python<'_>,
    channel_psd: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let (flat, shape) = copy_pyarray2_to_vec(&channel_psd)?;
    let out = py.detach(|| cavitation::integrate_channel_psd(&flat, shape[0], shape[1]));
    Ok(vec_to_pyarray1(py, out))
}

/// Synthetic passive RF received from one cavitation point source.
#[pyfunction]
#[pyo3(signature = (
    receiver_xyz,
    source_xyz,
    n_samples,
    sampling_frequency_hz,
    sound_speed_m_s,
    frequency_hz,
    n_cycles
))]
#[allow(clippy::too_many_arguments)]
pub fn passive_cavitation_point_source_rf(
    py: Python<'_>,
    receiver_xyz: PyReadonlyArray2<f64>,
    source_xyz: PyReadonlyArray1<f64>,
    n_samples: usize,
    sampling_frequency_hz: f64,
    sound_speed_m_s: f64,
    frequency_hz: f64,
    n_cycles: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let (recv_flat, recv_shape) = copy_pyarray2_to_vec(&receiver_xyz)?;
    let src = copy_pyarray1_to_vec(&source_xyz)?;
    if recv_shape[1] != 3 || src.len() != 3 {
        return Err(PyValueError::new_err(
            "receiver_xyz must have shape (n, 3) and source_xyz must have length 3",
        ));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be positive"));
    }
    let flat = py.detach(|| {
        cavitation::passive_cavitation_point_source_rf(
            &recv_flat,
            [src[0], src[1], src[2]],
            n_samples,
            sampling_frequency_hz,
            sound_speed_m_s,
            frequency_hz,
            n_cycles,
        )
    });
    if flat.is_empty() {
        return Err(PyValueError::new_err(
            "receiver/source coordinates and acoustic parameters must be finite and positive",
        ));
    }
    vec_to_pyarray2(py, [recv_shape[0], n_samples], flat)
}

/// Van Cittert-Zernike coherence for an incoherent planar source.
#[pyfunction]
#[pyo3(signature = (delta_x_m, source_extent_m, depth_m, wavelength_m))]
pub fn van_cittert_zernike_coherence(
    py: Python<'_>,
    delta_x_m: PyReadonlyArray1<f64>,
    source_extent_m: f64,
    depth_m: f64,
    wavelength_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let delta_x = copy_pyarray1_to_vec(&delta_x_m)?;
    let coherence =
        cavitation::van_cittert_zernike_coherence(&delta_x, source_extent_m, depth_m, wavelength_m)
            .map_err(PyValueError::new_err)?;
    Ok(vec_to_pyarray1(py, coherence))
}
