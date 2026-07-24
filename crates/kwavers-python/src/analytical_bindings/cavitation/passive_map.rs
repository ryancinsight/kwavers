//! Passive acoustic map and receiver-array PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Incoherent power sum of per-element PCD spectra (array integration over V_s).
///
/// Given a (n_channels, n_bins) matrix of per-element power spectra, returns the
/// array-integrated spectrum S(f) = sum_ch S_ch(f). Passive emissions from
/// independent collapse events are mutually incoherent, so their powers add.
///
/// Args:
///     channel_psds: (n_channels, n_bins) power spectra.
///
/// Returns:
///     Array-integrated PSD of length n_bins.
///
/// Reference:
///     Gyongy & Coussios (2010) IEEE TBME 57, 48.
#[pyfunction]
#[pyo3(signature = (channel_psds))]
pub fn integrate_receiver_array_psd(
    py: Python<'_>,
    channel_psds: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = channel_psds.as_array();
    let (n_channels, n_bins) = arr.dim();
    let flat: Vec<f64> = arr.iter().copied().collect();
    let result = cavitation::integrate_receiver_array_psd(&flat, n_channels, n_bins);
    Ok(result.to_pyarray(py).unbind())
}

/// Integrate a passive-acoustic-map emission-energy field over a sonication volume.
///
/// E(V_s) = sum_{voxels in mask} max(source`V`, 0) * dv_m3
///
/// Args:
///     source_map: Flattened emission-energy field.
///     mask: Flattened V_s mask (non-zero = inside V_s), same length.
///     dv_m3: Voxel volume `m³`.
///
/// Returns:
///     Total emission energy collected from V_s.
#[pyfunction]
#[pyo3(signature = (source_map, mask, dv_m3))]
pub fn emission_energy_in_volume(
    source_map: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
    dv_m3: f64,
) -> PyResult<f64> {
    let s = source_map
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let m = mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cavitation::emission_energy_in_volume(s, m, dv_m3))
}
