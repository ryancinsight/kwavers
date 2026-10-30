//! PyO3 bindings: CW angular-spectrum propagation (Zeng & McGough 2008).
//!
//! Thin bridge over `kwavers_solver::analytical::angular_spectrum`: NumPy
//! complex128 planes in, NumPy complex128 volumes out. All propagator math
//! (spectral grids, FFTs, absorption, restriction filter) lives in Rust;
//! the Python parity helper keeps signature parsing and medium-dict
//! handling only.

use crate::array_utils::{leto2_to_pyarray2, leto3_to_pyarray3, pyarray2_to_leto2};
use Complex64;
use kwavers_medium::absorption::power_law::power_law_db_cm_to_np_m;
use kwavers_solver::analytical::angular_spectrum::{CwMedium, backproject_cw, propagate_cw};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Project a 2-D CW pressure plane using the angular spectrum method.
///
/// Mirrors k-wave-python `angular_spectrum_cw` semantics; see
/// `pykwavers.kwave_parity.angular_spectrum_cw` for the user-facing contract.
///
/// Args:
///     input_plane: Complex pressure amplitude on the source plane `Pa`,
///         shape (Nx, Ny).
///     dx: Isotropic grid spacing [m].
///     z_planes: Propagation distance(s) from the source plane [m].
///     f0: Source frequency [Hz].
///     sound_speed: Sound speed [m/s].
///     alpha_db: Optional `(alpha_coeff, alpha_power)` power-law prefactor
///         [dB/(MHz^y·cm)], converted to Np/m at *f0* by the canonical
///         single-owner conversion. Default None (lossless).
///     angular_restriction: Apply the Zeng & McGough angular-restriction
///         filter. Default True.
///     grid_expansion: Zero-pad width around the input plane. Default 0.
///
/// Returns:
///     Complex pressure, shape (Nx, Ny, Nz).
#[pyfunction]
#[pyo3(
    signature = (input_plane, dx, z_planes, f0, sound_speed, alpha_db=None, angular_restriction=true, grid_expansion=0)
)]
#[allow(clippy::too_many_arguments)]
pub fn angular_spectrum_cw<'py>(
    py: Python<'py>,
    input_plane: PyReadonlyArray2<'py, Complex64>,
    dx: f64,
    z_planes: Vec<f64>,
    f0: f64,
    sound_speed: f64,
    alpha_db: Option<(f64, f64)>,
    angular_restriction: bool,
    grid_expansion: usize,
) -> PyResult<Py<PyArray3<Complex64>>> {
    let input = pyarray2_to_leto2(&input_plane)?;
    let alpha_np_per_m = alpha_db
        .map(|(coeff, power)| power_law_db_cm_to_np_m(coeff, power, f0))
        .unwrap_or(0.0);
    let medium = CwMedium {
        sound_speed,
        alpha_np_per_m,
    };
    let out = py.detach(|| {
        propagate_cw(
            &input,
            dx,
            &z_planes,
            f0,
            medium,
            angular_restriction,
            grid_expansion,
        )
    });
    let out = out.map_err(|e| PyValueError::new_err(e.to_string()))?;
    leto3_to_pyarray3(py, out)
}

/// Reconstruct a source plane from a CW measurement via the backward
/// (time-reversed) angular spectrum.
///
/// Args:
///     measurement_plane: Complex pressure at the measurement plane `Pa`,
///         shape (Nx, Ny).
///     dx: Isotropic grid spacing [m].
///     z_m: Propagation distance from source to measurement plane [m].
///     f0: Source frequency [Hz].
///     sound_speed: Sound speed [m/s].
///     angular_restriction: Apply the angular-restriction filter. Default True.
///
/// Returns:
///     Reconstructed source plane, shape (Nx, Ny).
#[pyfunction]
#[pyo3(signature = (measurement_plane, dx, z_m, f0, sound_speed, angular_restriction=true))]
pub fn backward_angular_spectrum_cw<'py>(
    py: Python<'py>,
    measurement_plane: PyReadonlyArray2<'py, Complex64>,
    dx: f64,
    z_m: f64,
    f0: f64,
    sound_speed: f64,
    angular_restriction: bool,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let measurement = pyarray2_to_leto2(&measurement_plane)?;
    let out = py.detach(|| {
        backproject_cw(
            &measurement,
            dx,
            z_m,
            f0,
            sound_speed,
            angular_restriction,
        )
    });
    let out = out.map_err(|e| PyValueError::new_err(e.to_string()))?;
    leto2_to_pyarray2(py, out)
}
