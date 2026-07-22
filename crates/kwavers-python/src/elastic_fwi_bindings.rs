//! PyO3 binding for the elastic shear-wave FWI (ADR 033).
//!
//! Thin conversion layer only: the acquisition setup, forward/adjoint physics,
//! and inversion all live in
//! `kwavers_solver::inverse::elastography::elastic_fwi`. This exposes the
//! convenience phantom-reconstruction entry point so Ch11's elastic FWI is
//! callable from Python like the other inverse solvers.

use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_solver::inverse::elastography::elastic_fwi::{
    reconstruct_lesion_transmission, TransmissionFwiParams,
};
use leto::Array3;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_py(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("elastic shear-wave FWI failed: {err}"))
}

/// Reconstruct the shear modulus `μ` from a known phantom via four-side
/// transmission elastic FWI (ADR 033).
///
/// `mu_true_pa` is the 2-D phantom shear-modulus map \`Pa` (lesion + background);
/// `c_shear_m_s` / `c_compression_m_s` set the homogeneous elastic background
/// (`c_p² ≥ 2 c_s²` required). Returns the recovered `μ` map \`Pa` of the same
/// shape. The synthetic shear-wave data are generated from `mu_true_pa` and
/// inverted from a homogeneous start; the GIL is released around the solve.
#[pyfunction]
#[pyo3(signature = (
    mu_true_pa,
    dx_m,
    density_kg_m3,
    c_shear_m_s,
    c_compression_m_s,
    n_steps = 200,
    iterations = 16,
    precond_eps = 0.1,
    mute_radius = 4,
))]
#[allow(clippy::too_many_arguments)]
pub fn elastic_shear_fwi_reconstruct(
    py: Python<'_>,
    mu_true_pa: PyReadonlyArray2<'_, f64>,
    dx_m: f64,
    density_kg_m3: f64,
    c_shear_m_s: f64,
    c_compression_m_s: f64,
    n_steps: usize,
    iterations: usize,
    precond_eps: f64,
    mute_radius: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let mu2d = mu_true_pa.as_array();
    let (nx, ny) = mu2d.dim();
    let mut mu_true = Array3::<f64>::zeros((nx, ny, 1));
    for i in 0..nx {
        for j in 0..ny {
            mu_true[[i, j, 0]] = mu2d[[i, j]];
        }
    }

    let grid = Grid::new(nx, ny, 1, dx_m, dx_m, dx_m).map_err(to_py)?;
    let medium = HomogeneousMedium::elastic_homogeneous(
        density_kg_m3,
        c_compression_m_s,
        c_shear_m_s,
        &grid,
    )
    .ok_or_else(|| {
        PyRuntimeError::new_err("invalid elastic medium (require positive ρ and c_p² ≥ 2·c_s²)")
    })?;

    let params = TransmissionFwiParams {
        n_steps,
        iterations,
        precond_eps,
        mute_radius,
        ..TransmissionFwiParams::default()
    };

    let rec = py
        .detach(|| reconstruct_lesion_transmission(&grid, &medium, &mu_true, &params))
        .map_err(to_py)?;

    let rec2d = Array2::from_shape_fn((nx, ny), |(i, j)| rec[[i, j, 0]]);
    Ok(rec2d.to_pyarray(py).into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(elastic_shear_fwi_reconstruct, m)?)?;
    Ok(())
}
