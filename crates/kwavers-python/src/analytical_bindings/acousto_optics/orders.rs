//! Acousto-optic diffraction-order intensity bindings.

use kwavers_physics::analytical::acousto_optics;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// Raman–Nath (thin-grating) order intensities Iₘ = Jₘ²(ν) for
/// m = −max_order..=max_order (index m + max_order).
#[pyfunction]
#[pyo3(signature = (nu, max_order))]
pub fn raman_nath_order_intensities(py: Python<'_>, nu: f64, max_order: u32) -> Py<PyArray1<f64>> {
    acousto_optics::raman_nath_order_intensities(nu, max_order)
        .into_pyarray(py)
        .unbind()
}

/// General Klein–Cook coupled-wave solver: exit order intensities |Eₗ(1)|² for
/// l = −max_order..=max_order (index l + max_order). Reduces to Raman–Nath
/// (Q→0) and Bragg (large Q, α=−½).
#[pyfunction]
#[pyo3(signature = (nu, q, incidence_alpha, max_order, n_steps))]
pub fn solve_coupled_orders(
    py: Python<'_>,
    nu: f64,
    q: f64,
    incidence_alpha: f64,
    max_order: u32,
    n_steps: usize,
) -> Py<PyArray1<f64>> {
    acousto_optics::solve_coupled_orders(nu, q, incidence_alpha, max_order, n_steps)
        .into_pyarray(py)
        .unbind()
}
