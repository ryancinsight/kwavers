//! PyO3 wrappers for Ali 2025 reduced-domain preparation.

use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    derive_reduced_breast_ust_array_geometry, derive_reduced_breast_ust_array_plan,
    prepare_reduced_breast_ust_phantom, BreastUstReducedArrayGeometry, BreastUstReducedArrayPlan,
    BreastUstReducedArrayRowPolicy, BreastUstReducedPhantom,
};
use numpy::{IntoPyArray, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

#[pyfunction]
pub fn prepare_breast_fwi_reduced_phantom<'py>(
    py: Python<'py>,
    sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    source_spacing_m: f64,
    max_shape: (usize, usize, usize),
    decimation: usize,
    dataset_path: &str,
    source_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let source = sound_speed_m_s.as_array().to_owned();
    let reduced = py
        .detach(|| {
            prepare_reduced_breast_ust_phantom(&source, source_spacing_m, max_shape, decimation)
        })
        .map_err(kwavers_to_value_py)?;
    reduced_phantom_to_dict(py, reduced, dataset_path, source_path)
}

#[pyfunction]
pub fn derive_breast_fwi_reduced_array_geometry<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    spacing_m: f64,
    rows: usize,
    diameter_m: Option<f64>,
    row_spacing_m: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let geometry =
        derive_reduced_breast_ust_array_geometry(shape, spacing_m, rows, diameter_m, row_spacing_m)
            .map_err(kwavers_to_value_py)?;
    reduced_array_geometry_to_dict(py, geometry)
}

#[pyfunction]
pub fn derive_breast_fwi_reduced_array_plan<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    spacing_m: f64,
    row_policy: &str,
    rows: Option<usize>,
    diameter_m: Option<f64>,
    row_spacing_m: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let plan = derive_reduced_breast_ust_array_plan(
        shape,
        spacing_m,
        parse_row_policy(row_policy)?,
        rows,
        diameter_m,
        row_spacing_m,
    )
    .map_err(kwavers_to_value_py)?;
    reduced_array_plan_to_dict(py, plan)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prepare_breast_fwi_reduced_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(
        derive_breast_fwi_reduced_array_geometry,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(derive_breast_fwi_reduced_array_plan, m)?)?;
    Ok(())
}

fn reduced_phantom_to_dict<'py>(
    py: Python<'py>,
    reduced: BreastUstReducedPhantom,
    dataset_path: &str,
    source_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("sound_speed_m_s", reduced.sound_speed_m_s.into_pyarray(py))?;
    out.set_item(
        "initial_sound_speed_m_s",
        reduced.initial_sound_speed_m_s.into_pyarray(py),
    )?;
    out.set_item("original_shape", reduced.original_shape)?;
    out.set_item("reduced_shape", reduced.reduced_shape)?;
    out.set_item("source_spacing_m", reduced.source_spacing_m)?;
    out.set_item("effective_spacing_m", reduced.effective_spacing_m)?;
    out.set_item("dataset_path", dataset_path)?;
    out.set_item("source_path", source_path)?;
    Ok(out)
}

fn reduced_array_geometry_to_dict<'py>(
    py: Python<'py>,
    geometry: BreastUstReducedArrayGeometry,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("diameter_m", geometry.diameter_m)?;
    out.set_item("row_spacing_m", geometry.row_spacing_m)?;
    Ok(out)
}

fn reduced_array_plan_to_dict<'py>(
    py: Python<'py>,
    plan: BreastUstReducedArrayPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let out = reduced_array_geometry_to_dict(py, plan.geometry)?;
    out.set_item("rows", plan.rows)?;
    out.set_item("row_policy", row_policy_name(plan.row_policy))?;
    Ok(out)
}

fn parse_row_policy(policy: &str) -> PyResult<BreastUstReducedArrayRowPolicy> {
    match policy {
        "smoke_single_ring" => Ok(BreastUstReducedArrayRowPolicy::SmokeSingleRing),
        "explicit" => Ok(BreastUstReducedArrayRowPolicy::Explicit),
        "table1_parity_interior" => {
            Ok(BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage)
        }
        other => Err(PyValueError::new_err(format!(
            "unknown breast FWI reduced-array row_policy '{other}'"
        ))),
    }
}

fn row_policy_name(policy: BreastUstReducedArrayRowPolicy) -> &'static str {
    match policy {
        BreastUstReducedArrayRowPolicy::SmokeSingleRing => "smoke_single_ring",
        BreastUstReducedArrayRowPolicy::Explicit => "explicit",
        BreastUstReducedArrayRowPolicy::Table1ParityInteriorCoverage => "table1_parity_interior",
    }
}

fn kwavers_to_value_py(err: kwavers_core::error::KwaversError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
