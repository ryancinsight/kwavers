//! Shared private helpers for theranostic_bindings sub-modules.

use kwavers_therapy::therapy::theranostic_guidance::{
    DevicePlacementMetrics, PlacementContext, Point3, ReconstructionMetrics,
    VolumeReconstructionMetrics,
};
use kwavers_core::error::KwaversError;
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(super) fn labels_from_volume(volume: ndarray::Array3<f64>) -> ndarray::Array3<i16> {
    volume.mapv(|value| value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
}

pub(super) fn kwavers_to_py(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers theranostic inverse failed: {err}"))
}

pub(super) fn placement_dict<'py>(
    py: Python<'py>,
    metrics: &DevicePlacementMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("min_body_clearance_m", metrics.min_body_clearance_m)?;
    out.set_item("mean_body_clearance_m", metrics.mean_body_clearance_m)?;
    out.set_item("max_body_clearance_m", metrics.max_body_clearance_m)?;
    out.set_item(
        "skin_contact_to_nearest_aperture_m",
        metrics.skin_contact_to_nearest_aperture_m,
    )?;
    Ok(out)
}

pub(super) fn placement_context_skin_gap(context: &PlacementContext) -> f64 {
    context
        .therapy_points_m
        .iter()
        .chain(context.imaging_points_m.iter())
        .map(|point| {
            ((point.x_m - context.skin_contact_m.x_m).powi(2)
                + (point.y_m - context.skin_contact_m.y_m).powi(2)
                + (point.z_m - context.skin_contact_m.z_m).powi(2))
            .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

pub(super) fn point_axis(
    points: &[kwavers_therapy::therapy::theranostic_guidance::Point2],
    x_axis: bool,
) -> Array1<f64> {
    Array1::from(
        points
            .iter()
            .map(|point| if x_axis { point.x_m } else { point.y_m })
            .collect::<Vec<_>>(),
    )
}

pub(super) fn points3_to_array(points: &[Point3]) -> Array2<f64> {
    Array2::from_shape_fn((points.len(), 3), |(row, col)| match col {
        0 => points[row].x_m,
        1 => points[row].y_m,
        _ => points[row].z_m,
    })
}

pub(super) fn metric_dict<'py>(
    py: Python<'py>,
    metrics: &ReconstructionMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pearson", metrics.pearson)?;
    out.set_item("nrmse", metrics.nrmse)?;
    out.set_item("dice_equal_area", metrics.dice_equal_area)?;
    out.set_item("cnr", metrics.cnr)?;
    Ok(out)
}

pub(super) fn metric3d_dict<'py>(
    py: Python<'py>,
    metrics: &VolumeReconstructionMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("dice_equal_area", metrics.dice_equal_area)?;
    out.set_item("cnr", metrics.cnr)?;
    out.set_item("nrmse", metrics.nrmse)?;
    Ok(out)
}
