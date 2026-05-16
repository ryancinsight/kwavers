//! `plan_brain_helmet_placement_from_ritk_ct` pyfunction.
//!
//! When `ct_nifti_path` does not exist on disk the function falls back to a
//! built-in synthetic brain CT phantom with clinically realistic dimensions
//! (see `kwavers::clinical::therapy::theranostic_guidance::synthetic`).

use kwavers::clinical::therapy::theranostic_guidance::{
    plan_brain_helmet_placement,
    synthetic::synthetic_brain_phantom,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::helpers::{kwavers_to_py, points3_to_array};
use crate::ritk_image::load_ritk_nifti;

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    element_count = 1024,
    surface_stride = 6,
    body_hu_threshold = -350.0,
    skull_hu_threshold = 300.0,
    target_fraction_xyz = None,
    scene_radius_m = None
))]
pub fn plan_brain_helmet_placement_from_ritk_ct<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    element_count: usize,
    surface_stride: usize,
    body_hu_threshold: f64,
    skull_hu_threshold: f64,
    target_fraction_xyz: Option<(f64, f64, f64)>,
    scene_radius_m: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ct_path = Path::new(ct_nifti_path);
    let synthetic = !ct_path.exists();

    let (ct, spacing_mm) = if synthetic {
        synthetic_brain_phantom()
    } else {
        let (mut ct, spacing_mm) = load_ritk_nifti(ct_path)?;
        ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
        (ct, spacing_mm)
    };

    let placement = py
        .detach(|| {
            plan_brain_helmet_placement(
                &ct,
                spacing_mm,
                element_count,
                surface_stride,
                body_hu_threshold,
                skull_hu_threshold,
                target_fraction_xyz.map(|(x, y, z)| [x, y, z]),
                scene_radius_m,
            )
        })
        .map_err(kwavers_to_py)?;

    let out = PyDict::new(py);
    out.set_item(
        "head_surface_points_m",
        points3_to_array(&placement.head_surface_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "skull_surface_points_m",
        points3_to_array(&placement.skull_surface_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "therapy_elements_m",
        points3_to_array(&placement.therapy_elements_m).into_pyarray(py),
    )?;
    out.set_item(
        "beam_start_points_m",
        points3_to_array(&placement.beam_start_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "beam_end_points_m",
        points3_to_array(&placement.beam_end_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "skull_intersections_m",
        points3_to_array(&placement.skull_intersections_m).into_pyarray(py),
    )?;
    out.set_item(
        "focus_m",
        (
            placement.focus_m.x_m,
            placement.focus_m.y_m,
            placement.focus_m.z_m,
        ),
    )?;
    out.set_item("helmet_radius_m", placement.helmet_radius_m)?;
    out.set_item("intersection_fraction", placement.intersection_fraction)?;
    out.set_item("element_count", element_count)?;
    out.set_item("surface_stride", surface_stride)?;
    out.set_item("body_hu_threshold", body_hu_threshold)?;
    out.set_item("skull_hu_threshold", skull_hu_threshold)?;
    if let Some((x, y, z)) = target_fraction_xyz {
        out.set_item("target_fraction_xyz", (x, y, z))?;
    }
    if let Some(radius) = scene_radius_m {
        out.set_item("scene_radius_m", radius)?;
    }
    out.set_item(
        "geometry_model",
        "ct_derived_calvarium_1024_element_helmet_with_skull_intersections",
    )?;
    out.set_item("synthetic_phantom", synthetic)?;
    Ok(out)
}
