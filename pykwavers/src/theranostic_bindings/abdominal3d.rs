//! `plan_abdominal_array_placement_from_ritk_ct` pyfunction.
//!
//! When `ct_nifti_path` does not exist on disk the function falls back to a
//! built-in synthetic abdominal CT phantom with clinically realistic dimensions
//! (see `kwavers::clinical::therapy::theranostic_guidance::synthetic`).

use kwavers::clinical::therapy::theranostic_guidance::{
    plan_abdominal_array_placement,
    synthetic::{synthetic_abdominal_kidney_phantom, synthetic_abdominal_liver_phantom},
};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::helpers::{kwavers_to_py, labels_from_volume, points3_to_array};
use crate::ritk_image::load_ritk_nifti;

/// Plan 3-D HistoSonics-like focused bowl placement on the abdominal skin.
///
/// Loads the CT and label NIfTI volumes via ritk, computes the body surface,
/// organ surface, and bowl element positions using the Fibonacci golden-spiral
/// method, and returns all geometry needed for a 3-D visualisation in which the
/// transducer bowl sits visibly on the skin surface outside the patient body.
///
/// When `ct_nifti_path` does not exist on disk a synthetic phantom is used
/// automatically (no error is raised).  The synthetic phantom has clinically
/// representative dimensions and produces a valid bowl placement.
///
/// # Returns
///
/// A `dict` with keys:
///
/// - `body_surface_points_m`: `ndarray (N, 3)` — exterior skin surface [m].
/// - `organ_surface_points_m`: `ndarray (M, 3)` — organ surface [m].
/// - `therapy_elements_m`: `ndarray (K, 3)` — bowl element positions [m].
/// - `beam_start_points_m`: `ndarray (B, 3)` — ray starts (elements) [m].
/// - `beam_end_points_m`: `ndarray (B, 3)` — ray ends (focus) [m].
/// - `focus_m`: `(fx, fy, fz)` — organ centroid [m].
/// - `skin_contact_m`: `(sx, sy, sz)` — nearest skin contact point [m].
/// - `transducer_radius_m`: `float` — bowl radius / focal length [m].
/// - `anatomy_label`: `str` — human-readable anatomy name.
/// - `geometry_model`: `str` — constant label for metrics tracking.
/// - `element_count`: `int` — number of bowl elements placed.
/// - `surface_stride`: `int` — stride used for surface sampling.
/// - `synthetic_phantom`: `bool` — `True` when real CT was absent.
#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    seg_nifti_path,
    anatomy_label = "abdomen",
    element_count = 256,
    surface_stride = 6,
    body_hu_threshold = -400.0,
))]
pub fn plan_abdominal_array_placement_from_ritk_ct<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    seg_nifti_path: &str,
    anatomy_label: &str,
    element_count: usize,
    surface_stride: usize,
    body_hu_threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // Determine whether to load real NIfTI files or use synthetic phantom.
    let ct_path = Path::new(ct_nifti_path);
    let synthetic = !ct_path.exists();

    let (ct, label, spacing_mm) = if synthetic {
        // Anatomy-specific synthetic phantom.
        match anatomy_label {
            "kidney" => synthetic_abdominal_kidney_phantom(),
            _ => synthetic_abdominal_liver_phantom(),
        }
    } else {
        let (mut ct, spacing_mm) = load_ritk_nifti(ct_path)?;
        ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
        let (label_f64, _) = load_ritk_nifti(Path::new(seg_nifti_path))?;
        let label = labels_from_volume(label_f64);
        (ct, label, spacing_mm)
    };

    let placement = py
        .detach(|| {
            plan_abdominal_array_placement(
                &ct,
                &label,
                spacing_mm,
                element_count,
                surface_stride,
                body_hu_threshold,
                anatomy_label.to_owned(),
            )
        })
        .map_err(kwavers_to_py)?;

    let out = PyDict::new(py);
    out.set_item(
        "body_surface_points_m",
        points3_to_array(&placement.body_surface_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "organ_surface_points_m",
        points3_to_array(&placement.organ_surface_points_m).into_pyarray(py),
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
        "focus_m",
        (
            placement.focus_m.x_m,
            placement.focus_m.y_m,
            placement.focus_m.z_m,
        ),
    )?;
    out.set_item(
        "skin_contact_m",
        (
            placement.skin_contact_m.x_m,
            placement.skin_contact_m.y_m,
            placement.skin_contact_m.z_m,
        ),
    )?;
    out.set_item("transducer_radius_m", placement.transducer_radius_m)?;
    out.set_item("anatomy_label", &placement.anatomy_label)?;
    out.set_item(
        "geometry_model",
        "ct_derived_abdominal_256_element_focused_bowl_on_skin",
    )?;
    out.set_item("element_count", element_count)?;
    out.set_item("surface_stride", surface_stride)?;
    out.set_item("body_hu_threshold", body_hu_threshold)?;
    out.set_item("synthetic_phantom", synthetic)?;
    Ok(out)
}
