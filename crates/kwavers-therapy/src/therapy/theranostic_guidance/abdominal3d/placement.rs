//! CT-derived 3-D abdominal array placement for visual verification.
//!
//! ## Algorithm
//!
//! Given a CT volume and a binary organ segmentation, this module computes the
//! 3-D geometry of a focused bowl transducer sitting on the exterior skin
//! surface of the abdomen, oriented toward the organ centroid.
//!
//! ### Theorem: Focused Bowl Placement
//!
//! Let F be the organ centroid in physical space [m] and S be the exterior skin
//! point that minimises ‖S − F‖ over all boundary voxels of the body mask.
//! Define the bowl axis d̂ = (F − S) / ‖F − S‖, focal depth `D = ‖F − S‖`,
//! and curvature radius `R = max(D / cos(θ_max), 60 mm)`. Any point on the
//! generated spherical cap of radius R centered at F with half-angle
//! θ ∈ [θ_cutout, θ_max] satisfies
//!
//! ```text
//! P(θ, φ) = F − R · [cos(θ)·d̂ + sin(θ)·(cos(φ)·ê₁ + sin(φ)·ê₂)]
//! ```
//!
//! where (ê₁, ê₂) form an orthonormal frame perpendicular to d̂. The clinical
//! layer chooses F, S, R, and angular bounds; source-domain `BowlTransducer`
//! owns equal-area spherical-cap sampling, angular validation, normals, and
//! weights.
//!
//! ### Skin Point Selection
//!
//! The nearest exterior skin point to the organ centroid minimises the
//! insertion depth and maximises the acoustic transmission window — the
//! standard clinical placement principle for all focused ultrasound devices.
//! Boundary voxels are identified by the 6-connected neighbour test
//! (`is_boundary_3d`).
//!
//! ### Parameter Choices
//!
//! - θ_cutout = 0.175 rad (≈ 10°): central cutout for a co-axial imaging probe.
//! - θ_max = 0.960 rad (≈ 55°): aperture half-angle giving an F-number of
//!   approximately 0.87 at the stated focal length.
//! - R = ‖F − S‖ / cos(θ_max): places the rim (θ_max) exactly at skin level
//!   so all elements are outside the body.  Minimum enforced at 60 mm.
//!
//! ## References
//!
//! - Parsons J E et al. (2006) Cost-effective assembly of a basic fiber-optic
//!   hydrophone for measurement of high-amplitude therapeutic ultrasound fields.
//!   J. Acoust. Soc. Am. 119(3): 1432–1440.
//! - Hynynen K & Jones R M (2016) Image-guided ultrasound phased arrays are a
//!   disruptive technology for non-invasive therapy. Phys. Med. Biol. 61(17):
//!   R206–R248.

use ndarray::Array3;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::geometry::{active_bounds_3d, Point3};
use super::super::nonlinear3d::volume::centroid_float;
use super::bowl::{bowl_elements, BOWL_THETA_MAX_RAD};
use super::helpers::{
    distance_3d, exterior_air_mask, exterior_body_surface_points, index_to_point,
    keep_largest_connected_component_3d, nearest_exterior_skin_point, surface_points_3d,
};
use super::types::AbdominalArrayPlacement3D;

/// Compute 3-D focused bowl placement on the abdominal skin.
///
/// # Parameters
///
/// - `ct_hu`: 3-D CT volume in Hounsfield units, shape `[NX, NY, NZ]`.
/// - `label`: Binary organ segmentation (any value > 0 = organ), same shape.
/// - `spacing_mm`: Voxel spacing in mm, `[sx, sy, sz]`.
/// - `element_count`: Number of bowl elements to distribute on the cap.
/// - `surface_stride`: Voxel stride for surface sampling (higher → fewer points).
/// - `body_hu_threshold`: HU threshold for body mask (default −400 HU).
/// - `anatomy_label`: Human-readable anatomy name for labelling.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` when the CT mask or organ mask is empty,
/// or when voxel spacing is non-positive.
pub fn plan_abdominal_array_placement(
    ct_hu: &Array3<f64>,
    label: &Array3<i16>,
    spacing_mm: [f64; 3],
    element_count: usize,
    surface_stride: usize,
    body_hu_threshold: f64,
    anatomy_label: String,
) -> KwaversResult<AbdominalArrayPlacement3D> {
    if element_count == 0 {
        return Err(KwaversError::InvalidInput(
            "abdominal array placement requires at least one element".to_owned(),
        ));
    }
    if spacing_mm.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(KwaversError::InvalidInput(
            "abdominal array placement requires positive finite CT spacing".to_owned(),
        ));
    }
    if ct_hu.dim() != label.dim() {
        return Err(KwaversError::InvalidInput(format!(
            "CT shape {:?} does not match segmentation shape {:?}",
            ct_hu.dim(),
            label.dim()
        )));
    }

    // Raw HU-thresholded mask includes patient body, imaging table, IV bags,
    // EKG leads, positioning cushions — anything above the soft-tissue floor.
    let raw_body_mask = ct_hu.mapv(|hu| hu.is_finite() && hu >= body_hu_threshold);
    // Keep only the largest 6-connected component: by definition the patient.
    // This drops the CT table and any other disjoint object from the body
    // surface cloud and from the nearest-skin search. See
    // [`keep_largest_connected_component_3d`] for the rationale.
    let body_mask = keep_largest_connected_component_3d(&raw_body_mask);
    let organ_mask: Array3<bool> = label.mapv(|l| l > 0);

    if !body_mask.iter().any(|&b| b) {
        return Err(KwaversError::InvalidInput(
            "CT body mask is empty — check body_hu_threshold".to_owned(),
        ));
    }
    if !organ_mask.iter().any(|&b| b) {
        return Err(KwaversError::InvalidInput(
            "organ segmentation mask is empty".to_owned(),
        ));
    }

    let spacing_m = [
        spacing_mm[0] * 1.0e-3,
        spacing_mm[1] * 1.0e-3,
        spacing_mm[2] * 1.0e-3,
    ];

    // Body centroid as coordinate origin (in voxel index space).
    let body_bounds = active_bounds_3d(&body_mask)?;
    let body_center_index = centroid_float(&body_mask, None).unwrap_or([
        0.5 * (body_bounds.x0 + body_bounds.x1) as f64,
        0.5 * (body_bounds.y0 + body_bounds.y1) as f64,
        0.5 * (body_bounds.z0 + body_bounds.z1) as f64,
    ]);

    // Organ centroid → geometric focus target.
    let organ_centroid = centroid_float(&organ_mask, None).ok_or_else(|| {
        KwaversError::InvalidInput("organ centroid could not be computed".to_owned())
    })?;
    let focus_m = index_to_point(organ_centroid, spacing_m, body_center_index);

    // Flood-fill the exterior air once; reuse for skin contact AND body surface.
    // This correctly excludes internal tissue interfaces (intestinal gas walls,
    // retroperitoneal fat boundaries, vessel lumens) that is_boundary_3d would
    // incorrectly treat as skin candidates.
    let exterior_air = exterior_air_mask(&body_mask);

    // Nearest exterior skin point to the organ centroid (uses pre-computed exterior).
    let skin_contact_m = nearest_exterior_skin_point(
        &body_mask,
        &exterior_air,
        spacing_m,
        body_center_index,
        focus_m,
    )?;

    // Bowl geometry: sphere of radius R (focal length) centred at focus_m.
    //
    // The constraint for all cap elements to lie OUTSIDE the body is:
    //   R · cos(θ_max) ≥ focal_depth
    // ⟹ R_min = focal_depth / cos(θ_max).
    //
    // Derivation: element axial height above focus F = R·cos(θ), and the
    // skin contact S is at axial height focal_depth from F. The worst-case
    // element (largest θ = θ_max) must be at or above the skin:
    //   R·cos(θ_max) ≥ focal_depth.
    // Setting R = focal_depth / cos(θ_max) places the rim element exactly at
    // skin level; the vertex (θ → θ_cutout ≈ 0) is at height R ≫ focal_depth
    // (well outside the body). A minimum clinical bowl radius of 60 mm is
    // enforced for mechanical feasibility.
    let focal_depth_m = distance_3d(skin_contact_m, focus_m);
    let transducer_radius_m = (focal_depth_m / BOWL_THETA_MAX_RAD.cos()).max(0.060);

    let stride = surface_stride.max(1);

    // Body surface: only true exterior skin using the flood-fill mask.
    let body_surface_points_m = exterior_body_surface_points(
        &body_mask,
        &exterior_air,
        spacing_m,
        body_center_index,
        stride,
    );

    // Organ surface.
    let organ_surface_points_m =
        surface_points_3d(&organ_mask, spacing_m, body_center_index, stride.max(2));

    // Bowl element positions (outside the body, on the skin surface cap).
    let therapy_elements_m =
        bowl_elements(element_count, skin_contact_m, focus_m, transducer_radius_m)?;

    // Beam visualisation: up to 64 beams from elements to focus.
    let beam_count = 64_usize.min(element_count).max(1);
    let beam_step = (element_count / beam_count).max(1);
    let beam_start_points_m: Vec<Point3> = therapy_elements_m
        .iter()
        .step_by(beam_step)
        .take(beam_count)
        .copied()
        .collect();
    let beam_end_points_m: Vec<Point3> = beam_start_points_m.iter().map(|_| focus_m).collect();

    Ok(AbdominalArrayPlacement3D {
        body_surface_points_m,
        organ_surface_points_m,
        therapy_elements_m,
        beam_start_points_m,
        beam_end_points_m,
        focus_m,
        skin_contact_m,
        transducer_radius_m,
        anatomy_label,
    })
}
