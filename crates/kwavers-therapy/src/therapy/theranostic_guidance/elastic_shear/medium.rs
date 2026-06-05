//! Elastic medium construction and receiver mask setup.

use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::forward::pstd::extensions::ElasticPstdMedium;
use ndarray::Array3;

use super::super::geometry::{DeviceLayout, Point2};
use super::super::medium::PreparedTheranosticSlice;
use super::geometry::index_point_m;

pub(super) fn elastic_medium(
    prepared: &PreparedTheranosticSlice,
    lesion_target: &ndarray::Array2<f64>,
    baseline_shear_speed_m_s: f64,
    lesion_fraction: f64,
) -> ElasticPstdMedium {
    let (nx, ny) = prepared.ct_hu.dim();
    let shape = (nx, ny, 1);
    let density = Array3::from_elem(shape, DENSITY_WATER_NOMINAL);
    let lame_mu = Array3::from_shape_fn(shape, |(ix, iy, _)| {
        if !prepared.body_mask[[ix, iy]] {
            return 0.0;
        }
        let factor = (1.0 + lesion_fraction * lesion_target[[ix, iy]]).max(0.10);
        let shear_speed = baseline_shear_speed_m_s * factor;
        DENSITY_WATER_NOMINAL * shear_speed * shear_speed
    });
    ElasticPstdMedium {
        lame_lambda: Array3::zeros(shape),
        lame_mu,
        density,
    }
}

pub(super) fn receiver_mask(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
) -> KwaversResult<Array3<bool>> {
    let (nx, ny) = prepared.ct_hu.dim();
    let body_indices = prepared
        .body_mask
        .indexed_iter()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect::<Vec<_>>();
    if body_indices.is_empty() {
        return Err(KwaversError::InvalidInput(
            "elastic shear reconstruction requires body support".to_owned(),
        ));
    }
    let mut mask = Array3::<bool>::from_elem((nx, ny, 1), false);
    for point in layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
    {
        let (ix, iy) = nearest_body_index(*point, prepared, &body_indices);
        mask[[ix, iy, 0]] = true;
    }
    Ok(mask)
}

fn nearest_body_index(
    point: Point2,
    prepared: &PreparedTheranosticSlice,
    body_indices: &[(usize, usize)],
) -> (usize, usize) {
    body_indices
        .iter()
        .copied()
        .min_by(|a, b| {
            let da = index_point_m(a.0, a.1, prepared.ct_hu.dim(), prepared.spacing_m);
            let db = index_point_m(b.0, b.1, prepared.ct_hu.dim(), prepared.spacing_m);
            use super::geometry::distance_sq;
            distance_sq(da, point).total_cmp(&distance_sq(db, point))
        })
        .unwrap_or((0, 0))
}
