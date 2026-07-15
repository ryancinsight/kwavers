//! CT and segmentation preprocessing for theranostic inverse slices.
//!
//! Anatomy-specific pipelines are isolated to sub-modules enforcing SRP:
//! - [`brain`]    — transcranial / skull imaging preprocessing
//! - [`abdominal`] — liver / kidney FUS preprocessing
//!
//! Shared infrastructure (the output type, masking helpers, interpolation)
//! lives here as the single canonical definition.

mod abdominal;
mod brain;

pub use abdominal::prepare_abdominal_slice;
pub(crate) use abdominal::{largest_connected_target_component, largest_target_slice};
pub use brain::{prepare_brain_slice, BrainTargetSelection};

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::numerics::operators::interpolation::bilinear_index_space;
use kwavers_solver::inverse::same_aperture::C_REF_M_S;
use leto::Array2;

use super::config::AnatomyKind;

// ── Output type ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PreparedTheranosticSlice {
    pub anatomy: AnatomyKind,
    pub ct_hu: Array2<f64>,
    pub label: Array2<i16>,
    pub sound_speed_m_s: Array2<f64>,
    pub attenuation_np_per_m_mhz: Array2<f64>,
    pub body_mask: Array2<bool>,
    pub organ_mask: Array2<bool>,
    pub target_mask: Array2<bool>,
    pub spacing_m: f64,
    pub source_slice_index: usize,
    pub source_dimensions: [usize; 2],
    pub source_spacing_m: [f64; 2],
    pub crop_bounds_index: [usize; 4],
}

// ── Shared analysis ───────────────────────────────────────────────────────────

/// Normalised sound-speed contrast within the body mask.
///
/// Returns `(c(x) − c_ref) / c_ref` where `c_ref` is the median sound speed
/// over the body mask, clamped to the global tissue reference if the mask is
/// empty.
pub fn target_contrast(prepared: &PreparedTheranosticSlice) -> Array2<f64> {
    let reference =
        median_in_mask(&prepared.sound_speed_m_s, &prepared.body_mask).unwrap_or(C_REF_M_S);
    let mut out = Array2::<f64>::zeros(prepared.sound_speed_m_s.shape());
    for ((idx, value), active) in prepared
        .sound_speed_m_s
        .indexed_iter()
        .zip(prepared.body_mask.iter())
    {
        if *active {
            out[idx] = (*value - reference) / C_REF_M_S;
        }
    }
    out
}

// ── Shared private helpers ────────────────────────────────────────────────────

/// Reject slices where body or target support is too thin to be physically
/// meaningful (minimum 16 body cells, 4 target cells).
pub(crate) fn validate_masks(body: &Array2<bool>, target: &Array2<bool>) -> KwaversResult<()> {
    let body_count = body.iter().filter(|v| **v).count();
    let target_count = target.iter().filter(|v| **v).count();
    if body_count < 16 || target_count < 4 {
        return Err(KwaversError::InvalidInput(format!(
            "insufficient active support: body={body_count}, target={target_count}"
        )));
    }
    Ok(())
}

/// Median of `values` restricted to cells where `mask` is `true`, or `None`
/// when the mask is empty.
pub(super) fn median_in_mask(values: &Array2<f64>, mask: &Array2<bool>) -> Option<f64> {
    let mut selected = values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return None;
    }
    selected.sort_by(f64::total_cmp);
    Some(selected[selected.len() / 2])
}

/// Resample a continuous 2-D field to a square `size × size` grid via bilinear
/// interpolation.
///
/// When `size == 1` the single output cell is the bilinear sample at the centre
/// of the input grid, avoiding division-by-zero in the index mapping.  Callers
/// should enforce `size >= 2` at their API boundary (e.g. `grid_size` in
/// [`prepare_abdominal_slice`]) to produce a meaningful solver grid.
pub(super) fn resample(input: &Array2<f64>, size: usize) -> Array2<f64> {
    let [nx, ny] = input.shape();
    if size == 1 {
        let cx = (nx - 1) as f64 * 0.5;
        let cy = (ny - 1) as f64 * 0.5;
        return Array2::from_elem((1, 1), bilinear_index_space(input, cx, cy));
    }
    let scale_x = (nx - 1) as f64 / (size - 1) as f64;
    let scale_y = (ny - 1) as f64 / (size - 1) as f64;
    Array2::from_shape_fn((size, size), |[ix, iy]| {
        bilinear_index_space(input, ix as f64 * scale_x, iy as f64 * scale_y)
    })
}

/// Resample an integer label map to `size × size` by max-pooling, preserving
/// the highest label value in each output cell.
pub(super) fn resample_labels_max(input: &Array2<i16>, size: usize) -> Array2<i16> {
    let [nx, ny] = input.shape();
    Array2::from_shape_fn((size, size), |[ix, iy]| {
        let x0 = (ix * nx) / size;
        let x1 = (((ix + 1) * nx).saturating_sub(1)) / size;
        let y0 = (iy * ny) / size;
        let y1 = (((iy + 1) * ny).saturating_sub(1)) / size;
        let mut label = 0i16;
        for x in x0..=x1.min(nx - 1) {
            for y in y0..=y1.min(ny - 1) {
                label = label.max(input[[x, y]]);
            }
        }
        label
    })
}
