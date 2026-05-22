//! Brain CT preprocessing for theranostic inverse.
//!
//! Converts a 2-D axial CT slice (HU) to the acoustic property maps and
//! anatomical masks required by the transcranial FWI and standing-wave
//! suppression pipelines.
//!
//! # HU-to-property mapping (skull bone)
//!
//! ```text
//! φ  = HU / 1000   (porosity proxy, clamped to [0, 1])
//! c  = 1500(1−φ) + 2900φ   [m/s]   (Marsac et al. 2017)
//! α  = α_soft(1−φ) + 70φ   [Np/m/MHz]
//! ```
//!
//! Soft tissue: `c ≈ 1510 + 55·HU_norm` (empirical, HU in [−20, 120]).
//! Attenuation: 0.5 dB/cm/MHz → `α_soft = 0.5 × 100 × ln10/20 Np/m/MHz`.

use crate::core::constants::acoustic_parameters::SOUND_SPEED_SKULL;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

use super::super::scene::target_index_from_mask_fraction_2d;
use super::{validate_masks, AnatomyKind, PreparedTheranosticSlice};

#[derive(Clone, Copy, Debug)]
pub enum BrainTargetSelection {
    OrganCentroid,
    SliceFraction([f64; 2]),
    ResampledIndex([f64; 2]),
}

pub fn prepare_brain_slice(
    ct_hu: Array2<f64>,
    spacing_m: f64,
    source_slice_index: usize,
    target_selection: BrainTargetSelection,
) -> KwaversResult<PreparedTheranosticSlice> {
    let (nx, ny) = ct_hu.dim();
    let mut label = Array2::<i16>::zeros((nx, ny));
    let mut sound_speed = Array2::<f64>::from_elem((nx, ny), 1480.0);
    let mut attenuation = Array2::<f64>::from_elem((nx, ny), soft_attenuation());
    let mut body = Array2::<bool>::from_elem((nx, ny), false);
    let mut organ = Array2::<bool>::from_elem((nx, ny), false);
    let centroid = head_centroid(&ct_hu);
    let radius = 0.34 * nx.min(ny) as f64;
    for ix in 0..nx {
        for iy in 0..ny {
            let hu = ct_hu[[ix, iy]];
            let in_body = hu > -300.0;
            body[[ix, iy]] = in_body;
            if hu >= 300.0 {
                let phi = (hu / 1000.0).clamp(0.0, 1.0);
                sound_speed[[ix, iy]] = SOUND_SPEED_WATER_SIM * (1.0 - phi) + SOUND_SPEED_SKULL * phi;
                attenuation[[ix, iy]] = soft_attenuation() * (1.0 - phi) + 70.0 * phi;
                label[[ix, iy]] = 4;
            } else if in_body {
                sound_speed[[ix, iy]] = brain_speed(hu);
                label[[ix, iy]] = 1;
            }
            let dx = ix as f64 - centroid.0;
            let dy = iy as f64 - centroid.1;
            let central = (dx * dx + dy * dy).sqrt() <= radius;
            organ[[ix, iy]] = central && (-40.0..=140.0).contains(&hu);
        }
    }
    let target = synthetic_deep_target(&organ, spacing_m, target_selection)?;
    validate_masks(&body, &target)?;
    Ok(PreparedTheranosticSlice {
        anatomy: AnatomyKind::Brain,
        ct_hu,
        label,
        sound_speed_m_s: sound_speed,
        attenuation_np_per_m_mhz: attenuation,
        body_mask: body,
        organ_mask: organ,
        target_mask: target,
        spacing_m,
        source_slice_index,
        source_dimensions: [nx, ny],
        source_spacing_m: [spacing_m, spacing_m],
        crop_bounds_index: [0, nx - 1, 0, ny - 1],
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Weighted centroid of all grid positions produced by `positions`.
///
/// Returns the geometric centre of the grid `(nx, ny)` when no positions are
/// active, avoiding a degenerate output for empty masks.
fn centroid_from_active(
    positions: impl Iterator<Item = (usize, usize)>,
    (nx, ny): (usize, usize),
) -> (f64, f64) {
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut n = 0.0_f64;
    for (ix, iy) in positions {
        sx += ix as f64;
        sy += iy as f64;
        n += 1.0;
    }
    if n > 0.0 {
        (sx / n, sy / n)
    } else {
        ((nx - 1) as f64 * 0.5, (ny - 1) as f64 * 0.5)
    }
}

fn head_centroid(ct_hu: &Array2<f64>) -> (f64, f64) {
    centroid_from_active(
        ct_hu
            .indexed_iter()
            .filter_map(|((ix, iy), &hu)| (hu > -300.0).then_some((ix, iy))),
        ct_hu.dim(),
    )
}

fn head_centroid_bool(mask: &Array2<bool>) -> (f64, f64) {
    centroid_from_active(
        mask.indexed_iter()
            .filter_map(|((ix, iy), &active)| active.then_some((ix, iy))),
        mask.dim(),
    )
}

/// Synthetic ellipsoidal deep target within the organ region.
///
/// Semi-axes: rx = 6 mm, ry = 8 mm (physical), centred on the organ centroid.
/// Used when a real segmentation is unavailable.
fn synthetic_deep_target(
    organ: &Array2<bool>,
    spacing_m: f64,
    target_selection: BrainTargetSelection,
) -> KwaversResult<Array2<bool>> {
    let (nx, ny) = organ.dim();
    let center = match target_selection {
        BrainTargetSelection::OrganCentroid => head_centroid_bool(organ),
        BrainTargetSelection::SliceFraction(fraction) => {
            let (ix, iy) = target_index_from_mask_fraction_2d(organ, fraction)?;
            (ix as f64, iy as f64)
        }
        BrainTargetSelection::ResampledIndex(index) => {
            validate_target_center(index, nx, ny)?;
            (index[0], index[1])
        }
    };
    let rx = 6.0e-3 / spacing_m;
    let ry = 8.0e-3 / spacing_m;
    Ok(Array2::from_shape_fn((nx, ny), |(ix, iy)| {
        organ[[ix, iy]]
            && ((ix as f64 - center.0) / rx).powi(2) + ((iy as f64 - center.1) / ry).powi(2) <= 1.0
    }))
}

fn validate_target_center(center: [f64; 2], nx: usize, ny: usize) -> KwaversResult<()> {
    if center
        .iter()
        .any(|coordinate| !coordinate.is_finite() || *coordinate < 0.0)
        || center[0] > (nx - 1) as f64
        || center[1] > (ny - 1) as f64
    {
        return Err(KwaversError::InvalidInput(format!(
            "brain target center {:?} lies outside resampled slice dimensions {nx}x{ny}",
            center
        )));
    }
    Ok(())
}

/// Linear HU-to-sound-speed for brain soft tissue.
///
/// Empirical: 1510 m/s at HU = −20 (CSF), 1565 m/s at HU = 120 (white matter).
fn brain_speed(hu: f64) -> f64 {
    1510.0 + 55.0 * ((hu + 20.0) / 140.0).clamp(0.0, 1.0)
}

/// Soft-tissue attenuation in Np/m/MHz.
///
/// Converts 0.5 dB/cm/MHz: `α = 0.5 × 100 cm/m × ln(10)/20`.
fn soft_attenuation() -> f64 {
    0.5 * 100.0 * std::f64::consts::LN_10 / 20.0
}
