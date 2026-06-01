//! CT slice preparation and CT-to-acoustic property mapping.

use crate::core::constants::acoustic_parameters::{
    NP_TO_DB, SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ,
};
use crate::core::constants::ct_acoustics::{
    HU_BONE_THRESHOLD, HU_BRAIN_BODY_THRESHOLD, SOUND_SPEED_SOFT_TISSUE_MAX,
};
use crate::core::constants::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, SOUND_SPEED_WATER_37C};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::numerics::operators::interpolation::bilinear_index_space;
use ndarray::{s, Array2, Array3};

use super::config::{SOUND_SPEED_SKULL, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM};

/// Soft-tissue attenuation in Np/(m·MHz) derived from `ACOUSTIC_ABSORPTION_TISSUE`
/// (0.5 dB/(cm·MHz)): α = 0.5 × 100 / NP_TO_DB.
pub(super) const SOFT_TISSUE_ATTENUATION_NP_PER_M_MHZ: f64 =
    ACOUSTIC_ABSORPTION_TISSUE * 100.0 / NP_TO_DB;

/// CT slice resampled to the square transcranial UST inversion domain.
#[derive(Clone, Debug)]
pub struct CtResampledSlice {
    pub hu: Array2<f64>,
    pub spacing_m: f64,
    pub slice_offset_m: f64,
    pub source_slice_index: usize,
    pub source_dimensions: [usize; 2],
    pub source_spacing_m: [f64; 2],
    pub crop_bounds_index: [usize; 4],
}

/// Acoustic fields derived from one CT slice.
#[derive(Clone, Debug)]
pub struct AcousticSlice {
    pub ct_hu: Array2<f64>,
    pub sound_speed_m_s: Array2<f64>,
    pub initial_sound_speed_m_s: Array2<f64>,
    pub attenuation_np_per_m_mhz: Array2<f64>,
    pub brain_mask: Array2<bool>,
    pub skull_mask: Array2<bool>,
    pub spacing_m: f64,
    pub slice_offset_m: f64,
}

/// Select the axial slice with the largest non-air head support.
pub fn select_head_slice(volume_hu: &Array3<f64>) -> KwaversResult<usize> {
    let (_, _, nz) = volume_hu.dim();
    if nz == 0 {
        return Err(KwaversError::InvalidInput(
            "CT volume must contain at least one slice".to_owned(),
        ));
    }

    let mut best = None;
    for z in 0..nz {
        let slice = volume_hu.slice(s![.., .., z]);
        let head_voxels = slice.iter().filter(|v| **v > HU_BRAIN_BODY_THRESHOLD).count();
        let skull_voxels = slice.iter().filter(|v| **v > HU_BONE_THRESHOLD).count();
        let score = head_voxels + 4 * skull_voxels;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((z, score));
        }
    }

    best.map(|(z, _)| z).ok_or_else(|| {
        KwaversError::InvalidInput("CT volume did not contain a head slice".to_owned())
    })
}

/// Crop the head support from a CT slice and resample it onto a square grid.
pub fn resample_head_slice(
    volume_hu: &Array3<f64>,
    spacing_mm: [f64; 3],
    slice_index: usize,
    grid_size: usize,
) -> KwaversResult<CtResampledSlice> {
    if grid_size < 16 {
        return Err(KwaversError::InvalidInput(
            "grid_size must be at least 16".to_owned(),
        ));
    }
    if spacing_mm.iter().any(|spacing| *spacing <= 0.0) {
        return Err(KwaversError::InvalidInput(
            "CT spacing must be positive in x, y, and z".to_owned(),
        ));
    }

    let (nx, ny, nz) = volume_hu.dim();
    if slice_index >= nz {
        return Err(KwaversError::InvalidInput(format!(
            "slice_index {slice_index} out of bounds for {nz} CT slices"
        )));
    }

    let slice = volume_hu.slice(s![.., .., slice_index]).to_owned();
    let bbox = head_bbox(&slice)?;
    let margin_x = ((bbox.1 - bbox.0 + 1) as f64 * 0.08).ceil() as usize;
    let margin_y = ((bbox.3 - bbox.2 + 1) as f64 * 0.08).ceil() as usize;
    let x0 = bbox.0.saturating_sub(margin_x);
    let x1 = (bbox.1 + margin_x).min(nx - 1);
    let y0 = bbox.2.saturating_sub(margin_y);
    let y1 = (bbox.3 + margin_y).min(ny - 1);

    let mut out = Array2::<f64>::zeros((grid_size, grid_size));
    let sx = if grid_size > 1 {
        (x1 - x0) as f64 / (grid_size - 1) as f64
    } else {
        0.0
    };
    let sy = if grid_size > 1 {
        (y1 - y0) as f64 / (grid_size - 1) as f64
    } else {
        0.0
    };

    for ix in 0..grid_size {
        let x = x0 as f64 + ix as f64 * sx;
        for iy in 0..grid_size {
            let y = y0 as f64 + iy as f64 * sy;
            out[[ix, iy]] = bilinear_index_space(&slice, x, y);
        }
    }

    let extent_x_m = (x1 - x0 + 1) as f64 * spacing_mm[0] * 1.0e-3;
    let extent_y_m = (y1 - y0 + 1) as f64 * spacing_mm[1] * 1.0e-3;
    Ok(CtResampledSlice {
        hu: out,
        spacing_m: extent_x_m.max(extent_y_m) / grid_size as f64,
        slice_offset_m: (slice_index as f64 - (nz - 1) as f64 * 0.5) * spacing_mm[2] * 1.0e-3,
        source_slice_index: slice_index,
        source_dimensions: [nx, ny],
        source_spacing_m: [spacing_mm[0] * 1.0e-3, spacing_mm[1] * 1.0e-3],
        crop_bounds_index: [x0, x1, y0, y1],
    })
}

impl AcousticSlice {
    /// Convert resampled CT HU values to acoustic speed and inversion masks.
    pub fn from_ct_hu(ct_hu: Array2<f64>, spacing_m: f64) -> KwaversResult<Self> {
        Self::from_ct_hu_at_offset(ct_hu, spacing_m, 0.0)
    }

    /// Convert resampled CT HU values at a fixed axial offset into acoustic fields.
    pub fn from_ct_hu_at_offset(
        ct_hu: Array2<f64>,
        spacing_m: f64,
        slice_offset_m: f64,
    ) -> KwaversResult<Self> {
        if !spacing_m.is_finite() || spacing_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "spacing_m must be finite and positive".to_owned(),
            ));
        }
        if !slice_offset_m.is_finite() {
            return Err(KwaversError::InvalidInput(
                "slice_offset_m must be finite".to_owned(),
            ));
        }
        let (nx, ny) = ct_hu.dim();
        if nx < 16 || ny < 16 {
            return Err(KwaversError::InvalidInput(
                "CT slice must be at least 16 x 16".to_owned(),
            ));
        }

        let mut sound_speed = Array2::<f64>::zeros((nx, ny));
        let mut initial = Array2::<f64>::from_elem((nx, ny), SOUND_SPEED_WATER_SIM);
        let mut attenuation = Array2::<f64>::zeros((nx, ny));
        let mut brain_mask = Array2::<bool>::from_elem((nx, ny), false);
        let mut skull_mask = Array2::<bool>::from_elem((nx, ny), false);
        let center =
            head_centroid(&ct_hu).unwrap_or(((nx - 1) as f64 / 2.0, (ny - 1) as f64 / 2.0));
        let radius = 0.38 * nx.min(ny) as f64;

        for ix in 0..nx {
            for iy in 0..ny {
                let hu = ct_hu[[ix, iy]];
                let skull = hu >= HU_BONE_THRESHOLD;
                skull_mask[[ix, iy]] = skull;
                sound_speed[[ix, iy]] = if skull {
                    let phi = (hu / 1000.0).clamp(0.0, 1.0);
                    attenuation[[ix, iy]] = SOFT_TISSUE_ATTENUATION_NP_PER_M_MHZ * (1.0 - phi)
                        + SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ * phi;
                    SOUND_SPEED_WATER_SIM * (1.0 - phi) + SOUND_SPEED_SKULL * phi
                } else if hu > HU_BRAIN_BODY_THRESHOLD {
                    attenuation[[ix, iy]] = SOFT_TISSUE_ATTENUATION_NP_PER_M_MHZ;
                    soft_tissue_speed(hu)
                } else {
                    SOUND_SPEED_WATER_SIM
                };

                let dx = ix as f64 - center.0;
                let dy = iy as f64 - center.1;
                let central = (dx * dx + dy * dy).sqrt() <= radius;
                let brain_hu = (-20.0..=120.0).contains(&hu);
                brain_mask[[ix, iy]] = central && brain_hu && !skull;
                initial[[ix, iy]] = if brain_mask[[ix, iy]] {
                    SOUND_SPEED_TISSUE
                } else {
                    sound_speed[[ix, iy]]
                };
            }
        }

        let active = brain_mask.iter().filter(|v| **v).count();
        if active < 8 {
            return Err(KwaversError::InvalidInput(
                "CT-derived brain mask contains fewer than 8 voxels".to_owned(),
            ));
        }

        Ok(Self {
            ct_hu,
            sound_speed_m_s: sound_speed,
            initial_sound_speed_m_s: initial,
            attenuation_np_per_m_mhz: attenuation,
            brain_mask,
            skull_mask,
            spacing_m,
            slice_offset_m,
        })
    }
}

fn head_bbox(slice: &Array2<f64>) -> KwaversResult<(usize, usize, usize, usize)> {
    let (nx, ny) = slice.dim();
    let mut bbox: Option<(usize, usize, usize, usize)> = None;
    for ix in 0..nx {
        for iy in 0..ny {
            if slice[[ix, iy]] > HU_BRAIN_BODY_THRESHOLD {
                bbox = Some(match bbox {
                    None => (ix, ix, iy, iy),
                    Some((x0, x1, y0, y1)) => (x0.min(ix), x1.max(ix), y0.min(iy), y1.max(iy)),
                });
            }
        }
    }
    bbox.ok_or_else(|| {
        KwaversError::InvalidInput("CT slice has no non-air head support".to_owned())
    })
}

fn head_centroid(slice: &Array2<f64>) -> Option<(f64, f64)> {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0.0;
    for ((ix, iy), hu) in slice.indexed_iter() {
        if *hu > HU_BRAIN_BODY_THRESHOLD {
            sx += ix as f64;
            sy += iy as f64;
            n += 1.0;
        }
    }
    (n > 0.0).then_some((sx / n, sy / n))
}

/// Convert CT Hounsfield units to sound speed for soft tissue.
///
/// Mast (2000) Biophysical Journal 79:1580-1589: linear fit over the soft-
/// tissue HU range gives `c(HU) ≈ 1524 + 0.68·HU` [m/s].  Clamped to the
/// brain HU range [−20, 120] before interpolation; the extrapolated constant
/// is acceptable for background non-brain tissue since those voxels are outside
/// the active inversion set.  A safety floor/ceiling of [1480, 1620] m/s
/// prevents physically implausible values if the clamp bounds are ever widened.
pub(super) fn soft_tissue_speed(hu: f64) -> f64 {
    (SOUND_SPEED_WATER_37C + 0.68 * hu.clamp(-20.0, 120.0))
        .clamp(SOUND_SPEED_WATER_SIM, SOUND_SPEED_SOFT_TISSUE_MAX)
}
