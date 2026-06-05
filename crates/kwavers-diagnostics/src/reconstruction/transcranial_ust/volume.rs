//! CT volume preparation and CT-to-acoustic property mapping for 3-D inversion.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::numerics::operators::interpolation::trilinear_index_space;
use ndarray::Array3;

use super::{
    config::{SOUND_SPEED_SKULL, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM},
    medium::{soft_tissue_speed, SOFT_TISSUE_ATTENUATION_NP_PER_M_MHZ},
};
use kwavers_core::constants::acoustic_parameters::SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ;
use kwavers_core::constants::ct_acoustics::{HU_BONE_THRESHOLD, HU_BRAIN_BODY_THRESHOLD};

/// CT volume resampled to the isotropic cubic transcranial UST inversion domain.
#[derive(Clone, Debug)]
pub struct CtResampledVolume {
    pub hu: Array3<f64>,
    pub spacing_m: f64,
    pub source_slice_index: usize,
    pub source_volume_index: usize,
}

/// Acoustic fields derived from one CT volume.
#[derive(Clone, Debug)]
pub struct AcousticVolume {
    pub ct_hu: Array3<f64>,
    pub sound_speed_m_s: Array3<f64>,
    pub initial_sound_speed_m_s: Array3<f64>,
    pub attenuation_np_per_m_mhz: Array3<f64>,
    pub brain_mask: Array3<bool>,
    pub skull_mask: Array3<bool>,
    pub spacing_m: f64,
    pub source_slice_index: usize,
    pub source_volume_index: usize,
}

impl AcousticVolume {
    /// Convert resampled CT HU values to 3-D acoustic fields and masks.
    pub fn from_ct_hu(volume: CtResampledVolume) -> KwaversResult<Self> {
        if !volume.spacing_m.is_finite() || volume.spacing_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "spacing_m must be finite and positive".to_owned(),
            ));
        }
        let (nx, ny, nz) = volume.hu.dim();
        if nx < 8 || ny < 8 || nz < 8 {
            return Err(KwaversError::InvalidInput(
                "CT volume must be at least 8 x 8 x 8".to_owned(),
            ));
        }

        let mut sound_speed = Array3::<f64>::zeros((nx, ny, nz));
        let mut initial = Array3::<f64>::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);
        let mut attenuation = Array3::<f64>::zeros((nx, ny, nz));
        let mut brain_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
        let mut skull_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
        let center = head_centroid3(&volume.hu).unwrap_or((
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            (nz - 1) as f64 / 2.0,
        ));
        let radii = (0.40 * nx as f64, 0.40 * ny as f64, 0.45 * nz as f64);

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let hu = volume.hu[[ix, iy, iz]];
                    let skull = hu >= HU_BONE_THRESHOLD;
                    skull_mask[[ix, iy, iz]] = skull;
                    sound_speed[[ix, iy, iz]] = if skull {
                        let phi = (hu / 1000.0).clamp(0.0, 1.0);
                        attenuation[[ix, iy, iz]] = SOFT_TISSUE_ATTENUATION_NP_PER_M_MHZ
                            * (1.0 - phi)
                            + SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ * phi;
                        SOUND_SPEED_WATER_SIM * (1.0 - phi) + SOUND_SPEED_SKULL * phi
                    } else if hu > HU_BRAIN_BODY_THRESHOLD {
                        attenuation[[ix, iy, iz]] = SOFT_TISSUE_ATTENUATION_NP_PER_M_MHZ;
                        soft_tissue_speed(hu)
                    } else {
                        SOUND_SPEED_WATER_SIM
                    };

                    let ellipsoid =
                        normalized_radius(ix as f64, iy as f64, iz as f64, center, radii);
                    let brain_hu = (-20.0..=120.0).contains(&hu);
                    brain_mask[[ix, iy, iz]] = ellipsoid <= 1.0 && brain_hu && !skull;
                    initial[[ix, iy, iz]] = if brain_mask[[ix, iy, iz]] {
                        SOUND_SPEED_TISSUE
                    } else {
                        sound_speed[[ix, iy, iz]]
                    };
                }
            }
        }

        let active = brain_mask.iter().filter(|v| **v).count();
        if active < 8 {
            return Err(KwaversError::InvalidInput(
                "CT-derived 3-D brain mask contains fewer than 8 voxels".to_owned(),
            ));
        }

        Ok(Self {
            ct_hu: volume.hu,
            sound_speed_m_s: sound_speed,
            initial_sound_speed_m_s: initial,
            attenuation_np_per_m_mhz: attenuation,
            brain_mask,
            skull_mask,
            spacing_m: volume.spacing_m,
            source_slice_index: volume.source_slice_index,
            source_volume_index: volume.source_volume_index,
        })
    }
}

/// Crop non-air head support and resample the CT onto an isotropic cube.
pub fn resample_head_volume(
    volume_hu: &Array3<f64>,
    spacing_mm: [f64; 3],
    source_slice_index: usize,
    grid_size: usize,
) -> KwaversResult<CtResampledVolume> {
    if grid_size < 8 {
        return Err(KwaversError::InvalidInput(
            "3-D grid_size must be at least 8".to_owned(),
        ));
    }
    if spacing_mm.iter().any(|spacing| *spacing <= 0.0) {
        return Err(KwaversError::InvalidInput(
            "CT spacing must be positive in x, y, and z".to_owned(),
        ));
    }
    let (nx, ny, nz) = volume_hu.dim();
    if nz == 0 || source_slice_index >= nz {
        return Err(KwaversError::InvalidInput(format!(
            "source_slice_index {source_slice_index} out of bounds for {nz} CT slices"
        )));
    }

    let (x0, x1, y0, y1, z0, z1) = head_bbox3(volume_hu)?;
    let margin_x = ((x1 - x0 + 1) as f64 * 0.08).ceil() as usize;
    let margin_y = ((y1 - y0 + 1) as f64 * 0.08).ceil() as usize;
    let margin_z = ((z1 - z0 + 1) as f64 * 0.08).ceil() as usize;
    let x0 = x0.saturating_sub(margin_x);
    let x1 = (x1 + margin_x).min(nx - 1);
    let y0 = y0.saturating_sub(margin_y);
    let y1 = (y1 + margin_y).min(ny - 1);
    let z0 = z0.saturating_sub(margin_z);
    let z1 = (z1 + margin_z).min(nz - 1);

    let mut out = Array3::<f64>::zeros((grid_size, grid_size, grid_size));
    let sx = (x1 - x0) as f64 / (grid_size - 1) as f64;
    let sy = (y1 - y0) as f64 / (grid_size - 1) as f64;
    let sz = (z1 - z0) as f64 / (grid_size - 1) as f64;
    for ix in 0..grid_size {
        for iy in 0..grid_size {
            for iz in 0..grid_size {
                let x = x0 as f64 + ix as f64 * sx;
                let y = y0 as f64 + iy as f64 * sy;
                let z = z0 as f64 + iz as f64 * sz;
                out[[ix, iy, iz]] = trilinear_index_space(volume_hu, x, y, z);
            }
        }
    }

    let extent_x_m = (x1 - x0 + 1) as f64 * spacing_mm[0] * 1.0e-3;
    let extent_y_m = (y1 - y0 + 1) as f64 * spacing_mm[1] * 1.0e-3;
    let extent_z_m = (z1 - z0 + 1) as f64 * spacing_mm[2] * 1.0e-3;
    let source_volume_index = if sz > 0.0 {
        ((source_slice_index.saturating_sub(z0)) as f64 / sz)
            .round()
            .clamp(0.0, (grid_size - 1) as f64) as usize
    } else {
        grid_size / 2
    };

    Ok(CtResampledVolume {
        hu: out,
        spacing_m: extent_x_m.max(extent_y_m).max(extent_z_m) / grid_size as f64,
        source_slice_index,
        source_volume_index,
    })
}

fn normalized_radius(
    ix: f64,
    iy: f64,
    iz: f64,
    center: (f64, f64, f64),
    radii: (f64, f64, f64),
) -> f64 {
    let dx = (ix - center.0) / radii.0.max(1.0);
    let dy = (iy - center.1) / radii.1.max(1.0);
    let dz = (iz - center.2) / radii.2.max(1.0);
    dx * dx + dy * dy + dz * dz
}

fn head_centroid3(volume: &Array3<f64>) -> Option<(f64, f64, f64)> {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    let mut n = 0.0;
    for ((ix, iy, iz), hu) in volume.indexed_iter() {
        if *hu > HU_BRAIN_BODY_THRESHOLD {
            sx += ix as f64;
            sy += iy as f64;
            sz += iz as f64;
            n += 1.0;
        }
    }
    (n > 0.0).then_some((sx / n, sy / n, sz / n))
}

fn head_bbox3(volume: &Array3<f64>) -> KwaversResult<(usize, usize, usize, usize, usize, usize)> {
    let (nx, ny, nz) = volume.dim();
    let mut bbox: Option<(usize, usize, usize, usize, usize, usize)> = None;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if volume[[ix, iy, iz]] > HU_BRAIN_BODY_THRESHOLD {
                    bbox = Some(match bbox {
                        None => (ix, ix, iy, iy, iz, iz),
                        Some((x0, x1, y0, y1, z0, z1)) => (
                            x0.min(ix),
                            x1.max(ix),
                            y0.min(iy),
                            y1.max(iy),
                            z0.min(iz),
                            z1.max(iz),
                        ),
                    });
                }
            }
        }
    }
    bbox.ok_or_else(|| {
        KwaversError::InvalidInput("CT volume has no non-air head support".to_owned())
    })
}
