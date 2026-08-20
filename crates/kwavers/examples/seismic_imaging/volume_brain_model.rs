//! 3-D brain-prior loading and tissue-velocity construction.

use super::seismic_imaging::medium::SkullModel;
use super::{
    seismic_volume_phantom, Array3, BrainPriorMode, C_CSF, C_GRAY, C_WHITE, DX,
    MNI_INNER_SKULL_RADIUS_MM, NX, NY, NZ, R_SKULL_IN, SOUND_SPEED_WATER_SIM,
};
use anyhow::Context as _;
use coeus_core::MoiraiBackend;
use ritk_io::format::nifti::native::NiftiReader as NativeNiftiReader;
use ritk_io::ImageReader;
use std::path::Path;

/// Load and normalize a T1 MRI NIfTI volume.
///
/// The source tensor is transposed from `[Z, Y, X]` to `[X, Y, Z]` and
/// normalized by the 99th percentile of non-zero voxels.
pub(super) fn load_t1_mri(path: &Path) -> anyhow::Result<(Array3<f64>, [f64; 3])> {
    let backend = MoiraiBackend;

    println!("  T1 NIfTI file   : {}", path.display());
    let img = ImageReader::read(&NativeNiftiReader::new(backend), path)
        .with_context(|| format!("T1 NIfTI read failed for '{}'", path.display()))?;

    let [depth, rows, cols] = img.shape();
    let spacing = img.spacing().into_vector().to_array();
    let values = img
        .data_slice()
        .map_err(|e| anyhow::anyhow!("T1 tensor data is not f32: {e:?}"))?;
    anyhow::ensure!(
        values.len() == depth * rows * cols,
        "T1 data length mismatch: got {}, expected {}",
        values.len(),
        depth * rows * cols
    );

    let mut vol = Array3::<f64>::zeros((cols, rows, depth));
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                vol[[x, y, z]] = f64::from(values[z * rows * cols + y * cols + x]).max(0.0);
            }
        }
    }

    let mut nonzero: Vec<f64> = vol.iter().copied().filter(|&v| v > 0.0).collect();
    let p99 = if nonzero.is_empty() {
        1.0
    } else {
        nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((nonzero.len() as f64) * 0.99).floor() as usize;
        nonzero[idx.min(nonzero.len() - 1)].max(1.0)
    };

    for value in vol.iter_mut() {
        *value = (*value / p99).clamp(0.0, 1.0);
    }

    Ok((vol, [spacing[0], spacing[1], spacing[2]]))
}

/// Map normalized T1 intensity to tissue sound speed using Duck (1990) bands.
#[inline]
fn t1_to_velocity(t1_norm: f64) -> f64 {
    if t1_norm > 0.70 {
        C_WHITE
    } else if t1_norm > 0.35 {
        C_GRAY
    } else if t1_norm > 0.05 {
        C_CSF
    } else {
        SOUND_SPEED_WATER_SIM
    }
}

/// Build a T1-derived 3-D brain model while preserving CT skull velocities.
pub(super) fn build_brain_velocity_from_t1(
    skull_phantom: &SkullModel,
    t1: &Array3<f64>,
    t1_spacing: [f64; 3],
) -> Array3<f64> {
    let [t1_nx, t1_ny, t1_nz] = t1.shape();
    let cx_t1 = t1_nx as f64 / 2.0;
    let cy_t1 = t1_ny as f64 / 2.0;
    let cz_t1 = t1_nz as f64 / 2.0;

    let fwi_inner_mm = R_SKULL_IN * DX * 1e3;
    let t1_inner_skull_mm = MNI_INNER_SKULL_RADIUS_MM;
    let fwi_to_t1 = t1_inner_skull_mm / (fwi_inner_mm * t1_spacing[0]);

    let mut model = skull_phantom.acoustic().sound_speed.clone();

    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let dx_fwi = ix as f64 - (NX / 2) as f64;
                let dy_fwi = iy as f64 - (NY / 2) as f64;
                let dz_fwi = iz as f64 - (NZ / 2) as f64;
                let r_3d = (dx_fwi * dx_fwi + dy_fwi * dy_fwi + dz_fwi * dz_fwi).sqrt();
                if r_3d >= R_SKULL_IN {
                    continue;
                }

                let tx = (cx_t1 + dx_fwi * fwi_to_t1).clamp(0.0, t1_nx as f64 - 1.001);
                let ty = (cy_t1 + dy_fwi * fwi_to_t1).clamp(0.0, t1_ny as f64 - 1.001);
                let tz = (cz_t1 + dz_fwi * fwi_to_t1).clamp(0.0, t1_nz as f64 - 1.001);

                let t1_val = seismic_volume_phantom::trilinear_hu(t1, tx, ty, tz);
                model[[ix, iy, iz]] = t1_to_velocity(t1_val);
            }
        }
    }
    model
}

/// Build a deterministic homogeneous-water brain prior without datasets.
fn build_uniform_brain_velocity_3d(skull_phantom: &SkullModel) -> Array3<f64> {
    let mut model = skull_phantom.acoustic().sound_speed.clone();
    let cx = NX as f64 / 2.0;
    let cy = NY as f64 / 2.0;
    let cz = NZ as f64 / 2.0;
    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let dx = ix as f64 - cx;
                let dy = iy as f64 - cy;
                let dz = iz as f64 - cz;
                if (dx * dx + dy * dy + dz * dz).sqrt() < R_SKULL_IN {
                    model[[ix, iy, iz]] = SOUND_SPEED_WATER_SIM;
                }
            }
        }
    }
    model
}

/// Load MNI tissue maps and construct a probability-weighted velocity model.
fn build_brain_velocity_3d(
    skull_phantom: &SkullModel,
    mni_dir: &Path,
) -> anyhow::Result<Array3<f64>> {
    let backend = MoiraiBackend;

    let load = |name: &str| -> anyhow::Result<Array3<f64>> {
        let path = mni_dir.join(name);
        anyhow::ensure!(
            path.exists(),
            "MNI tissue map not found: '{}' — download from {}",
            path.display(),
            "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip"
        );
        let img = ImageReader::read(&NativeNiftiReader::new(backend), &path)
            .with_context(|| format!("NIfTI load failed: '{}'", path.display()))?;
        let [depth, rows, cols] = img.shape();
        let vals = img
            .data_slice()
            .map_err(|e| anyhow::anyhow!("NIfTI data not f32: {e:?}"))?;
        let mut vol = Array3::<f64>::zeros((cols, rows, depth));
        for z in 0..depth {
            for y in 0..rows {
                for x in 0..cols {
                    vol[[x, y, z]] =
                        f64::from(vals[z * rows * cols + y * cols + x]).clamp(0.0, 1.0);
                }
            }
        }
        Ok(vol)
    };

    let gm = load("mni_icbm152_gm_tal_nlin_sym_09c.nii")?;
    let wm = load("mni_icbm152_wm_tal_nlin_sym_09c.nii")?;
    let csf = load("mni_icbm152_csf_tal_nlin_sym_09c.nii")?;

    let [mni_nx, mni_ny, mni_nz] = gm.shape();
    let cx_mni = mni_nx / 2;
    let cy_mni = mni_ny / 2;
    let cz_mni = mni_nz / 2;
    let fwi_inner_mm = R_SKULL_IN * DX * 1e3;
    let fwi_to_mni = MNI_INNER_SKULL_RADIUS_MM / fwi_inner_mm;

    let mut brain_model = skull_phantom.acoustic().sound_speed.clone();
    for iz in 0..NZ {
        for iy in 0..NY {
            for ix in 0..NX {
                let dx_fwi = ix as f64 - (NX / 2) as f64;
                let dy_fwi = iy as f64 - (NY / 2) as f64;
                let dz_fwi = iz as f64 - (NZ / 2) as f64;
                let r3 = (dx_fwi * dx_fwi + dy_fwi * dy_fwi + dz_fwi * dz_fwi).sqrt();
                if r3 >= R_SKULL_IN {
                    continue;
                }

                let mni_x = (cx_mni as f64 + dx_fwi * DX * 1e3 * fwi_to_mni).round() as isize;
                let mni_y = (cy_mni as f64 + dy_fwi * DX * 1e3 * fwi_to_mni).round() as isize;
                let mni_z = (cz_mni as f64 + dz_fwi * DX * 1e3 * fwi_to_mni).round() as isize;
                if mni_x < 0
                    || mni_x >= mni_nx as isize
                    || mni_y < 0
                    || mni_y >= mni_ny as isize
                    || mni_z < 0
                    || mni_z >= mni_nz as isize
                {
                    continue;
                }

                let mx = mni_x as usize;
                let my = mni_y as usize;
                let mz = mni_z as usize;
                let p_gm = gm[[mx, my, mz]];
                let p_wm = wm[[mx, my, mz]];
                let p_csf = csf[[mx, my, mz]];
                let p_rest: f64 = (1.0_f64 - p_gm - p_wm - p_csf).clamp(0.0, 1.0);
                brain_model[[ix, iy, iz]] =
                    p_gm * C_GRAY + p_wm * C_WHITE + p_csf * C_CSF + p_rest * SOUND_SPEED_WATER_SIM;
            }
        }
    }

    Ok(brain_model)
}

/// Select and construct the configured 3-D brain prior.
pub(super) fn build_brain_prior_3d(
    skull_phantom: &SkullModel,
    prior: &BrainPriorMode,
) -> anyhow::Result<Array3<f64>> {
    match prior {
        BrainPriorMode::Uniform => Ok(build_uniform_brain_velocity_3d(skull_phantom)),
        BrainPriorMode::Mni(path) | BrainPriorMode::MniT1 { mni: path, .. } => {
            build_brain_velocity_3d(skull_phantom, path)
        }
        BrainPriorMode::T1(path) => {
            let (t1, spacing) = load_t1_mri(path).with_context(|| {
                format!("explicit T1 prior could not be loaded: {}", path.display())
            })?;
            Ok(build_brain_velocity_from_t1(skull_phantom, &t1, spacing))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{t1_to_velocity, C_CSF, C_GRAY, C_WHITE, SOUND_SPEED_WATER_SIM};

    #[test]
    fn t1_velocity_bands_match_the_declared_contract() {
        assert_eq!(t1_to_velocity(0.70), C_GRAY);
        assert_eq!(t1_to_velocity(0.700_001), C_WHITE);
        assert_eq!(t1_to_velocity(0.35), C_CSF);
        assert_eq!(t1_to_velocity(0.350_001), C_GRAY);
        assert_eq!(t1_to_velocity(0.05), SOUND_SPEED_WATER_SIM);
        assert_eq!(t1_to_velocity(0.050_001), C_CSF);
    }
}
